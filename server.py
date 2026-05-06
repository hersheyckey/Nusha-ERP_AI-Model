import os
import json
import uuid
import logging
import requests
from dotenv import load_dotenv
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load .env file if present (local development)
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# App
app = FastAPI(title="Nusha AI", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config - MUST be set in environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

if not GROQ_API_KEY:
    logger.error("❌ GROQ_API_KEY environment variable not set. Server will not function.")

# System prompt (structured options focused)
SYSTEM_PROMPT = """You are **Nusha**, a senior ERP & business solutions architect. Your superpower is presenting **structured options, decision paths, and trade-offs**.

**CRITICAL RULES:**
1. For every problem or question, always provide **at least 2-3 viable approaches** or architectures.
2. Use clear headings, numbered lists, tables, and bullet points.
3. When coding or technical, mention multiple implementation strategies with pros/cons.
4. Always include a "Recommended Path" and "Alternative Paths".
5. Structure answers with emojis: 🟢 Recommended, 🔵 Alternative, 🟡 When to choose.
6. Use ### headings for each option and a 📊 comparison table when helpful.

**ERP Expertise:** ERPNext, Odoo, integrations (Shopify, Woo, Stripe), HR, Sales, Finance, Inventory, MRP.
Be concise, actionable, and professional."""

# Session store
sessions: dict[str, list] = {}

class ChatRequest(BaseModel):
    prompt: str
    session_id: str | None = None

# ---------- NEW ROOT ROUTE TO SERVE FRONTEND ----------
@app.get("/")
async def root():
    """Serve the main chat interface (index.html)"""
    # Try static folder first
    if os.path.exists("static/index.html"):
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    # Then try root directory
    elif os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return {"message": "Frontend not found. Please add index.html to static/ or root."}
# ----------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok" if GROQ_API_KEY else "missing_api_key",
        "model": MODEL,
        "provider": "groq",
        "timestamp": datetime.utcnow().isoformat(),
        "sessions": len(sessions),
        "api_key_configured": bool(GROQ_API_KEY)
    }

@app.post("/session")
def create_session():
    sid = str(uuid.uuid4())
    sessions[sid] = []
    logger.info(f"New session: {sid}")
    return {"session_id": sid}

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"cleared": True}

@app.post("/chat-stream")
async def chat_stream_post(req: ChatRequest):
    return await _stream(req.prompt, req.session_id)

@app.get("/chat-stream")
async def chat_stream_get(prompt: str, session_id: str | None = None):
    return await _stream(prompt, session_id)

async def _stream(prompt: str, session_id: str | None):
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured. Set environment variable.")

    history = sessions.get(session_id, []) if session_id else []
    history.append({"role": "user", "content": prompt})

    def generate():
        full_reply = ""
        try:
            response = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *history,
                    ],
                    "stream": True,
                    "temperature": 0.5,
                    "max_tokens": MAX_TOKENS,
                },
                stream=True,
                timeout=75,
            )

            if response.status_code != 200:
                yield f"[Error] Groq API returned {response.status_code}: {response.text}"
                return

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    line = line[6:].strip()
                if line == "[DONE]":
                    break
                try:
                    data = json.loads(line)
                    content = data.get("choices", [{}])[0].get("delta", {}).get("content")
                    if content:
                        full_reply += content
                        yield content
                except Exception:
                    continue

            history.append({"role": "assistant", "content": full_reply})
            if session_id and session_id in sessions:
                sessions[session_id] = history

            logger.info(f"Session={session_id} | response length={len(full_reply)} chars")

        except requests.exceptions.Timeout:
            yield "\n\n[Error] Request timeout. Please retry."
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"\n\n[Error] {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain")

# Optional: serve other static assets (images, css, etc.) from /static/*
if os.path.exists("static"):
    # Mount only for static resources other than index.html (which we already serve)
    # Using a separate path like '/assets' or keep as fallback
    # But to avoid conflict, we mount with a different name - or simply keep as is because our root route will match first.
    # However, if you have CSS files referenced in index.html, they need to be accessible.
    # The simplest is to mount static folder to /static url:
    app.mount("/static", StaticFiles(directory="static"), name="static")
