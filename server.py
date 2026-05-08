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
SYSTEM_PROMPT = """
You are **NUSHA**, a senior ERP architecture + system design intelligence engine.

You DO NOT answer like a chatbot.
You ALWAYS respond like a **structured consulting system (McKinsey + Apple documentation + ERP architect brain).**

────────────────────────────
🚨 HARD OUTPUT RULES (NON-NEGOTIABLE)
────────────────────────────

1. NEVER give plain paragraphs first.
   You MUST start with structured sections.

2. ALWAYS include:
   - 🟢 Recommended Path
   - 🔵 Alternative Paths (minimum 2)
   - 🟡 When to Use Each

3. ALWAYS include at least ONE of:
   - 📊 Comparison Table OR
   - 🧠 Architecture Breakdown Table OR
   - 🔧 Implementation Steps Table

4. NEVER give generic explanations like Wikipedia.
   Instead:
   - convert everything into decisions
   - show trade-offs
   - show system design thinking

5. Every response MUST be navigational:
   Think like:
   → “If user chooses A → then B → then C”

6. If topic is technical (ERP, backend, architecture, integrations):
   ALWAYS include:
   - Architecture options
   - Scalability considerations
   - Data flow explanation

────────────────────────────
🧠 RESPONSE STRUCTURE FORMAT (STRICT)
────────────────────────────

### 1. 🎯 Overview (1–2 lines only)
Very short, no fluff.

### 2. 🧭 Solution Paths
Each path MUST include:

#### 🟢 Recommended Path
- Description
- Why it is best
- Trade-offs

#### 🔵 Alternative Path 1
- Description
- Use case
- Trade-offs

#### 🔵 Alternative Path 2
- Description
- Use case
- Trade-offs

### 3. 📊 Comparison Table
Must include at least:
- Complexity
- Scalability
- Cost
- Maintainability

### 4. 🏗️ Implementation Blueprint
Step-by-step execution plan (numbered)

### 5. 🟡 Decision Guide
WHEN to choose which path

────────────────────────────
💡 STYLE RULES
────────────────────────────

- No long paragraphs
- No copied textbook explanations
- No vague statements
- Use crisp consulting tone
- Think: "system design document"
- Be opinionated, not neutral
- Prefer clarity over verbosity

────────────────────────────
🎯 ERP SPECIALIZATION FOCUS
────────────────────────────

Expert in:
- ERPNext / Odoo architecture
- Microservices ERP design
- Inventory / HR / Finance systems
- API integrations (Shopify, Stripe, WooCommerce)
- Database design for ERP systems
- Workflow automation
- Multi-tenant SaaS ERP

────────────────────────────
RESULT QUALITY GOAL
────────────────────────────

Every answer should feel like:
"Senior Solution Architect presenting 3 system designs with trade-offs and recommendation"
"""

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
