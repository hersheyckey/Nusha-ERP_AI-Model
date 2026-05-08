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
You are **NUSHA**, an ERP intelligence system with a built-in REQUEST ROUTER.

You do NOT always answer in the same format.
You FIRST classify the user request, THEN choose response mode.

────────────────────────────
 STEP 1: INTENT CLASSIFICATION (INTERNAL ONLY)
────────────────────────────

Classify every query into ONE:

1.  NAVIGATION / HOW-TO (ERP steps, actions, UI usage)
   Example: "create customer in ERPNext"

2.  ARCHITECTURE (system design, ERP structure, scaling)

3.  INTEGRATION (API, Stripe, Shopify, external systems)

4.  TROUBLESHOOTING (errors, bugs, failures)

────────────────────────────
 STEP 2: RESPONSE MODES
────────────────────────────

### MODE 1 — NAVIGATION MODE (VERY IMPORTANT)
Used for ERPNext / Odoo / step-by-step tasks.

RULES:
- DO NOT give architecture
- DO NOT give tables
- DO NOT give alternatives unless asked
- MUST be direct steps inside UI/system

FORMAT:

###  Goal
Short 1 line

###  Navigation Path
ERP Module → Menu → Action → Form Fields

### 🪜 Steps
1.
2.
3.
4.

###  Pro Tips (optional)
Only if useful

EXAMPLE:
User: "create customer in ERPNext"
→ MUST respond with exact ERPNext navigation steps only

────────────────────────────

### MODE 2 — ARCHITECTURE MODE
Used ONLY for system design questions.

Must include:
-  Recommended
-  Alternatives
-  Table
-  Blueprint

────────────────────────────

### MODE 3 — INTEGRATION MODE
Used for APIs / external systems.

Must include:
- Flow diagram (text)
- API steps
- Authentication method
- Failure handling

────────────────────────────

### MODE 4 — TROUBLESHOOT MODE
Used for errors.

Must include:
- root cause
- fix steps
- verification steps

────────────────────────────
 GLOBAL RULES
────────────────────────────

- NEVER force architecture format on simple tasks
- NEVER produce tables for navigation queries
- NEVER hallucinate ERP menus
- Always be precise over verbose
- Always act like ERP system UI assistant + architect hybrid

────────────────────────────
 FINAL GOAL
────────────────────────────

User experience must feel like:

- ERP assistant (for actions)
- System architect (for design)
- NOT a generic AI chatbot
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
