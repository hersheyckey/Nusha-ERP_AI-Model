import os
import uuid
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

sessions = {}

# ✅ REAL Supabase auth validation (FIXED)
async def get_current_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(401, "Missing token")

    token = authorization.split(" ")[1]

    res = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={
            "Authorization": f"Bearer {token}",
            "apikey": SUPABASE_ANON_KEY
        }
    )

    if res.status_code != 200:
        raise HTTPException(401, "Invalid session")

    return res.json()


@app.get("/")
def home():
    with open("index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/session")
def create_session(user=Depends(get_current_user)):
    sid = str(uuid.uuid4())
    sessions[sid] = []
    return {"session_id": sid}


@app.post("/chat-stream")
def chat(req: dict, user=Depends(get_current_user)):

    prompt = req.get("prompt")
    sid = req.get("session_id")

    history = sessions.get(sid, [])
    history.append({"role": "user", "content": prompt})

    def stream():
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": history,
                "stream": True
            },
            stream=True
        )

        full = ""

        for line in response.iter_lines():
            if line:
                if line.startswith(b"data: "):
                    data = line[6:]
                    if data == b"[DONE]":
                        break
                    try:
                        import json
                        j = json.loads(data)
                        content = j["choices"][0]["delta"].get("content")
                        if content:
                            full += content
                            yield content
                    except:
                        pass

        history.append({"role": "assistant", "content": full})
        sessions[sid] = history

    return StreamingResponse(stream(), media_type="text/plain")
