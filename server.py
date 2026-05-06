import os, json, uuid, logging, requests, jwt
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse

load_dotenv()

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

sessions = {}

# ✅ AUTH FIX
async def get_user(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(401, "Missing token")

    token = authorization.split(" ")[1]

    try:
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False}
        )
        return payload
    except Exception as e:
        logger.error(e)
        raise HTTPException(401, "Invalid token")

# serve frontend
@app.get("/")
def root():
    with open("index.html") as f:
        return HTMLResponse(f.read())

# session
@app.post("/session")
def create_session(user=Depends(get_user)):
    sid = str(uuid.uuid4())
    sessions[sid] = []
    return {"session_id": sid}

# chat
@app.post("/chat-stream")
def chat(req: dict, user=Depends(get_user)):

    prompt = req.get("prompt")
    sid = req.get("session_id")

    history = sessions.get(sid, [])
    history.append({"role":"user","content":prompt})

    def stream():
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":"application/json"
            },
            json={
                "model":"llama-3.3-70b-versatile",
                "messages": history,
                "stream":True
            },
            stream=True
        )

        for line in response.iter_lines():
            if line:
                if line.startswith(b"data: "):
                    data=line[6:]
                    if data==b"[DONE]": break
                    try:
                        j=json.loads(data)
                        content=j["choices"][0]["delta"].get("content")
                        if content:
                            yield content
                    except:
                        pass

    return StreamingResponse(stream(), media_type="text/plain")
