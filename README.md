# Nusha AI — ERP Intelligence v3.0

Powered by **Claude Sonnet 4** (Anthropic).

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
# Open index.html in browser
```

Get your key: https://console.anthropic.com

## Env Variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | required | Anthropic key |
| `CLAUDE_MODEL` | `claude-sonnet-4-20250514` | Model string |
| `MAX_TOKENS` | `2048` | Max response tokens |

## Deploy (Render/Railway/Fly)
- Build: `pip install -r requirements.txt`
- Start: `uvicorn server:app --host 0.0.0.0 --port $PORT`
- Set `ANTHROPIC_API_KEY` in environment

## Production Checklist
- [ ] Move `ANTHROPIC_API_KEY` to env var (never hardcode)
- [ ] Restrict `allow_origins` in CORS to your domain
- [ ] Add rate limiting (slowapi or nginx)
- [ ] Replace in-memory `sessions` with Redis for multi-worker
- [ ] Add HTTPS via reverse proxy
