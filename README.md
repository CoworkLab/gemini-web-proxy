# Gemini-Web OpenAI Proxy v3 (OPTIMIZED)

Free Gemini Web → OpenAI-compatible API proxy.

## Optimizations in v3
- Lazy client init (skip blocking init at startup)
- Streaming by default (first token ~1.5s vs ~2.5s before)
- Single working model: `gemini-flash-lite` (only model on Google free tier after Oct 9 2026)
- Reduced timeout (30s → faster failure recovery)
- Background warmup on startup

## Endpoints
- `GET /` — health check
- `GET /v1/models` — list models
- `POST /v1/chat/completions` — OpenAI chat completions (supports `stream: true`)

## Env vars
- `GEMINI_1PSID` — __Secure-1PSID cookie
- `GEMINI_1PSIDTS` — __Secure-1PSIDTS cookie
- `GEMINI_API_KEY` — shared secret for client auth (optional)
- `PORT` — listen port (default 7860)
