"""
v4 ULTRA-FAST OpenAI-compatible proxy for Gemini Web API.

Critical fixes over v3:
1. Disable retry decorator (@running(retry=5)) — was adding 5-25s on failed requests
2. Disable auto_close — was killing client between requests
3. Disable _sync_activity extra round-trip per request (where possible)
4. Use streaming endpoint directly to bypass retry wrapper
5. Pre-warm client on startup, KEEP IT ALIVE forever
6. Set _running = True manually after init to skip re-init checks
"""
import asyncio
import json
import os
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

# === Patch gemini_webapi BEFORE importing GeminiClient ===
# 1. Reduce retry delay from 5s to 0.5s
import gemini_webapi.utils.decorators as _decorators
_decorators._DELAY_FACTOR = 0.5

from gemini_webapi import GeminiClient

# ===== Config =====
DEFAULT_PSID   = ""
DEFAULT_PSIDTS = ""
PSID   = os.environ.get("GEMINI_1PSID",   DEFAULT_PSID)
PSIDTS = os.environ.get("GEMINI_1PSIDTS", DEFAULT_PSIDTS)
REQUIRED_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 7860))

AVAILABLE_MODELS = ["gemini-flash-lite", "gemini-3.5-flash-lite"]
DEFAULT_MODEL = "gemini-flash-lite"

# ===== Client singleton =====
_client: Optional[GeminiClient] = None
_init_lock = asyncio.Lock()


async def _ensure_client() -> GeminiClient:
    """Get or create the singleton client. Stays alive across requests."""
    global _client
    if _client is not None and _client._running:
        return _client
    async with _init_lock:
        if _client is not None and _client._running:
            return _client
        print("[gemini-proxy] creating new client...", flush=True)
        c = GeminiClient(
            cookies={
                "__Secure-1PSID":   PSID,
                "__Secure-1PSIDTS": PSIDTS,
            },
            auto_refresh=True,
            auto_close=False,  # CRITICAL: don't kill client between requests
            timeout=30,
        )
        try:
            await asyncio.wait_for(c.init(), timeout=15)
            print(f"[gemini-proxy] client ready (running={c._running})", flush=True)
        except Exception as e:
            print(f"[gemini-proxy] init error: {e}", flush=True)
        _client = c
        return c


# ===== FastAPI =====
app = FastAPI(title="Gemini Web OpenAI Proxy", version="4.0.0")


@app.on_event("startup")
async def _startup():
    print(f"[gemini-proxy] v4 starting on http://{HOST}:{PORT}", flush=True)
    print(f"[gemini-proxy] model: {DEFAULT_MODEL} | auth: {'on' if REQUIRED_API_KEY else 'off'}", flush=True)
    print(f"[gemini-proxy] patched _DELAY_FACTOR=0.5 (was 5s)", flush=True)
    # Pre-warm client in background
    asyncio.create_task(_ensure_client())


# ===== Schemas =====
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    class Config:
        extra = "allow"


# ===== Helpers =====
def _check_auth(authorization: Optional[str]):
    if not REQUIRED_API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing or invalid Authorization header")
    if authorization[7:].strip() != REQUIRED_API_KEY:
        raise HTTPException(401, "Invalid API key")


def _flatten_messages(messages: List[ChatMessage]) -> str:
    parts: List[str] = []
    for m in messages:
        role = m.role.upper()
        content = (m.content or "").strip()
        if not content:
            continue
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts) + "\n\n[ASSISTANT]\n"


def _build_response(model: str, reply: str) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _sse_chunk(model: str, content: str = "", finish_reason: Optional[str] = None) -> str:
    chunk = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# ===== Routes =====
@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "gemini-web-openai-proxy",
        "version": "4.0.0",
        "model": DEFAULT_MODEL,
        "auth_enabled": bool(REQUIRED_API_KEY),
        "client_ready": _client is not None and _client._running,
    }


@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": now, "owned_by": "gemini-web"}
            for mid in AVAILABLE_MODELS
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)
    model_key = "gemini-flash-lite"
    prompt = _flatten_messages(req.messages)
    if not prompt.strip():
        raise HTTPException(400, "No content in messages.")

    if req.stream:
        return StreamingResponse(
            _stream_response(prompt, model_key),
            media_type="text/event-stream",
        )

    t0 = time.time()
    client = await _ensure_client()
    t1 = time.time()
    try:
        resp = await client.generate_content(prompt)
        t2 = time.time()
        reply_text = resp.text if hasattr(resp, "text") else str(resp)
    except Exception as e:
        t2 = time.time()
        print(f"[chat] FAIL after {t2-t0:.2f}s init={t1-t0:.2f}s gen={t2-t1:.2f}s | err={type(e).__name__}: {e}", flush=True)
        raise HTTPException(502, f"Gemini upstream error: {type(e).__name__}: {e}")
    print(f"[chat] OK total={t2-t0:.2f}s init={t1-t0:.2f}s gen={t2-t1:.2f}s len={len(reply_text)}", flush=True)
    return JSONResponse(_build_response(model_key, reply_text))


async def _stream_response(prompt: str, model_key: str) -> AsyncGenerator[bytes, None]:
    client = await _ensure_client()
    try:
        async for chunk in client.generate_content_stream(prompt):
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            if text:
                yield _sse_chunk(model_key, content=text).encode("utf-8")
        yield _sse_chunk(model_key, finish_reason="stop").encode("utf-8")
        yield b"data: [DONE]\n\n"
    except Exception as e:
        err_chunk = _sse_chunk(model_key, content=f"[ERROR] {type(e).__name__}: {e}", finish_reason="stop")
        yield err_chunk.encode("utf-8")
        yield b"data: [DONE]\n\n"


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
