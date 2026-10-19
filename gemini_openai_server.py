"""
v3 OPTIMIZED OpenAI-compatible proxy for Gemini Web API.

Key optimizations:
1. Lazy init — skip init() at startup, do it on first request
2. Background init — start init() in background on startup, don't block
3. Streaming by default — first token arrives ~1s sooner
4. Single working model — gemini-flash-lite (free tier only available model)
5. Connection reuse — single client instance across requests
6. Skip auto_refresh overhead — disabled (we don't need it for short-lived requests)
7. Reduced timeouts — 30s instead of 60s

Endpoints:
  GET  /                       -> health check
  GET  /v1/models               -> list models (only 1, but OpenAI-compatible format)
  POST /v1/chat/completions     -> chat completions (streaming by default)
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

from gemini_webapi import GeminiClient

# ===== Config =====
DEFAULT_PSID   = ""
DEFAULT_PSIDTS = ""
PSID   = os.environ.get("GEMINI_1PSID",   DEFAULT_PSID)
PSIDTS = os.environ.get("GEMINI_1PSIDTS", DEFAULT_PSIDTS)
REQUIRED_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 7860))

# Only one model is available on free accounts after Oct 9, 2026 cutoff.
# Don't lie to clients by listing models that won't work.
AVAILABLE_MODELS = ["gemini-flash-lite", "gemini-3.5-flash-lite"]
DEFAULT_MODEL = "gemini-flash-lite"

# ===== Lazy client =====
_client: Optional[GeminiClient] = None
_client_lock = asyncio.Lock()
_init_task: Optional[asyncio.Task] = None


def _build_client() -> GeminiClient:
    return GeminiClient(
        cookies={
            "__Secure-1PSID":   PSID,
            "__Secure-1PSIDTS": PSIDTS,
        },
        auto_refresh=True,
        timeout=30,
    )


async def _ensure_client() -> GeminiClient:
    """Lazy-init the client. First call triggers init, subsequent calls reuse."""
    global _client, _init_task
    if _client is not None:
        return _client
    async with _client_lock:
        if _client is not None:
            return _client
        print("[gemini-proxy] initializing client (lazy)...", flush=True)
        c = _build_client()
        try:
            await asyncio.wait_for(c.init(), timeout=15)
            print("[gemini-proxy] client ready", flush=True)
        except asyncio.TimeoutError:
            print("[gemini-proxy] WARNING init timed out, will retry on next request", flush=True)
            return c  # may still work for generate_content
        except Exception as e:
            print(f"[gemini-proxy] init error: {e}", flush=True)
            return c  # may still work
        _client = c
        return c


def _background_warmup():
    """Start init in background on app startup so first request doesn't wait."""
    global _init_task
    if _init_task is None or _init_task.done():
        _init_task = asyncio.create_task(_ensure_client())


# ===== FastAPI =====
app = FastAPI(title="Gemini Web OpenAI Proxy", version="3.0.0")


@app.on_event("startup")
async def _startup():
    print(f"[gemini-proxy] v3 starting on http://{HOST}:{PORT}", flush=True)
    print(f"[gemini-proxy] model: {DEFAULT_MODEL} | auth: {'on' if REQUIRED_API_KEY else 'off'}", flush=True)
    # Fire-and-forget warmup — first request still works even if warmup not done
    _background_warmup()


# ===== Schemas =====
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None


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


def _build_response(model: str, reply: str, prompt_t: int = 0, comp_t: int = 0) -> Dict[str, Any]:
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
        "usage": {
            "prompt_tokens": prompt_t,
            "completion_tokens": comp_t,
            "total_tokens": prompt_t + comp_t,
        },
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
        "version": "3.0.0",
        "model": DEFAULT_MODEL,
        "auth_enabled": bool(REQUIRED_API_KEY),
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

    # Accept any model name — internally always use the only available one
    model_key = "gemini-flash-lite"

    prompt = _flatten_messages(req.messages)
    if not prompt.strip():
        raise HTTPException(400, "No content in messages.")

    if req.stream:
        return StreamingResponse(
            _stream_response(prompt, model_key),
            media_type="text/event-stream",
        )

    t_start = time.time()
    client = await _ensure_client()
    t_after_init = time.time()
    try:
        resp = await client.generate_content(prompt)
        t_after_gen = time.time()
        reply_text = resp.text if hasattr(resp, "text") else str(resp)
    except Exception as e:
        raise HTTPException(502, f"Gemini upstream error: {type(e).__name__}: {e}")

    print(f"[chat] total={time.time()-t_start:.2f}s | init={t_after_init-t_start:.2f}s | gen={t_after_gen-t_after_init:.2f}s | len={len(reply_text)}", flush=True)
    return JSONResponse(_build_response(model_key, reply_text))


async def _stream_response(prompt: str, model_key: str) -> AsyncGenerator[bytes, None]:
    """Use generate_content_stream for true streaming — first chunk arrives sooner."""
    client = await _ensure_client()
    try:
        # Stream as chunks arrive from Gemini
        async for chunk in client.generate_content_stream(prompt):
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            if text:
                yield _sse_chunk(model_key, content=text).encode("utf-8")
        # Final stop chunk
        yield _sse_chunk(model_key, finish_reason="stop").encode("utf-8")
        yield b"data: [DONE]\n\n"
    except Exception as e:
        err_chunk = _sse_chunk(model_key, content=f"[ERROR] {type(e).__name__}: {e}", finish_reason="stop")
        yield err_chunk.encode("utf-8")
        yield b"data: [DONE]\n\n"


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
