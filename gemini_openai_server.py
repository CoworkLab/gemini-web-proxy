"""
OpenAI-compatible proxy for Gemini Web API.

Endpoints:
  GET  /v1/models                  -> list available models
  POST /v1/chat/completions         -> OpenAI chat completions (non-streaming & streaming)
  GET  /                            -> health check

Environment variables (preferred — set these in HF Spaces / Render / Koyeb secrets):
  GEMINI_1PSID       = your __Secure-1PSID cookie value
  GEMINI_1PSIDTS     = your __Secure-1PSIDTS cookie value
  GEMINI_API_KEY     = optional shared API key clients must send as Bearer token
  PORT               = port to listen on (defaults to 7860 for HF Spaces, else 8000)

Usage:
  python gemini_openai_server.py
  curl http://localhost:7860/v1/chat/completions \
       -H "Content-Type: application/json" \
       -H "Authorization: Bearer $GEMINI_API_KEY" \
       -d '{"model":"gemini","messages":[{"role":"user","content":"Hi!"}]}'
"""

import asyncio
import json
import os
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn

from gemini_webapi import GeminiClient

# ===== Configuration =====
DEFAULT_PSID   = ""  # set via env var GEMINI_1PSID
DEFAULT_PSIDTS = ""  # set via env var GEMINI_1PSIDTS

PSID   = os.environ.get("GEMINI_1PSID",   DEFAULT_PSID)
PSIDTS = os.environ.get("GEMINI_1PSIDTS", DEFAULT_PSIDTS)

# API key gate (optional) — set GEMINI_API_KEY env var to require auth from clients
REQUIRED_API_KEY = os.environ.get("GEMINI_API_KEY", "")

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 7860))

# Mapping OpenAI model names -> Gemini internal model names
MODEL_MAP = {
    "gemini":         "gemini-2.5-flash",
    "gemini-flash":   "gemini-2.5-flash",
    "gemini-pro":     "gemini-2.5-pro",
    "gemini-2.5-pro": "gemini-2.5-pro",
}

# ===== Persistence for refreshed cookies =====
COOKIES_FILE = os.environ.get("COOKIES_FILE", "/tmp/gemini_cookies.json")

def _load_cookies():
    default = {
        "__Secure-1PSID":   PSID,
        "__Secure-1PSIDTS": PSIDTS,
    }
    if os.path.exists(COOKIES_FILE):
        try:
            with open(COOKIES_FILE) as f:
                data = json.load(f)
            if data.get("__Secure-1PSID"):
                print(f"[gemini-proxy] loaded persisted cookies from {COOKIES_FILE}")
                return data
        except Exception as e:
            print(f"[gemini-proxy] warning: could not load persisted cookies: {e}")
    return default

def _save_cookies(cookies):
    try:
        with open(COOKIES_FILE, "w") as f:
            json.dump(cookies, f, indent=2)
        print(f"[gemini-proxy] persisted refreshed cookies to {COOKIES_FILE}")
    except Exception as e:
        print(f"[gemini-proxy] warning: could not persist cookies: {e}")

_persisted = _load_cookies()

def _exit_callback(cookies):
    print(f"[gemini-proxy] cookies refreshed; new 1PSIDTS={cookies.get('__Secure-1PSIDTS','?')[:20]}...")
    _save_cookies(cookies)

# ===== Init Gemini client =====
client = GeminiClient(
    cookies=_persisted,
    auto_refresh=True,
    exit_callback=_exit_callback,
    verbose=True,
    timeout=60,
)

app = FastAPI(title="Gemini Web -> OpenAI Proxy", version="1.1.0")


@app.on_event("startup")
async def _startup() -> None:
    print(f"[gemini-proxy] init client...")
    try:
        await client.init()
        print(f"[gemini-proxy] listening on http://{HOST}:{PORT}/v1")
    except Exception as e:
        print(f"[gemini-proxy] init error: {e}")


# ===== Schemas =====
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str = "gemini"
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    class Config:
        extra = "allow"


# ===== Auth gate (optional) =====
def _check_auth(authorization: Optional[str]):
    if not REQUIRED_API_KEY:
        return
    if not authorization:
        raise HTTPException(401, "Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Expected Bearer token")
    token = authorization[7:].strip()
    if token != REQUIRED_API_KEY:
        raise HTTPException(401, "Invalid API key")


# ===== Helpers =====
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
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
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
    return {"status": "ok", "service": "gemini-web-openai-proxy", "version": "1.1.0"}


@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": now, "owned_by": "gemini-web"}
            for mid in MODEL_MAP.keys()
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    authorization: Optional[str] = Header(None),
):
    _check_auth(authorization)

    model_key = req.model if req.model in MODEL_MAP else "gemini"
    gemini_model = MODEL_MAP[model_key]

    prompt = _flatten_messages(req.messages)
    if not prompt.strip():
        raise HTTPException(400, "No content in messages.")

    if req.stream:
        return StreamingResponse(
            _stream_response(prompt, gemini_model, model_key),
            media_type="text/event-stream",
        )

    try:
        resp = await client.generate_content(prompt, model=gemini_model)
        reply_text = resp.text if hasattr(resp, "text") else str(resp)
    except Exception as e:
        raise HTTPException(502, f"Gemini upstream error: {type(e).__name__}: {e}")

    return JSONResponse(_build_response(model_key, reply_text))


async def _stream_response(prompt: str, gemini_model: str, model_key: str) -> AsyncGenerator[bytes, None]:
    try:
        resp = await client.generate_content(prompt, model=gemini_model)
        reply_text = resp.text if hasattr(resp, "text") else str(resp)
    except Exception as e:
        err_chunk = _sse_chunk(model_key, content=f"[ERROR] {type(e).__name__}: {e}", finish_reason="stop")
        yield err_chunk.encode("utf-8")
        yield b"data: [DONE]\n\n"
        return

    yield _sse_chunk(model_key, content=reply_text).encode("utf-8")
    yield _sse_chunk(model_key, finish_reason="stop").encode("utf-8")
    yield b"data: [DONE]\n\n"


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
