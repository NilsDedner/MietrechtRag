#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from .config import PgConfig
from .pg_db import pg_connect
from .rag_answer import (
    _date,
    _ts,
    build_messages,
    call_chat_completion,
    format_context,
    retrieve_chunks,
    vec_to_pgvector_str,
)


DEFAULT_MODEL_ID = os.getenv("RAG_API_MODEL_ID", "mietrecht-rag")
PASSTHROUGH_MODEL_ID = os.getenv("RAG_API_PASSTHROUGH_MODEL_ID", "openai-direct")
ENABLE_PASSTHROUGH = os.getenv("RAG_API_ENABLE_PASSTHROUGH", "1") not in ("0", "false", "False")
EMBED_MODEL_NAME = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

UPSTREAM_API_URL = os.getenv("RAG_LLM_API_URL", "https://api.openai.com/v1/chat/completions")
if not os.getenv("RAG_LLM_API_URL"):
    UPSTREAM_API_URL = os.getenv("OPENAI_API_BASE_URL", UPSTREAM_API_URL)

UPSTREAM_API_KEY = os.getenv("RAG_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
UPSTREAM_MODEL = os.getenv("RAG_LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TOP_K = int(os.getenv("RAG_TOP_K", "10"))
MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "12000"))
CHUNK_CHARS = int(os.getenv("RAG_CHUNK_CHARS", "1600"))
TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.2"))
LLM_TIMEOUT = int(os.getenv("RAG_LLM_TIMEOUT", "90"))
LLM_MAX_RETRIES = int(os.getenv("RAG_LLM_MAX_RETRIES", "5"))
LLM_INITIAL_BACKOFF = float(os.getenv("RAG_LLM_INITIAL_BACKOFF", "2.0"))

app = FastAPI(title="Mietrecht RAG OpenAI Adapter", version="0.1.0")
embedder = SentenceTransformer(EMBED_MODEL_NAME)


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    stream: Optional[bool] = False


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text") or ""))
        return "\n".join([x for x in parts if x]).strip()
    return str(content or "")


def _last_user_question(messages: List[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            txt = _content_to_text(m.content).strip()
            if txt:
                return txt
    return ""


def _normalize_messages(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in messages:
        out.append({"role": m.role, "content": _content_to_text(m.content)})
    return out


def _source_list_markdown(rows: List[Dict[str, Any]]) -> str:
    lines = ["", "Quellen:"]
    for i, r in enumerate(rows, 1):
        meta = r.get("meta_json")
        ecli = meta.get("ecli") if isinstance(meta, dict) else None
        lines.append(
            f"- [S{i}] case_id={int(r['case_id'])}, chunk_id={int(r['chunk_id'])}, "
            f"date={_date(r.get('decision_date'))}, dist={float(r.get('distance') or 0.0):.4f}, "
            f"ecli={ecli or '-'}"
        )
    return "\n".join(lines)


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {
        "ok": True,
        "model": DEFAULT_MODEL_ID,
        "passthrough_enabled": ENABLE_PASSTHROUGH,
        "passthrough_model": PASSTHROUGH_MODEL_ID,
        "upstream_api_url": UPSTREAM_API_URL,
        "upstream_model": UPSTREAM_MODEL,
        "upstream_api_key_set": bool(UPSTREAM_API_KEY),
    }


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    now = int(time.time())
    data = [
        {
            "id": DEFAULT_MODEL_ID,
            "object": "model",
            "created": now,
            "owned_by": "mietrecht-rag",
        }
    ]
    if ENABLE_PASSTHROUGH:
        data.append(
            {
                "id": PASSTHROUGH_MODEL_ID,
                "object": "model",
                "created": now,
                "owned_by": "openai-upstream",
            }
        )

    return {
        "object": "list",
        "data": data,
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> Dict[str, Any]:
    requested_model = (req.model or DEFAULT_MODEL_ID).strip()
    use_rag = requested_model == DEFAULT_MODEL_ID
    if not use_rag and not (ENABLE_PASSTHROUGH and requested_model == PASSTHROUGH_MODEL_ID):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{requested_model}'. Available: {DEFAULT_MODEL_ID}" + (f", {PASSTHROUGH_MODEL_ID}" if ENABLE_PASSTHROUGH else ""),
        )

    rows: List[Dict[str, Any]] = []
    if use_rag:
        question = _last_user_question(req.messages)
        if not question:
            raise HTTPException(status_code=400, detail="No user message content found")

        q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]
        q_vec = vec_to_pgvector_str(q_emb)

        conn = pg_connect(PgConfig())
        try:
            rows = retrieve_chunks(
                conn=conn,
                q_vec=q_vec,
                k=TOP_K,
                chunk_chars=CHUNK_CHARS,
            )
        finally:
            conn.close()

        context = format_context(rows, max_context_chars=MAX_CONTEXT_CHARS)
        messages = build_messages(question, context)
    else:
        messages = _normalize_messages(req.messages)

    try:
        answer = call_chat_completion(
            api_url=UPSTREAM_API_URL,
            api_key=UPSTREAM_API_KEY,
            model=UPSTREAM_MODEL,
            messages=messages,
            timeout=LLM_TIMEOUT,
            temperature=req.temperature if req.temperature is not None else TEMPERATURE,
            max_retries=LLM_MAX_RETRIES,
            initial_backoff=LLM_INITIAL_BACKOFF,
        )
    except Exception as e:
        detail = f"Upstream LLM error: {type(e).__name__}: {e}"
        resp = getattr(e, "response", None)
        if resp is not None:
            body = (resp.text or "")[:1500]
            detail = f"{detail}; status={resp.status_code}; body={body}"
        raise HTTPException(status_code=502, detail=detail)

    final_text = (answer or "").strip()
    if use_rag:
        final_text = final_text + "\n" + _source_list_markdown(rows)

    now = int(time.time())

    if req.stream:
        def event_stream():
            chunk = {
                "id": f"chatcmpl-rag-{now}",
                "object": "chat.completion.chunk",
                "created": now,
                "model": requested_model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": final_text},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    resp = {
        "id": f"chatcmpl-rag-{now}",
        "object": "chat.completion",
        "created": now,
        "model": requested_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": final_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

    if use_rag:
        resp["rag"] = {
            "retrieved": len(rows),
            "top_k": TOP_K,
            "upstream_model": UPSTREAM_MODEL,
            "sources": [
                {
                    "source_id": f"S{i}",
                    "case_id": int(r["case_id"]),
                    "chunk_id": int(r["chunk_id"]),
                    "decision_date": _date(r.get("decision_date")),
                    "updated_date": _ts(r.get("updated_date")),
                    "distance": float(r.get("distance") or 0.0),
                    "ecli": (r.get("meta_json") or {}).get("ecli") if isinstance(r.get("meta_json"), dict) else None,
                }
                for i, r in enumerate(rows, 1)
            ],
        }
    else:
        resp["routing"] = {
            "mode": "passthrough",
            "upstream_model": UPSTREAM_MODEL,
        }

    return resp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenAI-compatible API adapter for Mietrecht RAG")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (Default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8010, help="Bind port (Default 8010)")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
