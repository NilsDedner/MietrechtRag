#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime as dt
import json
import os
import time
from typing import Any, Dict, List, Optional

import psycopg2.extras as pgx
import requests
from sentence_transformers import SentenceTransformer

from .config import PgConfig
from .pg_db import pg_connect


def vec_to_pgvector_str(v) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in v.tolist()) + "]"


def _date(d) -> Optional[str]:
    if d is None:
        return None
    return str(d)[:10]


def _ts(t) -> Optional[str]:
    if t is None:
        return None
    try:
        return t.isoformat()
    except Exception:
        return str(t)


def retrieve_chunks(
    conn,
    q_vec: str,
    k: int,
    chunk_chars: int,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    only_has_date: bool = False,
) -> List[Dict[str, Any]]:
    where = ["cc.embedding IS NOT NULL"]
    params: List[Any] = []

    if only_has_date:
        where.append("cc.decision_date IS NOT NULL")

    if min_year is not None:
        where.append("cc.decision_date >= %s")
        params.append(dt.date(min_year, 1, 1))

    if max_year is not None:
        where.append("cc.decision_date < %s")
        params.append(dt.date(max_year + 1, 1, 1))

    where_sql = " AND ".join(where)

    sql = f"""
    SELECT
      cc.case_id,
      cc.chunk_id,
      cc.decision_date,
      cc.updated_date,
      cc.meta_json,
      LEFT(cc.chunk_text, %s) AS chunk_text,
      (cc.embedding <-> %s::vector) AS distance
    FROM case_chunks cc
    WHERE {where_sql}
    ORDER BY cc.embedding <-> %s::vector
    LIMIT %s
    """

    sql_params = [chunk_chars, q_vec] + params + [q_vec, k]

    with conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
        cur.execute(sql, sql_params)
        return list(cur.fetchall())


def format_context(rows: List[Dict[str, Any]], max_context_chars: int) -> str:
    blocks: List[str] = []
    used = 0
    for i, r in enumerate(rows, 1):
        meta = r.get("meta_json")
        ecli = None
        if isinstance(meta, dict):
            ecli = meta.get("ecli")

        header = (
            f"[S{i}] case_id={r.get('case_id')} chunk_id={r.get('chunk_id')} "
            f"decision_date={_date(r.get('decision_date'))} distance={float(r.get('distance') or 0.0):.4f}"
        )
        if ecli:
            header += f" ecli={ecli}"

        body = (r.get("chunk_text") or "").strip()
        block = header + "\n" + body

        if used + len(block) + 2 > max_context_chars:
            remaining = max_context_chars - used - len(header) - 2
            if remaining > 120:
                block = header + "\n" + body[:remaining].rstrip()
                blocks.append(block)
            break

        blocks.append(block)
        used += len(block) + 2

    return "\n\n".join(blocks)


def build_messages(question: str, context: str) -> List[Dict[str, str]]:
    system = (
        "Du bist ein juristischer Assistent für deutsches Mietrecht. "
        "Beantworte ausschließlich auf Basis der bereitgestellten Quellen. "
        "Wenn die Quellen nicht ausreichen, sage das klar. "
        "Zitiere Aussagen mit [S1], [S2], ... und erfinde keine Quellen."
    )
    user = (
        "Frage:\n"
        f"{question}\n\n"
        "Quellenkontext:\n"
        f"{context}\n\n"
        "Antworte präzise in Deutsch. Struktur:\n"
        "1) Kurzantwort\n"
        "2) Begründung mit Quellenhinweisen\n"
        "3) Unsicherheit/Hinweis falls Kontext unvollständig"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_chat_completion(
    api_url: str,
    api_key: Optional[str],
    model: str,
    messages: List[Dict[str, str]],
    timeout: int,
    temperature: float,
    max_retries: int,
    initial_backoff: float,
) -> str:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    backoff = max(0.5, float(initial_backoff))
    retries = max(0, int(max_retries))
    last_err: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            r = requests.post(api_url, headers=headers, json=payload, timeout=timeout)

            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                retry_after = r.headers.get("Retry-After")
                wait_s = backoff
                if retry_after:
                    try:
                        wait_s = max(wait_s, float(retry_after))
                    except Exception:
                        pass
                time.sleep(wait_s)
                backoff = min(backoff * 2.0, 60.0)
                continue

            r.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            last_err = e
            if attempt >= retries:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 60.0)
    else:
        if last_err:
            raise last_err
        raise RuntimeError("LLM request failed without specific error")

    data = r.json()
    choices = data.get("choices") or []
    if not choices:
        return ""

    msg = choices[0].get("message") or {}
    content = msg.get("content")

    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text") or "")
        return "\n".join([p for p in parts if p]).strip()

    return (content or "").strip()


def main():
    ap = argparse.ArgumentParser(description="RAG QA: Query -> Retrieve pgvector -> LLM answer with citations")
    ap.add_argument("question", help="Frage in natürlicher Sprache")
    ap.add_argument("--k", type=int, default=10, help="Top-k Retrieval (Default 10)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
    ap.add_argument("--chunk-chars", type=int, default=1600, help="Max chars pro Chunk im Retrieval (Default 1600)")
    ap.add_argument("--max-context-chars", type=int, default=12000, help="Max chars für LLM-Kontext (Default 12000)")
    ap.add_argument("--min-year", type=int, default=None, help="Filter: min Entscheidungsjahr")
    ap.add_argument("--max-year", type=int, default=None, help="Filter: max Entscheidungsjahr")
    ap.add_argument("--only-has-date", action="store_true", help="Nur Chunks mit decision_date")

    ap.add_argument("--llm-api-url", default=os.getenv("RAG_LLM_API_URL", "https://api.openai.com/v1/chat/completions"), help="Chat Completions URL")
    ap.add_argument("--llm-api-key", default=os.getenv("RAG_LLM_API_KEY"), help="API key (optional for local endpoints)")
    ap.add_argument("--llm-model", default=os.getenv("RAG_LLM_MODEL", "gpt-4o-mini"), help="LLM model id")
    ap.add_argument("--llm-timeout", type=int, default=90, help="LLM request timeout seconds")
    ap.add_argument("--temperature", type=float, default=0.2, help="LLM temperature")
    ap.add_argument("--llm-max-retries", type=int, default=5, help="Retries for LLM 429/5xx/network (Default 5)")
    ap.add_argument("--llm-initial-backoff", type=float, default=2.0, help="Initial retry backoff seconds (Default 2.0)")

    ap.add_argument("--no-generate", action="store_true", help="Nur Retrieval/Context ausgeben, kein LLM Call")
    ap.add_argument("--strict-llm", action="store_true", help="Bei LLM-Fehler mit Exception abbrechen")
    ap.add_argument("--pretty", action="store_true", help="JSON pretty-print")
    ap.add_argument("--out", help="Optional: JSON in Datei schreiben")
    args = ap.parse_args()

    st_model = SentenceTransformer(args.model)
    q_emb = st_model.encode([args.question], convert_to_numpy=True, normalize_embeddings=True)[0]
    q_vec = vec_to_pgvector_str(q_emb)

    conn = pg_connect(PgConfig())
    rows = retrieve_chunks(
        conn=conn,
        q_vec=q_vec,
        k=args.k,
        chunk_chars=args.chunk_chars,
        min_year=args.min_year,
        max_year=args.max_year,
        only_has_date=args.only_has_date,
    )
    conn.close()

    context = format_context(rows, max_context_chars=args.max_context_chars)
    sources: List[Dict[str, Any]] = []
    for i, r in enumerate(rows, 1):
        meta = r.get("meta_json")
        source = {
            "source_id": f"S{i}",
            "case_id": int(r["case_id"]),
            "chunk_id": int(r["chunk_id"]),
            "decision_date": _date(r.get("decision_date")),
            "updated_date": _ts(r.get("updated_date")),
            "distance": float(r.get("distance") or 0.0),
            "ecli": meta.get("ecli") if isinstance(meta, dict) else None,
            "chunk_text": r.get("chunk_text") or "",
        }
        sources.append(source)

    answer = None
    generation_error = None
    if not args.no_generate:
        messages = build_messages(args.question, context)
        try:
            answer = call_chat_completion(
                api_url=args.llm_api_url,
                api_key=args.llm_api_key,
                model=args.llm_model,
                messages=messages,
                timeout=args.llm_timeout,
                temperature=args.temperature,
                max_retries=args.llm_max_retries,
                initial_backoff=args.llm_initial_backoff,
            )
        except Exception as e:
            generation_error = f"{type(e).__name__}: {e}"
            if args.strict_llm:
                raise

    out_obj = {
        "question": args.question,
        "retrieval": {
            "embedding_model": args.model,
            "k": args.k,
            "filters": {
                "only_has_date": bool(args.only_has_date),
                "min_year": args.min_year,
                "max_year": args.max_year,
            },
            "returned": len(rows),
        },
        "generation": {
            "enabled": not args.no_generate,
            "llm_api_url": args.llm_api_url,
            "llm_model": args.llm_model,
            "temperature": args.temperature,
            "llm_max_retries": args.llm_max_retries,
            "llm_initial_backoff": args.llm_initial_backoff,
        },
        "answer": answer,
        "generation_error": generation_error,
        "sources": sources,
        "context": context,
    }

    s = json.dumps(out_obj, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(s)
    else:
        print(s)


if __name__ == "__main__":
    main()
