#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chunk Step: cases_text -> case_chunks

- Teilt clean_text in überlappende Chunks
- Inkrementell: verarbeitet nur updated_date > loader_state['chunk_last_updated']
- Schreibt (case_id, chunk_id) als PK, idempotent
"""

import argparse
import datetime as dt
import re
from typing import Any, Dict, List, Optional

import psycopg2.extras as pgx

from .config import PgConfig
from .pg_db import pg_connect, ensure_out_db, db_state_get, db_state_set_many
from .timeutils import parse_iso, isoformat_utc


DDL = [
    """
    CREATE TABLE IF NOT EXISTS case_chunks (
      case_id BIGINT,
      chunk_id INT,
      updated_date TIMESTAMPTZ,
      decision_date DATE,
      chunk_text TEXT,
      meta_json JSONB,
      PRIMARY KEY (case_id, chunk_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_case_chunks_updated ON case_chunks(updated_date)",
]


UPSERT = """
INSERT INTO case_chunks(
  case_id, chunk_id, updated_date, decision_date, chunk_text, meta_json
) VALUES (
  %(case_id)s, %(chunk_id)s, %(updated_date)s, %(decision_date)s, %(chunk_text)s, %(meta_json)s
)
ON CONFLICT (case_id, chunk_id) DO UPDATE SET
  updated_date=EXCLUDED.updated_date,
  decision_date=EXCLUDED.decision_date,
  chunk_text=EXCLUDED.chunk_text,
  meta_json=EXCLUDED.meta_json
"""


NL_RE = re.compile(r"\n{3,}")


def ensure_chunk_tables(conn) -> None:
    with conn.cursor() as cur:
        for stmt in DDL:
            cur.execute(stmt)
    conn.commit()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Einfaches char-basiertes Chunking mit Overlap."""
    if not text:
        return []
    t = NL_RE.sub("\n\n", text).strip()
    if len(t) <= chunk_size:
        return [t]

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(t):
        end = min(len(t), start + chunk_size)
        c = t[start:end].strip()
        if c:
            chunks.append(c)
        if end >= len(t):
            break
        start += step
    return chunks


def fetch_candidates(conn, since_updated: dt.datetime, batch: int) -> List[Dict[str, Any]]:
    q = """
    SELECT id, updated_date, decision_date, clean_text, meta_json
    FROM cases_text
    WHERE updated_date > %s
    ORDER BY updated_date ASC
    LIMIT %s
    """
    with conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
        cur.execute(q, (since_updated, batch))
        return list(cur.fetchall())


def _wrap_jsonb(val: Any):
    """psycopg2 kann dict/list nicht direkt -> JSONB: wrap with pgx.Json."""
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return pgx.Json(val)
    # falls es schon pgx.Json ist, lassen
    if isinstance(val, pgx.Json):
        return val
    # sonst (z.B. string) geben wir es unverändert weiter
    return val


def main():
    ap = argparse.ArgumentParser(description="Chunk cases_text -> case_chunks (incremental)")
    ap.add_argument("--batch", type=int, default=1000, help="Batchgröße (Default 1000)")
    ap.add_argument("--chunk-size", type=int, default=1200, help="Chunkgröße in Zeichen (Default 1200)")
    ap.add_argument("--overlap", type=int, default=150, help="Overlap in Zeichen (Default 150)")
    ap.add_argument("--state-key", default="chunk_last_updated", help="loader_state key (Default chunk_last_updated)")
    args = ap.parse_args()

    conn = pg_connect(PgConfig())
    ensure_out_db(conn)
    ensure_chunk_tables(conn)

    last = db_state_get(conn, args.state_key)
    since_updated = parse_iso(last) if last else dt.datetime(1900, 1, 1, tzinfo=dt.timezone.utc)

    total_chunks = 0
    newest = since_updated

    while True:
        rows = fetch_candidates(conn, since_updated, args.batch)
        if not rows:
            break

        payload: List[Dict[str, Any]] = []
        max_seen_in_batch = newest

        for r in rows:
            cid = r["id"]
            upd = r.get("updated_date") or since_updated
            if hasattr(upd, "tzinfo") and upd.tzinfo is None:
                upd = upd.replace(tzinfo=dt.timezone.utc)
            if upd > max_seen_in_batch:
                max_seen_in_batch = upd

            chunks = chunk_text(r.get("clean_text") or "", args.chunk_size, args.overlap)
            meta_wrapped = _wrap_jsonb(r.get("meta_json"))

            for i, c in enumerate(chunks):
                payload.append({
                    "case_id": cid,
                    "chunk_id": i,
                    "updated_date": upd,
                    "decision_date": r.get("decision_date"),
                    "chunk_text": c,
                    "meta_json": meta_wrapped,  # ✅ dict/list -> JSONB
                })

        if max_seen_in_batch > newest:
            newest = max_seen_in_batch

        if payload:
            with conn.cursor() as cur:
                pgx.execute_batch(cur, UPSERT, payload, page_size=min(len(payload), 2000))

        db_state_set_many(conn, {args.state_key: isoformat_utc(newest)})
        conn.commit()

        total_chunks += len(payload)
        since_updated = newest
        print(f"chunk: written_chunks_total={total_chunks} last_updated={isoformat_utc(newest)} (batch_written={len(payload)})")

    print(f"chunk done. total_chunks_written={total_chunks}")
    conn.close()


if __name__ == "__main__":
    main()
