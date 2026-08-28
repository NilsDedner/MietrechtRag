#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import datetime as dt
from typing import Any, Dict, List, Optional

import psycopg2.extras as pgx
from sentence_transformers import SentenceTransformer

from .config import PgConfig
from .pg_db import pg_connect
from .timeutils import isoformat_utc


def vec_to_pgvector_str(v):
    return "[" + ",".join(f"{float(x):.6f}" for x in v.tolist()) + "]"


def main():
    ap = argparse.ArgumentParser(description="Query pgvector index (local embedding) -> top-k chunks")
    ap.add_argument("query", help="Freitext-Frage / Suchanfrage")
    ap.add_argument("--k", type=int, default=8, help="Top-k (Default 8)")
    ap.add_argument("--model", default=os.getenv("RAG_EMBED_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2"), help="Embedding model")
    ap.add_argument("--min-year", type=int, default=None, help="Optional: mind. Entscheidungsjahr")
    ap.add_argument("--max-year", type=int, default=None, help="Optional: max. Entscheidungsjahr")
    ap.add_argument("--only-has-date", action="store_true", help="Nur Fälle mit decision_date")
    args = ap.parse_args()

    model = SentenceTransformer(args.model)
    q_emb = model.encode([args.query], convert_to_numpy=True, normalize_embeddings=True)[0]
    q_vec = vec_to_pgvector_str(q_emb)

    conn = pg_connect(PgConfig())

    where = ["embedding IS NOT NULL"]
    params: List[Any] = [q_vec]

    if args.only_has_date:
        where.append("decision_date IS NOT NULL")

    if args.min_year is not None:
        where.append("decision_date >= %s")
        params.append(dt.date(args.min_year, 1, 1))
    if args.max_year is not None:
        where.append("decision_date < %s")
        params.append(dt.date(args.max_year + 1, 1, 1))

    where_sql = " AND ".join(where)

    # cosine distance via vector_cosine_ops (embedding is normalized => cosine distance ok)
    sql = f"""
    SELECT
      case_id, chunk_id, decision_date, updated_date,
      LEFT(chunk_text, 1200) AS chunk_preview,
      (embedding <=> %s::vector) AS distance
    FROM case_chunks
    WHERE {where_sql}
    ORDER BY embedding <=> %s::vector
    LIMIT {args.k}
    """

    # note: we need q_vec twice for ORDER BY; first placeholder already included
    params2 = params + [q_vec]

    with conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
        cur.execute(sql, params2)
        rows = cur.fetchall()

    conn.close()

    print("\n=== QUERY ===")
    print(args.query)
    print("\n=== TOP RESULTS ===")
    for i, r in enumerate(rows, 1):
        print(f"\n[{i}] case_id={r['case_id']} chunk_id={r['chunk_id']} date={r['decision_date']} dist={r['distance']:.4f}")
        print(r["chunk_preview"])


if __name__ == "__main__":
    main()
