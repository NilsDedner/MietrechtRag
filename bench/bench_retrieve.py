#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any, List

import psycopg2.extras as pgx
from sentence_transformers import SentenceTransformer

from etl.config import PgConfig
from etl.pg_db import pg_connect


def vec_to_pgvector_str(v) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in v.tolist()) + "]"


def run_query(conn, q_vec: str, k: int) -> int:
    sql = """
    SELECT case_id, chunk_id
    FROM case_chunks
    WHERE embedding IS NOT NULL
    ORDER BY embedding <-> %s::vector
    LIMIT %s
    """
    with conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
        cur.execute(sql, (q_vec, int(k)))
        rows = cur.fetchall()
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark retrieval latency over a query list")
    ap.add_argument("--queries", nargs="*", default=[
        "Wann ist eine Eigenbedarfskündigung wirksam?",
        "Welche Anforderungen gelten für Mieterhöhungen?",
        "Wann ist eine Betriebskostenabrechnung formell wirksam?",
    ])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--rounds", type=int, default=1, help="Repeat complete query set N times")
    args = ap.parse_args()

    model = SentenceTransformer(args.model)
    conn = pg_connect(PgConfig())

    timings_ms: List[float] = []
    returned_total = 0

    try:
        for _ in range(max(1, int(args.rounds))):
            for q in args.queries:
                t0 = time.perf_counter()
                emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
                q_vec = vec_to_pgvector_str(emb)
                n = run_query(conn, q_vec, args.k)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                timings_ms.append(elapsed_ms)
                returned_total += n
                print(f"query_ms={elapsed_ms:.2f} returned={n} q={q}")
    finally:
        conn.close()

    if not timings_ms:
        raise SystemExit("No timings collected")

    print("=== bench_retrieve ===")
    print(f"queries={len(args.queries)} rounds={args.rounds} total_calls={len(timings_ms)}")
    print(f"avg_ms={statistics.mean(timings_ms):.2f}")
    print(f"p50_ms={statistics.median(timings_ms):.2f}")
    print(f"min_ms={min(timings_ms):.2f}")
    print(f"max_ms={max(timings_ms):.2f}")
    print(f"returned_total={returned_total}")


if __name__ == "__main__":
    main()
