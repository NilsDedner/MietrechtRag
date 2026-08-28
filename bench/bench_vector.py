#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark der Vektorsuche in pgvector.

Misst pro Retrieval-Variante Latenz und ANN-Recall gegen die exakte Suche:
  - exact      : erzwungener Seq Scan, cosine (Ground Truth für Recall)
  - l2_operator: '<->' wie im ursprünglichen Code (Index-Opclass passt nicht)
  - hnsw       : '<=>' mit variierendem hnsw.ef_search

Schreibt ein JSON-Artefakt pro Lauf, damit Läufe (z.B. vor/nach DB-Tuning)
vergleichbar bleiben.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import psycopg2.extras as pgx
from sentence_transformers import SentenceTransformer

from etl.config import PgConfig
from etl.pg_db import pg_connect


def vec_to_pgvector_str(v) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in v.tolist()) + "]"


def load_queries(path: str) -> List[str]:
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    if not out:
        raise SystemExit(f"keine Queries in {path}")
    return out


def build_sql(operator: str, subcorpus_run_id: Optional[str]) -> str:
    join = ""
    where = ["cc.embedding IS NOT NULL"]
    if subcorpus_run_id:
        where.append(
            "EXISTS (SELECT 1 FROM case_topics ct "
            "WHERE ct.case_id = cc.case_id AND ct.run_id = %(run_id)s)"
        )
    return f"""
    SELECT cc.case_id, cc.chunk_id
    FROM case_chunks cc
    {join}
    WHERE {' AND '.join(where)}
    ORDER BY cc.embedding {operator} %(qv)s::vector
    LIMIT %(k)s
    """


def run_one(
    conn,
    sql: str,
    params: Dict[str, Any],
    mode: str,
    ef_search: Optional[int],
) -> Tuple[List[Tuple[int, int]], float]:
    """Führt eine Query aus und liefert (Treffer, Laufzeit in ms)."""
    with conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
        # Extension-Modul laden, damit hnsw.* als GUC bekannt ist
        cur.execute("SELECT '[1]'::vector;")

        if mode == "exact":
            cur.execute("SET LOCAL enable_indexscan = off;")
            cur.execute("SET LOCAL enable_bitmapscan = off;")
        elif ef_search is not None:
            cur.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)};")

        t0 = time.perf_counter()
        cur.execute(sql, params)
        rows = cur.fetchall()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # rollback beendet die Transaktion und setzt damit alle SET LOCAL zurueck
    conn.rollback()
    return [(int(r["case_id"]), int(r["chunk_id"])) for r in rows], elapsed_ms


def recall_at_k(got: Sequence[Tuple[int, int]], truth: Sequence[Tuple[int, int]]) -> float:
    if not truth:
        return 0.0
    return len(set(got) & set(truth)) / float(len(truth))


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    p95_idx = max(0, int(round(0.95 * (len(ordered) - 1))))
    return {
        "mean": round(statistics.mean(values), 3),
        "p50": round(statistics.median(values), 3),
        "p95": round(ordered[p95_idx], 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark pgvector retrieval (Latenz + ANN-Recall)")
    ap.add_argument("--run-id", required=True, help="Eindeutige Kennung des Messlaufs")
    ap.add_argument("--queries-file", default="bench/queries_de.txt")
    ap.add_argument("--model", default=os.getenv("RAG_EMBED_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3, help="Messwiederholungen pro Query (Default 3)")
    ap.add_argument("--warmup", action="store_true", help="Einen ungemessenen Aufwärmlauf je Query vorschalten")
    ap.add_argument("--ef-search", default="40,100,200,400", help="Kommaliste für hnsw.ef_search")
    ap.add_argument("--skip-exact", action="store_true", help="Exakte Suche überspringen (dann kein Recall)")
    ap.add_argument("--skip-l2", action="store_true", help="Legacy-'<->'-Variante überspringen")
    ap.add_argument("--subcorpus-run-id", default=None, help="Nur Fälle aus diesem analysis-run_id durchsuchen")
    ap.add_argument("--out-dir", default="artifacts/bench")
    args = ap.parse_args()

    queries = load_queries(args.queries_file)
    ef_values = [int(x) for x in args.ef_search.split(",") if x.strip()]

    print(f"bench_vector: {len(queries)} Queries, k={args.k}, repeats={args.repeats}")
    model = SentenceTransformer(args.model)

    t_enc0 = time.perf_counter()
    embeddings = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
    encode_ms_total = (time.perf_counter() - t_enc0) * 1000.0
    q_vecs = [vec_to_pgvector_str(e) for e in embeddings]

    # bewusst KEIN autocommit: SET LOCAL wirkt nur innerhalb einer Transaktion
    conn = pg_connect(PgConfig())

    sql_cos = build_sql("<=>", args.subcorpus_run_id)
    sql_l2 = build_sql("<->", args.subcorpus_run_id)

    variants: List[Dict[str, Any]] = []
    if not args.skip_exact:
        variants.append({"name": "exact", "mode": "exact", "sql": sql_cos, "ef_search": None})
    if not args.skip_l2:
        variants.append({"name": "l2_operator", "mode": "ann", "sql": sql_l2, "ef_search": None})
    for ef in ef_values:
        variants.append({"name": f"hnsw_ef{ef}", "mode": "ann", "sql": sql_cos, "ef_search": ef})

    truth_per_query: Dict[int, List[Tuple[int, int]]] = {}
    results: List[Dict[str, Any]] = []

    for variant in variants:
        latencies: List[float] = []
        recalls: List[float] = []

        for qi, qv in enumerate(q_vecs):
            params: Dict[str, Any] = {"qv": qv, "k": int(args.k)}
            if args.subcorpus_run_id:
                params["run_id"] = args.subcorpus_run_id

            if args.warmup:
                run_one(conn, variant["sql"], params, variant["mode"], variant["ef_search"])

            hits: List[Tuple[int, int]] = []
            for _ in range(max(1, int(args.repeats))):
                hits, ms = run_one(conn, variant["sql"], params, variant["mode"], variant["ef_search"])
                latencies.append(ms)

            if variant["name"] == "exact":
                truth_per_query[qi] = hits
            elif qi in truth_per_query:
                recalls.append(recall_at_k(hits, truth_per_query[qi]))

        entry: Dict[str, Any] = {
            "variant": variant["name"],
            "ef_search": variant["ef_search"],
            "latency_ms": summarize(latencies),
            "measurements": len(latencies),
        }
        if recalls:
            entry["recall_at_k"] = round(statistics.mean(recalls), 4)

        results.append(entry)
        rec = f" recall@{args.k}={entry['recall_at_k']}" if "recall_at_k" in entry else ""
        print(f"  {variant['name']:<16} p50={entry['latency_ms']['p50']:>9.2f} ms  p95={entry['latency_ms']['p95']:>9.2f} ms{rec}")

    settings: Dict[str, str] = {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT name, setting, unit FROM pg_settings WHERE name = ANY(%s)",
            (["shared_buffers", "work_mem", "effective_cache_size", "maintenance_work_mem",
              "max_parallel_workers_per_gather", "random_page_cost", "effective_io_concurrency"],),
        )
        for name, setting, unit in cur.fetchall():
            settings[name] = f"{setting}{unit or ''}"

        cur.execute("SELECT count(*) FROM case_chunks WHERE embedding IS NOT NULL")
        embedded_chunks = int(cur.fetchone()[0])

    conn.close()

    payload = {
        "run_id": args.run_id,
        "params": {
            "k": args.k,
            "repeats": args.repeats,
            "warmup": bool(args.warmup),
            "queries": len(queries),
            "embedding_model": args.model,
            "ef_search_values": ef_values,
            "subcorpus_run_id": args.subcorpus_run_id,
        },
        "environment": {
            "pg_settings": settings,
            "embedded_chunks": embedded_chunks,
            "query_encode_ms_total": round(encode_ms_total, 2),
        },
        "results": results,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.run_id}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"bench_vector fertig. Artefakt: {out_path}")


if __name__ == "__main__":
    main()
