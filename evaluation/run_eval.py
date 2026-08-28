#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Führt eine Retrieval-Variante über den Goldstandard aus und berechnet Metriken.

Beispiel:
    python -m evaluation.run_eval --run-id eval_dense_k10 --variant dense --k 10
    python -m evaluation.run_eval --run-id eval_hybrid_k10 --variant hybrid --k 10
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple

import psycopg2.extras as pgx
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from etl.config import PgConfig
from etl.pg_db import pg_connect

from .db import ensure_eval_tables, insert_run, replace_results
from .metrics import aggregate, evaluate_hits
from .retrievers import dense_search, hybrid_search, lexical_search, vec_to_pgvector_str


def load_questions(conn, corpus_run_id: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    sql = """
    SELECT question_id, question, gold_case_id, gold_chunk_id, source
    FROM eval_questions
    WHERE corpus_run_id = %s
    ORDER BY question_id
    """
    params: List[Any] = [corpus_run_id]
    if limit:
        sql += " LIMIT %s"
        params.append(int(limit))

    with conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        raise SystemExit(f"keine Fragen für corpus_run_id={corpus_run_id} (erst goldstandard laufen lassen)")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval-Variante gegen den Goldstandard evaluieren")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--variant", required=True, choices=["dense", "lexical", "hybrid"])
    ap.add_argument("--corpus-run-id", default="analysis_night_20260225_mietrecht_v9")
    ap.add_argument("--table", default="eval_chunks")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef-search", type=int, default=None, help="hnsw.ef_search für dense/hybrid")
    ap.add_argument("--candidate-k", type=int, default=None, help="Kandidatentiefe je Teilretriever im Hybrid")
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--model", default=os.getenv("RAG_EMBED_MODEL_PATH", "sentence-transformers/all-MiniLM-L6-v2"))
    ap.add_argument("--embedding-column", default="embedding", help="Vektorspalte in der Tabelle")
    ap.add_argument("--query-prefix", default="", help="Prefix für die Frage (e5: 'query: ')")
    ap.add_argument("--limit", type=int, default=None, help="nur die ersten N Fragen (Testlauf)")
    ap.add_argument("--out-dir", default="artifacts/eval")
    ap.add_argument("--no-db-write", dest="db_write", action="store_false")
    ap.set_defaults(db_write=True)
    args = ap.parse_args()

    t0 = time.perf_counter()
    conn = pg_connect(PgConfig())
    ensure_eval_tables(conn)
    questions = load_questions(conn, args.corpus_run_id, args.limit)
    print(f"run_eval: {len(questions)} Fragen, Variante={args.variant}, k={args.k}")

    q_vecs: List[str] = []
    encode_ms = 0.0
    if args.variant in ("dense", "hybrid"):
        model = SentenceTransformer(args.model)
        t_enc = time.perf_counter()
        embeddings = model.encode(
            [args.query_prefix + q["question"] for q in questions],
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
        )
        encode_ms = (time.perf_counter() - t_enc) * 1000.0
        q_vecs = [vec_to_pgvector_str(e) for e in embeddings]

    per_question: List[Dict[str, float]] = []
    latencies: List[float] = []
    result_rows: List[Tuple[Any, ...]] = []
    detail: List[Dict[str, Any]] = []

    for i, q in enumerate(tqdm(questions, unit="frage")):
        gold = (int(q["gold_case_id"]), int(q["gold_chunk_id"]))

        t_q = time.perf_counter()
        if args.variant == "dense":
            hits = dense_search(conn, q_vecs[i], args.k, table=args.table, ef_search=args.ef_search,
                                column=args.embedding_column)
        elif args.variant == "lexical":
            hits = lexical_search(conn, q["question"], args.k, table=args.table)
        else:
            hits = hybrid_search(
                conn,
                q["question"],
                q_vecs[i],
                args.k,
                table=args.table,
                ef_search=args.ef_search,
                candidate_k=args.candidate_k,
                rrf_k=args.rrf_k,
                column=args.embedding_column,
            )
        latencies.append((time.perf_counter() - t_q) * 1000.0)

        scores = evaluate_hits(hits, gold)
        per_question.append(scores)
        detail.append({
            "question_id": q["question_id"],
            "question": q["question"],
            "gold": list(gold),
            "metrics": scores,
        })

        for rank, (case_id, chunk_id, score) in enumerate(hits, start=1):
            result_rows.append((
                q["question_id"], rank, case_id, chunk_id, float(score),
                (case_id, chunk_id) == gold,
            ))

    metrics = aggregate(per_question)
    metrics["latency_p50_ms"] = round(statistics.median(latencies), 2)
    metrics["latency_mean_ms"] = round(statistics.mean(latencies), 2)
    metrics["questions"] = len(questions)

    params = {
        "variant": args.variant,
        "corpus_run_id": args.corpus_run_id,
        "table": args.table,
        "k": args.k,
        "ef_search": args.ef_search,
        "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k,
        "embedding_model": args.model,
        "embedding_column": args.embedding_column,
        "query_prefix": args.query_prefix,
        "query_encode_ms_total": round(encode_ms, 2),
    }

    if args.db_write:
        insert_run(conn, run_id=args.run_id, pipeline="retrieval_eval", params=params, metrics=metrics)
        replace_results(conn, args.run_id, result_rows)
        conn.commit()
    conn.close()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.run_id}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"run_id": args.run_id, "params": params, "metrics": metrics, "per_question": detail},
                  fh, ensure_ascii=False, indent=2)

    print(f"\n=== {args.run_id} ({args.variant}) ===")
    for key in ("hit@1", "hit@3", "hit@5", "hit@10", "mrr", "ndcg@10", "case_hit@10", "case_mrr"):
        if key in metrics:
            print(f"  {key:<12} {metrics[key]:.4f}")
    print(f"  {'latenz p50':<12} {metrics['latency_p50_ms']:.2f} ms")
    print(f"Artefakt: {out_path}  (elapsed_s={time.perf_counter() - t0:.1f})")


if __name__ == "__main__":
    main()
