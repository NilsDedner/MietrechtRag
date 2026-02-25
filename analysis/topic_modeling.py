#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, List

import numpy as np
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm

from etl.config import PgConfig
from etl.pg_db import ensure_out_db, pg_connect

from .db import ensure_analysis_tables, insert_run, upsert_case_topics, upsert_topic_terms
from .metrics import topic_diversity


def _load_docs(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            rows.append(
                {
                    "row_id": int(r["row_id"]),
                    "case_id": int(r["case_id"]),
                    "decision_date": r.get("decision_date") or "",
                    "text_len": int(r.get("text_len") or 0),
                }
            )
    rows.sort(key=lambda x: x["row_id"])
    return rows


def _load_vocab(path: str) -> Dict[str, int]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {str(k): int(v) for k, v in raw.items()}


def _write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train LDA topics and persist results")
    ap.add_argument("--run-id", required=True, help="Unique analysis run id")
    ap.add_argument("--in-dir", required=True, help="Input feature dir")
    ap.add_argument("--n-topics", type=int, default=15)
    ap.add_argument("--top-terms", type=int, default=15)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--max-iter", type=int, default=20)
    ap.add_argument("--learning-method", default="batch", choices=["batch", "online"])
    ap.add_argument("--db-write", dest="db_write", action="store_true")
    ap.add_argument("--no-db-write", dest="db_write", action="store_false")
    ap.set_defaults(db_write=True)
    ap.add_argument("--out-dir", default=None, help="Output directory (defaults to --in-dir)")
    ap.add_argument("--resume", action="store_true", help="Skip step if expected output artifacts already exist")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    args = ap.parse_args()

    t0 = time.perf_counter()

    out_dir = args.out_dir or args.in_dir
    os.makedirs(out_dir, exist_ok=True)

    expected = [
        os.path.join(out_dir, "topics.csv"),
        os.path.join(out_dir, "case_topics.csv"),
        os.path.join(out_dir, "run.json"),
    ]
    if args.resume and all(os.path.exists(p) for p in expected):
        print(f"topic_modeling: resume active, artifacts already exist -> skip ({out_dir})")
        return

    docs = _load_docs(os.path.join(args.in_dir, "docs.csv"))
    vocab = _load_vocab(os.path.join(args.in_dir, "vocab.json"))

    counts_path = os.path.join(args.in_dir, "counts.npz")
    if not os.path.exists(counts_path):
        raise SystemExit(f"counts.npz missing at {counts_path}; run analysis.features first")

    counts = sparse.load_npz(counts_path)
    if counts.shape[0] != len(docs):
        raise SystemExit("Row mismatch between docs.csv and counts.npz")

    lda = LatentDirichletAllocation(
        n_components=int(args.n_topics),
        random_state=int(args.random_state),
        max_iter=int(args.max_iter),
        learning_method=str(args.learning_method),
        evaluate_every=-1,
        n_jobs=1,
    )

    print("topic_modeling: fitting LDA ...")
    doc_topic = lda.fit_transform(counts)

    idx_to_term = {idx: term for term, idx in vocab.items()}
    top_terms_rows: List[Dict[str, Any]] = []
    top_terms_db_rows: List[tuple[int, str, float]] = []
    topic_top_terms: List[List[str]] = []

    for topic_id, weights in tqdm(
        enumerate(lda.components_),
        total=lda.components_.shape[0],
        desc="topic_modeling: extract topic terms",
        unit="topic",
        disable=args.no_progress,
    ):
        top_idx = np.argsort(weights)[::-1][: int(args.top_terms)]
        tterms: List[str] = []
        for ix in top_idx:
            term = idx_to_term.get(int(ix), f"_term_{ix}")
            weight = float(weights[ix])
            top_terms_rows.append({"topic_id": topic_id, "term": term, "weight": weight})
            top_terms_db_rows.append((topic_id, term, weight))
            tterms.append(term)
        topic_top_terms.append(tterms)

    case_topic_rows: List[Dict[str, Any]] = []
    case_topic_db_rows: List[tuple[int, int, float]] = []
    for i, d in tqdm(
        enumerate(docs),
        total=len(docs),
        desc="topic_modeling: build case-topic rows",
        unit="doc",
        disable=args.no_progress,
    ):
        case_id = int(d["case_id"])
        probs = doc_topic[i]
        for topic_id, prob in enumerate(probs):
            p = float(prob)
            case_topic_rows.append({"case_id": case_id, "topic_id": int(topic_id), "weight": p})
            case_topic_db_rows.append((case_id, int(topic_id), p))

    perplexity_val = float(lda.perplexity(counts))
    diversity = topic_diversity(topic_top_terms)
    metrics = {
        "perplexity": perplexity_val,
        "topic_diversity": diversity,
        "doc_count": int(counts.shape[0]),
        "vocab_size": int(counts.shape[1]),
    }
    params = {
        "n_topics": int(args.n_topics),
        "top_terms": int(args.top_terms),
        "random_state": int(args.random_state),
        "max_iter": int(args.max_iter),
        "learning_method": str(args.learning_method),
    }

    _write_csv(os.path.join(out_dir, "topics.csv"), ["topic_id", "term", "weight"], top_terms_rows)
    _write_csv(os.path.join(out_dir, "case_topics.csv"), ["case_id", "topic_id", "weight"], case_topic_rows)

    run_obj = {
        "run_id": args.run_id,
        "pipeline": "topic_modeling",
        "params": params,
        "metrics": metrics,
    }
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as fh:
        json.dump(run_obj, fh, ensure_ascii=False, indent=2)

    if args.db_write:
        conn = pg_connect(PgConfig())
        ensure_out_db(conn)
        ensure_analysis_tables(conn)
        insert_run(conn, run_id=args.run_id, pipeline="topic_modeling", params=params, metrics=metrics)
        upsert_topic_terms(conn, run_id=args.run_id, rows=top_terms_db_rows)
        upsert_case_topics(conn, run_id=args.run_id, rows=case_topic_db_rows)
        conn.commit()
        conn.close()

    elapsed = time.perf_counter() - t0
    print(
        f"topic_modeling done. run_id={args.run_id} docs={counts.shape[0]} topics={args.n_topics} "
        f"perplexity={perplexity_val:.3f} diversity={diversity:.3f} elapsed_s={elapsed:.2f}"
    )


if __name__ == "__main__":
    main()
