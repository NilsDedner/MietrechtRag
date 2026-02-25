#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Any, Dict, List

from scipy import sparse
from sklearn.cluster import KMeans
from tqdm import tqdm

from etl.config import PgConfig
from etl.pg_db import ensure_out_db, pg_connect

from .db import ensure_analysis_tables, insert_run, upsert_case_clusters
from .metrics import sampled_silhouette


def _load_docs(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            rows.append(
                {
                    "row_id": int(r["row_id"]),
                    "case_id": int(r["case_id"]),
                }
            )
    rows.sort(key=lambda x: x["row_id"])
    return rows


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case_id", "cluster_id", "score"])
        w.writeheader()
        w.writerows(rows)


def _merge_run_json(path: str, section: str, payload: Dict[str, Any]) -> None:
    obj: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except Exception:
            obj = {}
    obj[section] = payload
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cluster documents and persist assignments")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--db-write", dest="db_write", action="store_true")
    ap.add_argument("--no-db-write", dest="db_write", action="store_false")
    ap.set_defaults(db_write=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--resume", action="store_true", help="Skip step if expected output artifacts already exist")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    args = ap.parse_args()

    t0 = time.perf_counter()

    out_dir = args.out_dir or args.in_dir
    os.makedirs(out_dir, exist_ok=True)

    expected = [
        os.path.join(out_dir, "clusters.csv"),
        os.path.join(out_dir, "run.json"),
    ]
    if args.resume and all(os.path.exists(p) for p in expected):
        print(f"clustering: resume active, artifacts already exist -> skip ({out_dir})")
        return

    docs = _load_docs(os.path.join(args.in_dir, "docs.csv"))

    tfidf_path = os.path.join(args.in_dir, "tfidf.npz")
    counts_path = os.path.join(args.in_dir, "counts.npz")
    if os.path.exists(tfidf_path):
        matrix = sparse.load_npz(tfidf_path)
        matrix_name = "tfidf"
    elif os.path.exists(counts_path):
        matrix = sparse.load_npz(counts_path)
        matrix_name = "counts"
    else:
        raise SystemExit("Need tfidf.npz or counts.npz in --in-dir")

    if matrix.shape[0] != len(docs):
        raise SystemExit("Row mismatch between docs.csv and feature matrix")

    km = KMeans(
        n_clusters=int(args.k),
        random_state=int(args.random_state),
        n_init=10,
        max_iter=300,
    )
    print("clustering: fitting k-means ...")
    labels = km.fit_predict(matrix)
    dists = km.transform(matrix)

    rows: List[Dict[str, Any]] = []
    db_rows: List[tuple[int, int, float]] = []
    for i, d in tqdm(
        enumerate(docs),
        total=len(docs),
        desc="clustering: build cluster rows",
        unit="doc",
        disable=args.no_progress,
    ):
        case_id = int(d["case_id"])
        cluster_id = int(labels[i])
        score = float(dists[i, cluster_id])
        rows.append({"case_id": case_id, "cluster_id": cluster_id, "score": score})
        db_rows.append((case_id, cluster_id, score))

    sil = sampled_silhouette(matrix, labels, max_samples=2000, random_state=int(args.random_state))
    params = {
        "k": int(args.k),
        "random_state": int(args.random_state),
        "feature_matrix": matrix_name,
    }
    metrics = {
        "doc_count": int(matrix.shape[0]),
        "feature_dim": int(matrix.shape[1]),
        "silhouette_cosine_sampled": sil,
    }

    _write_csv(os.path.join(out_dir, "clusters.csv"), rows)
    _merge_run_json(
        os.path.join(out_dir, "run.json"),
        "clustering",
        {"run_id": args.run_id, "pipeline": "clustering", "params": params, "metrics": metrics},
    )

    if args.db_write:
        conn = pg_connect(PgConfig())
        ensure_out_db(conn)
        ensure_analysis_tables(conn)
        insert_run(conn, run_id=args.run_id, pipeline="clustering", params=params, metrics=metrics)
        upsert_case_clusters(conn, run_id=args.run_id, rows=db_rows)
        conn.commit()
        conn.close()

    elapsed = time.perf_counter() - t0
    print(
        f"clustering done. run_id={args.run_id} k={args.k} docs={matrix.shape[0]} "
        f"silhouette={sil if sil is not None else 'n/a'} elapsed_s={elapsed:.2f}"
    )


if __name__ == "__main__":
    main()
