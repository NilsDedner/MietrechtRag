#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Iterable, List

import psycopg2.extras as pgx

from etl.config import PgConfig
from etl.pg_db import pg_connect


def _write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _query_rows(conn, sql: str, params: tuple[Any, ...]) -> List[Dict[str, Any]]:
    with conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export analysis DB results by run_id")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--include-topics", action="store_true")
    ap.add_argument("--include-clusters", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    conn = pg_connect(PgConfig())

    run_rows = _query_rows(
        conn,
        "SELECT run_id, created_at, pipeline, params, metrics FROM analysis_runs WHERE run_id=%s",
        (args.run_id,),
    )
    if not run_rows:
        conn.close()
        raise SystemExit(f"run_id not found: {args.run_id}")

    run = run_rows[0]
    with open(os.path.join(args.out_dir, "run.json"), "w", encoding="utf-8") as fh:
        json.dump(run, fh, ensure_ascii=False, indent=2, default=str)

    include_topics = args.include_topics or (not args.include_topics and not args.include_clusters)
    include_clusters = args.include_clusters or (not args.include_topics and not args.include_clusters)

    if include_topics:
        topics = _query_rows(
            conn,
            "SELECT topic_id, term, weight FROM topic_terms WHERE run_id=%s ORDER BY topic_id, weight DESC",
            (args.run_id,),
        )
        _write_csv(os.path.join(args.out_dir, "topics.csv"), ["topic_id", "term", "weight"], topics)

        case_topics = _query_rows(
            conn,
            "SELECT case_id, topic_id, weight FROM case_topics WHERE run_id=%s ORDER BY case_id, topic_id",
            (args.run_id,),
        )
        _write_csv(os.path.join(args.out_dir, "case_topics.csv"), ["case_id", "topic_id", "weight"], case_topics)

    if include_clusters:
        clusters = _query_rows(
            conn,
            "SELECT case_id, cluster_id, score FROM case_clusters WHERE run_id=%s ORDER BY case_id",
            (args.run_id,),
        )
        if clusters:
            _write_csv(os.path.join(args.out_dir, "clusters.csv"), ["case_id", "cluster_id", "score"], clusters)

    conn.close()
    print(f"export done. run_id={args.run_id} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
