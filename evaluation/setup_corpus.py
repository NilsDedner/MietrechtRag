#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baut den Evaluationskorpus als eigene Tabelle.

Kopiert alle Chunks der Fälle eines Topic-Runs (Standard: das Referenzmodell aus
Projektarbeit 1) nach eval_chunks und legt darauf einen HNSW- und einen
GIN-Volltextindex an. Dadurch laufen Qualitätsexperimente auf einem kleinen,
kontrollierten Korpus, unabhängig von der 5-Mio-Chunk-Tabelle.
"""

from __future__ import annotations

import argparse
import time

from etl.config import PgConfig
from etl.pg_db import pg_connect

from .db import ensure_eval_tables


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluationskorpus aus einem Topic-Run aufbauen")
    ap.add_argument("--corpus-run-id", default="analysis_night_20260225_mietrecht_v9",
                    help="analysis_runs.run_id, dessen Fälle den Korpus bilden")
    ap.add_argument("--min-chars", type=int, default=200, help="Mindestlänge eines Chunks")
    ap.add_argument("--rebuild", action="store_true", help="eval_chunks vorher leeren")
    ap.add_argument("--skip-indexes", action="store_true", help="Indexaufbau überspringen")
    args = ap.parse_args()

    t0 = time.perf_counter()
    conn = pg_connect(PgConfig())
    ensure_eval_tables(conn)

    with conn.cursor() as cur:
        if args.rebuild:
            print("setup_corpus: leere eval_chunks ...")
            cur.execute("TRUNCATE eval_chunks")
            conn.commit()

        print(f"setup_corpus: kopiere Chunks für corpus_run_id={args.corpus_run_id} ...")
        cur.execute(
            """
            INSERT INTO eval_chunks (case_id, chunk_id, decision_date, chunk_text, embedding)
            SELECT cc.case_id, cc.chunk_id, cc.decision_date, cc.chunk_text, cc.embedding
            FROM case_chunks cc
            WHERE cc.embedding IS NOT NULL
              AND length(cc.chunk_text) >= %s
              AND cc.case_id IN (SELECT DISTINCT case_id FROM case_topics WHERE run_id = %s)
            ON CONFLICT (case_id, chunk_id) DO NOTHING
            """,
            (int(args.min_chars), args.corpus_run_id),
        )
        copied = cur.rowcount
        conn.commit()
        print(f"setup_corpus: {copied} Chunks eingefügt")

        if not args.skip_indexes:
            print("setup_corpus: baue Indexe (HNSW cosine, GIN Volltext deutsch) ...")
            # Der Container hat nur das Docker-Default von 64 MB unter /dev/shm.
            # Parallele Indexbauer fordern darüber Shared Memory an und scheitern
            # mit DiskFull, deshalb hier bewusst seriell bauen.
            cur.execute("SET max_parallel_maintenance_workers = 0")
            cur.execute("SET maintenance_work_mem = '512MB'")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_chunks_hnsw "
                "ON eval_chunks USING hnsw (embedding vector_cosine_ops)"
            )
            conn.commit()
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_chunks_fts "
                "ON eval_chunks USING gin (to_tsvector('german', chunk_text))"
            )
            conn.commit()

        cur.execute("ANALYZE eval_chunks")
        conn.commit()

        cur.execute("SELECT count(*), count(DISTINCT case_id) FROM eval_chunks")
        n_chunks, n_cases = cur.fetchone()
        cur.execute("SELECT pg_size_pretty(pg_total_relation_size('eval_chunks'))")
        size = cur.fetchone()[0]

    conn.close()
    print(f"setup_corpus fertig. chunks={n_chunks} faelle={n_cases} groesse={size} "
          f"elapsed_s={time.perf_counter() - t0:.1f}")


if __name__ == "__main__":
    main()
