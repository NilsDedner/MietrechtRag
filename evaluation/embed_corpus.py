#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embeddet eval_chunks mit einem zweiten Modell in eine eigene Spalte.

Damit lassen sich Modelle auf identischem Korpus und identischem Goldstandard
vergleichen, ohne die Produktivtabelle case_chunks anzufassen.

Beispiel:
    python -m evaluation.embed_corpus \
        --model intfloat/multilingual-e5-base \
        --column embedding_e5 --passage-prefix "passage: "
"""

from __future__ import annotations

import argparse
import time
from typing import List, Tuple

import numpy as np
import psycopg2.extras as pgx
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from etl.config import PgConfig
from etl.pg_db import pg_connect


def vec_to_pgvector_str(v: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in v.tolist()) + "]"


def ensure_column(conn, column: str, dim: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name='eval_chunks' AND column_name=%s",
            (column,),
        )
        if cur.fetchone() is None:
            cur.execute(f"ALTER TABLE eval_chunks ADD COLUMN {column} vector({dim})")
    conn.commit()


def fetch_batch(conn, column: str, batch: int) -> List[Tuple[int, int, str]]:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT case_id, chunk_id, chunk_text
            FROM eval_chunks
            WHERE {column} IS NULL
            ORDER BY case_id, chunk_id
            LIMIT %s
            """,
            (int(batch),),
        )
        return [(int(a), int(b), c or "") for a, b, c in cur.fetchall()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Zweites Embedding-Modell auf eval_chunks anwenden")
    ap.add_argument("--model", required=True)
    ap.add_argument("--column", required=True, help="Zielspalte, z.B. embedding_e5")
    ap.add_argument("--passage-prefix", default="", help="Prefix für Dokumente (e5: 'passage: ')")
    ap.add_argument("--batch", type=int, default=2000, help="Zeilen pro DB-Runde")
    ap.add_argument("--encode-batch", type=int, default=64)
    ap.add_argument("--max-seq-length", type=int, default=None, help="Kürzt lange Chunks (spart CPU-Zeit)")
    ap.add_argument("--skip-index", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    model = SentenceTransformer(args.model)
    if args.max_seq_length:
        model.max_seq_length = int(args.max_seq_length)
    dim = int(model.get_sentence_embedding_dimension())
    print(f"embed_corpus: Modell={args.model} dim={dim} max_seq={model.max_seq_length}")

    conn = pg_connect(PgConfig())
    ensure_column(conn, args.column, dim)

    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM eval_chunks WHERE {args.column} IS NULL")
        todo = int(cur.fetchone()[0])
    print(f"embed_corpus: {todo} Chunks offen")

    done = 0
    with tqdm(total=todo, unit="chunk") as bar:
        while True:
            rows = fetch_batch(conn, args.column, args.batch)
            if not rows:
                break

            texts = [args.passage_prefix + t for _, _, t in rows]
            emb = model.encode(
                texts,
                batch_size=args.encode_batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)

            payload = [
                (vec_to_pgvector_str(e), case_id, chunk_id)
                for (case_id, chunk_id, _), e in zip(rows, emb)
            ]
            with conn.cursor() as cur:
                pgx.execute_batch(
                    cur,
                    f"UPDATE eval_chunks SET {args.column} = %s::vector WHERE case_id=%s AND chunk_id=%s",
                    payload,
                    page_size=1000,
                )
            conn.commit()
            done += len(rows)
            bar.update(len(rows))

    if not args.skip_index:
        print("embed_corpus: baue HNSW-Index ...")
        with conn.cursor() as cur:
            # seriell bauen, /dev/shm im Container ist auf 64 MB begrenzt
            cur.execute("SET max_parallel_maintenance_workers = 0")
            cur.execute("SET maintenance_work_mem = '512MB'")
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_eval_chunks_{args.column}_hnsw "
                f"ON eval_chunks USING hnsw ({args.column} vector_cosine_ops)"
            )
        conn.commit()

    conn.close()
    elapsed = time.perf_counter() - t0
    rate = done / elapsed if elapsed > 0 else 0.0
    print(f"embed_corpus fertig. chunks={done} elapsed_s={elapsed:.1f} rate={rate:.1f}/s")


if __name__ == "__main__":
    main()
