#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding Step: case_chunks -> embeddings in pgvector

- Lädt Chunks inkrementell nach updated_date > loader_state['embed_last_updated']
- Erstellt Spalte embedding VECTOR(dimension), falls nicht vorhanden
- Speichert Embeddings lokal über sentence-transformers
- Optional normalisieren (cosine similarity ist dann nur dot product)
"""

import argparse
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psycopg2.extras as pgx
from sentence_transformers import SentenceTransformer

from .config import PgConfig
from .pg_db import pg_connect, ensure_out_db, db_state_get, db_state_set_many
from .timeutils import parse_iso, isoformat_utc


def ensure_vector_extension(conn) -> None:
    # if user is not superuser this may fail; that's okay, you already enabled it via make vector-enable
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    except Exception:
        conn.rollback()


def column_exists(conn, table: str, column: str) -> bool:
    q = """
    SELECT 1
    FROM information_schema.columns
    WHERE table_name=%s AND column_name=%s
    """
    with conn.cursor() as cur:
        cur.execute(q, (table, column))
        return cur.fetchone() is not None


def ensure_embedding_column(conn, dim: int) -> None:
    ensure_vector_extension(conn)
    if not column_exists(conn, "case_chunks", "embedding"):
        with conn.cursor() as cur:
            cur.execute(f"ALTER TABLE case_chunks ADD COLUMN embedding vector({dim});")
        conn.commit()


def ensure_embedding_index(conn, kind: str = "hnsw") -> None:
    """
    Optional: Creates an ANN index. Safe to call multiple times.
    Requires pgvector >= versions supporting those index types.
    If it fails, we ignore (still works with sequential scan for dev).
    """
    try:
        with conn.cursor() as cur:
            if kind == "ivfflat":
                cur.execute("CREATE INDEX IF NOT EXISTS idx_case_chunks_embedding_ivfflat ON case_chunks USING ivfflat (embedding vector_cosine_ops);")
            else:
                cur.execute("CREATE INDEX IF NOT EXISTS idx_case_chunks_embedding_hnsw ON case_chunks USING hnsw (embedding vector_cosine_ops);")
        conn.commit()
    except Exception:
        conn.rollback()


def vec_to_pgvector_str(v: np.ndarray) -> str:
    # pgvector input format: '[0.1,0.2,...]'
    return "[" + ",".join(f"{float(x):.6f}" for x in v.tolist()) + "]"


def fetch_candidates(conn, since_updated: dt.datetime, batch: int, require_missing: bool) -> List[Dict[str, Any]]:
    base = """
    SELECT case_id, chunk_id, updated_date, chunk_text
    FROM case_chunks
    WHERE updated_date > %s
    """
    if require_missing:
        base += " AND embedding IS NULL"
    base += """
    ORDER BY updated_date ASC
    LIMIT %s
    """
    with conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
        cur.execute(base, (since_updated, batch))
        return list(cur.fetchall())


def upsert_embeddings(conn, rows: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
    q = """
    UPDATE case_chunks
    SET embedding = %s::vector
    WHERE case_id = %s AND chunk_id = %s
    """
    payload = []
    for r, emb in zip(rows, embeddings):
        payload.append((vec_to_pgvector_str(emb), r["case_id"], r["chunk_id"]))
    with conn.cursor() as cur:
        pgx.execute_batch(cur, q, payload, page_size=min(len(payload), 2000))


def main():
    ap = argparse.ArgumentParser(description="Embed case_chunks -> pgvector (local sentence-transformers)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model name")
    ap.add_argument("--batch", type=int, default=256, help="Chunks pro DB-Batch (Default 256)")
    ap.add_argument("--encode-batch", type=int, default=64, help="Batchgröße fürs Encoding (Default 64)")
    ap.add_argument("--normalize", action="store_true", help="Embeddings normalisieren (cosine-friendly)")
    ap.add_argument("--state-key", default="embed_last_updated", help="loader_state key (Default embed_last_updated)")
    ap.add_argument("--only-missing", action="store_true", help="Nur Chunks ohne embedding verarbeiten")
    ap.add_argument("--index", choices=["none", "hnsw", "ivfflat"], default="hnsw", help="Optional ANN index (Default hnsw)")
    args = ap.parse_args()

    conn = pg_connect(PgConfig())
    ensure_out_db(conn)

    # load model once
    model = SentenceTransformer(args.model)
    dim = int(model.get_sentence_embedding_dimension())
    ensure_embedding_column(conn, dim)
    if args.index != "none":
        ensure_embedding_index(conn, kind=args.index)

    last = db_state_get(conn, args.state_key)
    since_updated = parse_iso(last) if last else dt.datetime(1900, 1, 1, tzinfo=dt.timezone.utc)

    total = 0
    newest = since_updated

    while True:
        rows = fetch_candidates(conn, since_updated, args.batch, require_missing=args.only_missing)
        if not rows:
            break

        texts = [(r.get("chunk_text") or "").strip() for r in rows]
        # avoid encoding empty chunks
        idx_map = [i for i, t in enumerate(texts) if t]
        if not idx_map:
            # advance cursor
            for r in rows:
                upd = r.get("updated_date")
                if upd and upd > newest:
                    newest = upd
            db_state_set_many(conn, {args.state_key: isoformat_utc(newest)})
            conn.commit()
            since_updated = newest
            continue

        texts_nonempty = [texts[i] for i in idx_map]
        emb = model.encode(
            texts_nonempty,
            batch_size=args.encode_batch,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=args.normalize,
        ).astype(np.float32)

        # Build aligned embedding array for all rows (skip empties -> keep NULL)
        aligned = [None] * len(rows)
        for j, i in enumerate(idx_map):
            aligned[i] = emb[j]

        # Update only rows we embedded
        rows_to_update = []
        emb_to_update = []
        for r, e in zip(rows, aligned):
            if e is None:
                continue
            rows_to_update.append(r)
            emb_to_update.append(e)

        if rows_to_update:
            upsert_embeddings(conn, rows_to_update, np.vstack(emb_to_update))

        # advance cursor
        for r in rows:
            upd = r.get("updated_date")
            if upd is None:
                continue
            if hasattr(upd, "tzinfo") and upd.tzinfo is None:
                upd = upd.replace(tzinfo=dt.timezone.utc)
            if upd > newest:
                newest = upd

        db_state_set_many(conn, {args.state_key: isoformat_utc(newest)})
        conn.commit()
        total += len(rows_to_update)
        since_updated = newest

        print(f"embed: updated_rows_total={total} last_updated={isoformat_utc(newest)} dim={dim}")

    print(f"embed done. total_rows_embedded={total} dim={dim}")
    conn.close()


if __name__ == "__main__":
    main()
