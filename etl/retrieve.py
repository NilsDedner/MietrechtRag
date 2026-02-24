#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retriever: Textquery -> Top-k Chunks (pgvector) -> JSON Output

- Lokal Embedding via sentence-transformers
- Similarity Search in Postgres (pgvector)
- JSON Ausgabe: ideal für spätere API/Frontend Integration
"""

import argparse
import datetime as dt
import json
from typing import Any, Dict, List, Optional

import psycopg2.extras as pgx
from sentence_transformers import SentenceTransformer

from .config import PgConfig
from .pg_db import pg_connect


def vec_to_pgvector_str(v) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in v.tolist()) + "]"


def _date(d) -> Optional[str]:
    if d is None:
        return None
    return str(d)[:10]


def _ts(t) -> Optional[str]:
    if t is None:
        return None
    # psycopg2 returns datetime; keep ISO-like
    try:
        return t.isoformat()
    except Exception:
        return str(t)


def main():
    ap = argparse.ArgumentParser(description="Retrieve top-k chunks via pgvector (JSON output)")
    ap.add_argument("query", help="Freitext-Suchanfrage")
    ap.add_argument("--k", type=int, default=10, help="Top-k (Default 10)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
    ap.add_argument("--min-year", type=int, default=None, help="Filter: min Entscheidungsjahr")
    ap.add_argument("--max-year", type=int, default=None, help="Filter: max Entscheidungsjahr")
    ap.add_argument("--only-has-date", action="store_true", help="Nur Chunks mit decision_date")
    ap.add_argument("--pretty", action="store_true", help="JSON pretty-print")
    ap.add_argument("--out", help="Optional: JSON in Datei schreiben")
    ap.add_argument("--with-case-text", action="store_true", help="Zusätzlich case clean_text (gekürzt) joinen")
    ap.add_argument("--case-text-chars", type=int, default=2000, help="Länge des case-text Preview (Default 2000)")
    ap.add_argument("--chunk-chars", type=int, default=1200, help="Länge chunk preview (Default 1200)")
    args = ap.parse_args()

    # embed query locally
    model = SentenceTransformer(args.model)
    q_emb = model.encode([args.query], convert_to_numpy=True, normalize_embeddings=True)[0]
    q_vec = vec_to_pgvector_str(q_emb)

    conn = pg_connect(PgConfig())

    # filters
    where = ["cc.embedding IS NOT NULL"]
    params: List[Any] = [q_vec]  # for distance calc

    if args.only_has_date:
        where.append("cc.decision_date IS NOT NULL")

    if args.min_year is not None:
        where.append("cc.decision_date >= %s")
        params.append(dt.date(args.min_year, 1, 1))

    if args.max_year is not None:
        where.append("cc.decision_date < %s")
        params.append(dt.date(args.max_year + 1, 1, 1))

    where_sql = " AND ".join(where)

    join_case = ""
    select_case = ""
    if args.with_case_text:
        join_case = "LEFT JOIN cases_text ct ON ct.id = cc.case_id"
        select_case = f", LEFT(ct.clean_text, {int(args.case_text_chars)}) AS case_text_preview"

    sql = f"""
    SELECT
      cc.case_id,
      cc.chunk_id,
      cc.decision_date,
      cc.updated_date,
      LEFT(cc.chunk_text, {int(args.chunk_chars)}) AS chunk_text_preview,
      (cc.embedding <-> %s::vector) AS distance
      {select_case}
    FROM case_chunks cc
    {join_case}
    WHERE {where_sql}
    ORDER BY cc.embedding <-> %s::vector
    LIMIT {int(args.k)}
    """

    # need q_vec twice (distance + order)
    params2 = params + [q_vec]

    with conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
        cur.execute(sql, params2)
        rows = cur.fetchall()

    conn.close()

    results: List[Dict[str, Any]] = []
    for r in rows:
        item: Dict[str, Any] = {
            "case_id": int(r["case_id"]),
            "chunk_id": int(r["chunk_id"]),
            "decision_date": _date(r.get("decision_date")),
            "updated_date": _ts(r.get("updated_date")),
            "distance": float(r.get("distance")),
            "chunk_text_preview": r.get("chunk_text_preview") or "",
        }
        if args.with_case_text:
            item["case_text_preview"] = r.get("case_text_preview") or ""
        results.append(item)

    out_obj = {
        "query": args.query,
        "model": args.model,
        "k": args.k,
        "filters": {
            "only_has_date": bool(args.only_has_date),
            "min_year": args.min_year,
            "max_year": args.max_year,
        },
        "results": results,
    }

    s = json.dumps(out_obj, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(s)
    else:
        print(s)


if __name__ == "__main__":
    main()
