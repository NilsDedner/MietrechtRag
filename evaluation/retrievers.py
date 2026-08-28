#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrieval-Varianten für die Evaluation.

Alle Retriever liefern eine absteigend sortierte Liste von Treffern
(case_id, chunk_id, score). Die Scores sind zwischen den Varianten NICHT
vergleichbar; verglichen wird ausschließlich die Reihenfolge.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import psycopg2.extras as pgx

Hit = Tuple[int, int, float]


def vec_to_pgvector_str(v) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in v.tolist()) + "]"


def dense_search(
    conn,
    q_vec: str,
    k: int,
    table: str = "eval_chunks",
    ef_search: Optional[int] = None,
    column: str = "embedding",
) -> List[Hit]:
    """Vektorsuche über den Cosine-Operator (nutzt den HNSW-Index)."""
    with conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
        cur.execute("SELECT '[1]'::vector;")  # laedt das pgvector-Modul, macht hnsw.* als GUC bekannt
        if ef_search is not None:
            cur.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)};")
        cur.execute(
            f"""
            SELECT case_id, chunk_id, 1.0 - ({column} <=> %s::vector) AS score
            FROM {table}
            WHERE {column} IS NOT NULL
            ORDER BY {column} <=> %s::vector
            LIMIT %s
            """,
            (q_vec, q_vec, int(k)),
        )
        rows = cur.fetchall()
    conn.rollback()
    return [(int(r["case_id"]), int(r["chunk_id"]), float(r["score"])) for r in rows]


def lexical_search(
    conn,
    query: str,
    k: int,
    table: str = "eval_chunks",
) -> List[Hit]:
    """Lexikalische Suche über die deutsche Volltextkonfiguration von Postgres.

    plainto_tsquery verknüpft alle Terme mit UND. Bei ausformulierten Fragen
    enthält kein Chunk sämtliche Begriffe, das Ergebnis wäre also stets leer.
    Deshalb wird die Query auf ODER umgeschrieben; ts_rank_cd mit Normalisierung
    32 gewichtet danach die Dichte der Treffer und dämpft lange Chunks.
    """
    with conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
        cur.execute(
            f"""
            SELECT case_id, chunk_id,
                   ts_rank_cd(tsv, q, 32) AS score
            FROM {table},
                 (SELECT replace(plainto_tsquery('german', %s)::text, '&', '|')::tsquery AS q) AS sub
            WHERE tsv @@ q
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, int(k)),
        )
        rows = cur.fetchall()
    conn.rollback()
    return [(int(r["case_id"]), int(r["chunk_id"]), float(r["score"])) for r in rows]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hit]],
    k: int,
    rrf_k: int = 60,
) -> List[Hit]:
    """Fusioniert mehrere Rankings über Reciprocal Rank Fusion (Cormack et al.).

    Score eines Dokuments = Summe über alle Listen von 1 / (rrf_k + Rang).
    Bewusst rangbasiert, damit unvergleichbare Scores nicht normalisiert werden muessen.
    """
    fused: Dict[Tuple[int, int], float] = {}
    for ranking in rankings:
        for rank, (case_id, chunk_id, _score) in enumerate(ranking, start=1):
            key = (case_id, chunk_id)
            fused[key] = fused.get(key, 0.0) + 1.0 / (float(rrf_k) + float(rank))

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[: int(k)]
    return [(case_id, chunk_id, score) for (case_id, chunk_id), score in ordered]


def hybrid_search(
    conn,
    query: str,
    q_vec: str,
    k: int,
    table: str = "eval_chunks",
    ef_search: Optional[int] = None,
    candidate_k: Optional[int] = None,
    rrf_k: int = 60,
    column: str = "embedding",
) -> List[Hit]:
    """Dense und lexikalisch getrennt abfragen, dann per RRF fusionieren."""
    cand = int(candidate_k or max(k * 5, 50))
    dense = dense_search(conn, q_vec, cand, table=table, ef_search=ef_search, column=column)
    lexical = lexical_search(conn, query, cand, table=table)
    return reciprocal_rank_fusion([dense, lexical], k=k, rrf_k=rrf_k)


def fetch_chunk_texts(conn, hits: Sequence[Hit], table: str = "eval_chunks") -> Dict[Tuple[int, int], str]:
    """Holt die Texte zu Treffern nach (case_id, chunk_id)."""
    if not hits:
        return {}
    pairs = [(int(c), int(ch)) for c, ch, _ in hits]
    with conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
        cur.execute(
            f"SELECT case_id, chunk_id, chunk_text FROM {table} "
            "WHERE (case_id, chunk_id) IN %s",
            (tuple(pairs),),
        )
        rows = cur.fetchall()
    conn.rollback()
    return {(int(r["case_id"]), int(r["chunk_id"])): r["chunk_text"] for r in rows}
