from __future__ import annotations

from typing import Any, Dict, Sequence

import psycopg2.extras as pgx

DDL = [
    """
    CREATE TABLE IF NOT EXISTS eval_chunks (
      case_id BIGINT NOT NULL,
      chunk_id INT NOT NULL,
      decision_date DATE,
      chunk_text TEXT NOT NULL,
      embedding vector(384),
      PRIMARY KEY (case_id, chunk_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_questions (
      question_id TEXT PRIMARY KEY,
      corpus_run_id TEXT NOT NULL,
      question TEXT NOT NULL,
      gold_case_id BIGINT NOT NULL,
      gold_chunk_id INT NOT NULL,
      source TEXT NOT NULL,
      meta_json JSONB NOT NULL DEFAULT '{}'::jsonb,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_runs (
      run_id TEXT PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      pipeline TEXT NOT NULL,
      params JSONB NOT NULL DEFAULT '{}'::jsonb,
      metrics JSONB NOT NULL DEFAULT '{}'::jsonb
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_results (
      run_id TEXT NOT NULL,
      question_id TEXT NOT NULL,
      rank INT NOT NULL,
      case_id BIGINT NOT NULL,
      chunk_id INT NOT NULL,
      score DOUBLE PRECISION,
      is_gold BOOLEAN NOT NULL DEFAULT false,
      PRIMARY KEY (run_id, question_id, rank)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_eval_questions_corpus ON eval_questions(corpus_run_id)",
    "CREATE INDEX IF NOT EXISTS idx_eval_results_run ON eval_results(run_id)",
]


def ensure_eval_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        for stmt in DDL:
            cur.execute(stmt)
    conn.commit()


def insert_run(conn, run_id: str, pipeline: str, params: dict, metrics: dict) -> None:
    q = """
    INSERT INTO eval_runs(run_id, pipeline, params, metrics)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (run_id) DO UPDATE SET
      pipeline = EXCLUDED.pipeline,
      params = EXCLUDED.params,
      metrics = EXCLUDED.metrics,
      created_at = now()
    """
    with conn.cursor() as cur:
        cur.execute(q, (run_id, pipeline, pgx.Json(params or {}), pgx.Json(metrics or {})))


def upsert_questions(conn, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    q = """
    INSERT INTO eval_questions(question_id, corpus_run_id, question, gold_case_id, gold_chunk_id, source, meta_json)
    VALUES (%(question_id)s, %(corpus_run_id)s, %(question)s, %(gold_case_id)s, %(gold_chunk_id)s, %(source)s, %(meta_json)s)
    ON CONFLICT (question_id) DO UPDATE SET
      question = EXCLUDED.question,
      gold_case_id = EXCLUDED.gold_case_id,
      gold_chunk_id = EXCLUDED.gold_chunk_id,
      source = EXCLUDED.source,
      meta_json = EXCLUDED.meta_json
    """
    with conn.cursor() as cur:
        pgx.execute_batch(cur, q, rows, page_size=min(len(rows), 500))


def replace_results(conn, run_id: str, rows: Sequence[tuple]) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM eval_results WHERE run_id = %s", (run_id,))
        if not rows:
            return
        q = """
        INSERT INTO eval_results(run_id, question_id, rank, case_id, chunk_id, score, is_gold)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        payload = [(run_id,) + tuple(r) for r in rows]
        pgx.execute_batch(cur, q, payload, page_size=2000)
