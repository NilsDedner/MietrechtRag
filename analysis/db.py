from __future__ import annotations

from typing import Any, Iterable, Sequence

import psycopg2.extras as pgx

DDL = [
    """
    CREATE TABLE IF NOT EXISTS analysis_runs (
      run_id TEXT PRIMARY KEY,
      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
      pipeline TEXT NOT NULL,
      params JSONB NOT NULL DEFAULT '{}'::jsonb,
      metrics JSONB NOT NULL DEFAULT '{}'::jsonb
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS topic_terms (
      run_id TEXT NOT NULL,
      topic_id INT NOT NULL,
      term TEXT NOT NULL,
      weight DOUBLE PRECISION NOT NULL,
      PRIMARY KEY (run_id, topic_id, term)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS case_topics (
      run_id TEXT NOT NULL,
      case_id BIGINT NOT NULL,
      topic_id INT NOT NULL,
      weight DOUBLE PRECISION NOT NULL,
      PRIMARY KEY (run_id, case_id, topic_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS case_clusters (
      run_id TEXT NOT NULL,
      case_id BIGINT NOT NULL,
      cluster_id INT NOT NULL,
      score DOUBLE PRECISION NULL,
      PRIMARY KEY (run_id, case_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_case_topics_run_topic ON case_topics(run_id, topic_id)",
    "CREATE INDEX IF NOT EXISTS idx_case_clusters_run_cluster ON case_clusters(run_id, cluster_id)",
]


def ensure_analysis_tables(conn) -> None:
    with conn.cursor() as cur:
        for stmt in DDL:
            cur.execute(stmt)
    conn.commit()


def insert_run(conn, run_id: str, pipeline: str, params: dict, metrics: dict) -> None:
    q = """
    INSERT INTO analysis_runs(run_id, pipeline, params, metrics)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (run_id) DO UPDATE SET
      pipeline = EXCLUDED.pipeline,
      params = EXCLUDED.params,
      metrics = EXCLUDED.metrics
    """
    with conn.cursor() as cur:
        cur.execute(q, (run_id, pipeline, pgx.Json(params or {}), pgx.Json(metrics or {})))


def upsert_topic_terms(conn, run_id: str, rows: Sequence[tuple[int, str, float]]) -> None:
    if not rows:
        return
    q = """
    INSERT INTO topic_terms(run_id, topic_id, term, weight)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (run_id, topic_id, term) DO UPDATE SET
      weight = EXCLUDED.weight
    """
    payload = [(run_id, int(topic_id), term, float(weight)) for topic_id, term, weight in rows]
    with conn.cursor() as cur:
        pgx.execute_batch(cur, q, payload, page_size=min(len(payload), 2000))


def upsert_case_topics(conn, run_id: str, rows: Sequence[tuple[int, int, float]]) -> None:
    if not rows:
        return
    q = """
    INSERT INTO case_topics(run_id, case_id, topic_id, weight)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (run_id, case_id, topic_id) DO UPDATE SET
      weight = EXCLUDED.weight
    """
    payload = [(run_id, int(case_id), int(topic_id), float(weight)) for case_id, topic_id, weight in rows]
    with conn.cursor() as cur:
        pgx.execute_batch(cur, q, payload, page_size=min(len(payload), 2000))


def upsert_case_clusters(conn, run_id: str, rows: Sequence[tuple[int, int, float | None]]) -> None:
    if not rows:
        return
    q = """
    INSERT INTO case_clusters(run_id, case_id, cluster_id, score)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (run_id, case_id) DO UPDATE SET
      cluster_id = EXCLUDED.cluster_id,
      score = EXCLUDED.score
    """
    payload = [(run_id, int(case_id), int(cluster_id), None if score is None else float(score)) for case_id, cluster_id, score in rows]
    with conn.cursor() as cur:
        pgx.execute_batch(cur, q, payload, page_size=min(len(payload), 2000))
