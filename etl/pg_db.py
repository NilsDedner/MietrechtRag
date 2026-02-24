# etl/pg_db.py
import datetime as dt
from typing import Optional
import psycopg2
import psycopg2.extras as pgx

from .config import PgConfig
from .timeutils import parse_iso

DDL = [
    """
    CREATE TABLE IF NOT EXISTS cases_raw (
        id BIGINT PRIMARY KEY,
        slug TEXT,
        file_number TEXT,
        date DATE,
        created_date TIMESTAMPTZ,
        updated_date TIMESTAMPTZ,
        type TEXT,
        ecli TEXT,
        court_json JSONB,
        content_html TEXT,
        raw_json JSONB
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cases_raw_updated ON cases_raw(updated_date)",
    """
    CREATE TABLE IF NOT EXISTS loader_state (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """
]

STATE_GET = "SELECT value FROM loader_state WHERE key=%s"
STATE_SET = """
INSERT INTO loader_state(key, value)
VALUES (%s, %s)
ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value
"""

def pg_connect(cfg: PgConfig):
    conn = psycopg2.connect(
        host=cfg.host, port=cfg.port, dbname=cfg.dbname,
        user=cfg.user, password=cfg.password, sslmode=cfg.sslmode,
    )
    conn.autocommit = False
    return conn

def ensure_out_db(conn):
    with conn.cursor() as cur:
        for stmt in DDL:
            cur.execute(stmt)
    conn.commit()

def db_get_max_updated(conn) -> Optional[dt.datetime]:
    with conn.cursor() as cur:
        cur.execute("SELECT MAX(updated_date) FROM cases_raw")
        row = cur.fetchone()
        if not row or not row[0]:
            return None
        return row[0].astimezone(dt.timezone.utc) if hasattr(row[0], "tzinfo") else parse_iso(str(row[0]))

def db_state_get(conn, key: str) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute(STATE_GET, (key,))
        row = cur.fetchone()
        return row[0] if row else None

def db_state_set_many(conn, items: dict):
    """WICHTIG: Kein Commit hier – Commit soll zusammen mit Case-Upserts passieren."""
    with conn.cursor() as cur:
        for k, v in items.items():
            cur.execute(STATE_SET, (k, v))
