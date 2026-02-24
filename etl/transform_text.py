#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transform Step: cases_raw -> cases_text

- Extrahiert Text aus HTML (oder plain content)
- Normalisiert Whitespace
- Speichert pro Case genau 1 clean_text + Metadaten
- Inkrementell: verarbeitet nur updated_date > loader_state['transform_last_updated']
- Filtert unsinnige Entscheidungsdaten (z.B. 2029-Ausreißer) in der abgeleiteten Tabelle
"""

import argparse
import datetime as dt
import json
import re
from typing import Any, Dict, List, Optional

import psycopg2.extras as pgx
from bs4 import BeautifulSoup

from .config import PgConfig
from .pg_db import pg_connect, ensure_out_db, db_state_get, db_state_set_many
from .timeutils import parse_iso, isoformat_utc


DDL = [
    """
    CREATE TABLE IF NOT EXISTS cases_text (
      id BIGINT PRIMARY KEY,
      updated_date TIMESTAMPTZ,
      decision_date DATE,
      slug TEXT,
      file_number TEXT,
      court_json JSONB,
      clean_text TEXT,
      meta_json JSONB
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cases_text_updated ON cases_text(updated_date)",
]


UPSERT = """
INSERT INTO cases_text(
  id, updated_date, decision_date, slug, file_number, court_json, clean_text, meta_json
) VALUES (
  %(id)s, %(updated_date)s, %(decision_date)s, %(slug)s, %(file_number)s, %(court_json)s, %(clean_text)s, %(meta_json)s
)
ON CONFLICT (id) DO UPDATE SET
  updated_date=EXCLUDED.updated_date,
  decision_date=EXCLUDED.decision_date,
  slug=EXCLUDED.slug,
  file_number=EXCLUDED.file_number,
  court_json=EXCLUDED.court_json,
  clean_text=EXCLUDED.clean_text,
  meta_json=EXCLUDED.meta_json
"""


WS_RE = re.compile(r"[ \t\r\f\v]+")
NL_RE = re.compile(r"\n{3,}")


def ensure_transform_tables(conn) -> None:
    with conn.cursor() as cur:
        for stmt in DDL:
            cur.execute(stmt)
    conn.commit()


def html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = text.replace("\u00a0", " ")
    text = WS_RE.sub(" ", text)
    text = text.replace(" \n", "\n").replace("\n ", "\n")
    text = NL_RE.sub("\n\n", text)
    return text.strip()


def parse_date_safe(s: Any) -> Optional[dt.date]:
    if not s:
        return None
    try:
        if isinstance(s, dt.date):
            return s
        return dt.date.fromisoformat(str(s)[:10])
    except Exception:
        return None


def is_reasonable_decision_date(d: Optional[dt.date], min_date: dt.date, max_date: dt.date) -> bool:
    if d is None:
        return True
    return (d >= min_date) and (d <= max_date)


def fetch_candidates(conn, since_updated: dt.datetime, batch: int) -> List[Dict[str, Any]]:
    q = """
    SELECT
      id, slug, file_number, date, created_date, updated_date, type, ecli,
      court_json, content_html, raw_json
    FROM cases_raw
    WHERE updated_date > %s
    ORDER BY updated_date ASC
    LIMIT %s
    """
    with conn.cursor(cursor_factory=pgx.RealDictCursor) as cur:
        cur.execute(q, (since_updated, batch))
        return list(cur.fetchall())


def _json_obj(x: Any) -> Any:
    """Ensure raw_json is a python object (dict) if possible."""
    if x is None:
        return {}
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}


def main():
    ap = argparse.ArgumentParser(description="Transform cases_raw -> cases_text (HTML->Text, incremental)")
    ap.add_argument("--batch", type=int, default=2000, help="Batchgröße DB-Lesen/Schreiben (Default 2000)")
    ap.add_argument("--min-date", default="1900-01-01", help="Min. erlaubtes Entscheidungsdatum (YYYY-MM-DD)")
    ap.add_argument(
        "--max-date",
        default=None,
        help="Max. erlaubtes Entscheidungsdatum (YYYY-MM-DD). Default: heute + 365 Tage",
    )
    ap.add_argument("--min-chars", type=int, default=200, help="Minimale Textlänge, sonst skip (Default 200)")
    ap.add_argument("--state-key", default="transform_last_updated", help="loader_state key (Default transform_last_updated)")
    args = ap.parse_args()

    min_date = dt.date.fromisoformat(args.min_date)
    max_date = dt.date.fromisoformat(args.max_date) if args.max_date else (dt.date.today() + dt.timedelta(days=365))

    conn = pg_connect(PgConfig())
    ensure_out_db(conn)
    ensure_transform_tables(conn)

    last = db_state_get(conn, args.state_key)
    since_updated = parse_iso(last) if last else dt.datetime(1900, 1, 1, tzinfo=dt.timezone.utc)

    total = 0
    newest = since_updated

    while True:
        rows = fetch_candidates(conn, since_updated, args.batch)
        if not rows:
            break

        payload: List[Dict[str, Any]] = []
        max_seen_in_batch = newest

        for r in rows:
            cid = r.get("id")
            if cid is None:
                continue

            updated_dt = r.get("updated_date") or since_updated
            if hasattr(updated_dt, "tzinfo") and updated_dt.tzinfo is None:
                updated_dt = updated_dt.replace(tzinfo=dt.timezone.utc)

            if updated_dt > max_seen_in_batch:
                max_seen_in_batch = updated_dt

            decision_date = parse_date_safe(r.get("date"))
            if not is_reasonable_decision_date(decision_date, min_date, max_date):
                continue

            raw = _json_obj(r.get("raw_json"))
            content = r.get("content_html") or ""
            if not content and isinstance(raw, dict):
                content = raw.get("content_html") or raw.get("content") or ""

            clean = html_to_text(content)
            if len(clean) < args.min_chars:
                continue

            # IMPORTANT: wrap dicts into pgx.Json for JSONB columns
            court_val = r.get("court_json")
            court_json = pgx.Json(court_val) if court_val is not None else None

            meta = {
                "type": r.get("type"),
                "ecli": r.get("ecli"),
                "created_date": str(r.get("created_date")) if r.get("created_date") else None,
            }

            payload.append({
                "id": cid,
                "updated_date": updated_dt,
                "decision_date": decision_date,
                "slug": r.get("slug"),
                "file_number": r.get("file_number"),
                "court_json": court_json,
                "clean_text": clean,
                "meta_json": pgx.Json(meta),
            })

        # advance cursor even if payload empty, based on rows processed
        if max_seen_in_batch > newest:
            newest = max_seen_in_batch

        if payload:
            with conn.cursor() as cur:
                pgx.execute_batch(cur, UPSERT, payload, page_size=min(len(payload), 2000))

        db_state_set_many(conn, {args.state_key: isoformat_utc(newest)})
        conn.commit()

        total += len(payload)
        since_updated = newest
        print(f"transform: written_total={total} last_updated={isoformat_utc(newest)} (batch_written={len(payload)})")

    print(f"transform done. total_written={total}")
    conn.close()


if __name__ == "__main__":
    main()
