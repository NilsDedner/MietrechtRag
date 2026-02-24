# etl/load_raw_pg.py
from typing import Dict, Any, List, Optional

import psycopg2.extras as pgx
from tqdm import tqdm

from .dump_reader import iter_dump_cases
from .timeutils import parse_iso


UPSERT_CASE_VALUES = """
INSERT INTO cases_raw (
  id, slug, file_number, date, created_date, updated_date, type, ecli,
  court_json, content_html, raw_json
) VALUES %s
ON CONFLICT (id) DO UPDATE SET
  slug=EXCLUDED.slug,
  file_number=EXCLUDED.file_number,
  date=EXCLUDED.date,
  created_date=EXCLUDED.created_date,
  updated_date=EXCLUDED.updated_date,
  type=EXCLUDED.type,
  ecli=EXCLUDED.ecli,
  court_json=EXCLUDED.court_json,
  content_html=EXCLUDED.content_html,
  raw_json=EXCLUDED.raw_json
"""


def map_case(obj: Dict[str, Any]) -> Dict[str, Any]:
    court = obj.get("court")
    return {
        "id": obj.get("id"),
        "slug": obj.get("slug"),
        "file_number": obj.get("file_number"),
        "date": obj.get("date"),
        "created_date": obj.get("created_date"),
        "updated_date": obj.get("updated_date"),
        "type": obj.get("type"),
        "ecli": obj.get("ecli"),
        "court_json": pgx.Json(court) if court is not None else None,
        "content_html": obj.get("content_html") or obj.get("content") or "",
        "raw_json": pgx.Json(obj),
    }


def _dedup_cases_by_id(objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Dedupliziert innerhalb eines Batches nach 'id', damit ON CONFLICT nicht
    dieselbe Zielzeile mehrfach in einem Statement updaten muss.

    Strategie:
    - Pro id behalten wir den "neuesten" Case anhand updated_date (falls vorhanden),
      sonst gewinnt der zuletzt gelesene.
    """
    by_id: Dict[int, Dict[str, Any]] = {}

    for obj in objs:
        cid = obj.get("id")
        if cid is None:
            # ohne ID können wir nicht upserten -> überspringen
            continue

        prev = by_id.get(cid)
        if prev is None:
            by_id[cid] = obj
            continue

        # Wenn updated_date parsebar ist: neuesten behalten
        u_new = parse_iso(obj.get("updated_date") or "")
        u_old = parse_iso(prev.get("updated_date") or "")
        if u_new >= u_old:
            by_id[cid] = obj

    return list(by_id.values())


def upsert_cases_batch(conn, objs: List[Dict[str, Any]], page_size: int = 1000) -> int:
    """
    Schneller Batch-Upsert via execute_values.
    Zusätzlich deduplizieren wir innerhalb des Batches nach id, um
    Postgres CardinalityViolation zu vermeiden.

    Rückgabe: Anzahl (nach Dedup) upserteter Fälle
    """
    if not objs:
        return 0

    deduped = _dedup_cases_by_id(objs)
    if not deduped:
        return 0

    mapped = [map_case(o) for o in deduped]
    values = [
        (
            r["id"],
            r["slug"],
            r["file_number"],
            r["date"],
            r["created_date"],
            r["updated_date"],
            r["type"],
            r["ecli"],
            r["court_json"],
            r["content_html"],
            r["raw_json"],
        )
        for r in mapped
    ]

    with conn.cursor() as cur:
        pgx.execute_values(
            cur,
            UPSERT_CASE_VALUES,
            values,
            page_size=page_size,
        )

    return len(deduped)


def import_dump_to_db(
    dump_path: str,
    conn,
    limit: Optional[int] = None,
    batch: int = 2000,
) -> int:
    """
    Dump-Import (JSONL(.gz)/SQLite) → Postgres
    - batch: wie viele Fälle pro Upsert-Commit (vor Dedup)
    """
    n = 0
    buf: List[Dict[str, Any]] = []

    for obj in tqdm(iter_dump_cases(dump_path), desc="Dump→PG", unit="case", dynamic_ncols=True):
        buf.append(obj)

        if len(buf) >= batch:
            n += upsert_cases_batch(conn, buf, page_size=batch)
            conn.commit()
            buf.clear()

        if limit and n >= limit:
            break

    if buf:
        n += upsert_cases_batch(conn, buf, page_size=batch)
        conn.commit()
        buf.clear()

    return n
