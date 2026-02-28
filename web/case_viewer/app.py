#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from html import escape
from typing import Any, Optional

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse


app = FastAPI(title="Case HTML Viewer", version="1.0.0")


def _db_conn():
  try:
    import psycopg2  # type: ignore
  except Exception as exc:
    raise RuntimeError("psycopg2 is not installed. Install project dependencies first.") from exc

    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "mietrecht"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", ""),
        sslmode=os.getenv("PGSSLMODE", "prefer"),
    )


def _layout(body: str, title: str = "Gerichtsurteil Viewer") -> str:
    return f"""
<!doctype html>
<html lang=\"de\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
    <title>{escape(title)}</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
      h1 {{ margin-top: 0; }}
      .muted {{ color: #666; }}
      .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-top: 12px; }}
      .meta {{ display: grid; grid-template-columns: 180px 1fr; row-gap: 6px; column-gap: 12px; }}
      label {{ font-weight: 600; }}
      input[type=number] {{ padding: 8px; width: 220px; }}
      button {{ padding: 8px 12px; cursor: pointer; }}
      .doc {{ margin-top: 16px; border: 1px solid #e5e5e5; border-radius: 8px; padding: 16px; }}
      .warn {{ background: #fff7e6; border: 1px solid #ffd591; padding: 12px; border-radius: 8px; }}
    </style>
  </head>
  <body>
    <h1>Gerichtsurteil Viewer</h1>
    <p class=\"muted\">Suche über <code>cases_raw.id</code> und zeige das strukturierte HTML aus <code>content_html</code>.</p>
    {body}
  </body>
</html>
"""


@app.get("/healthz", response_class=HTMLResponse)
def healthz() -> str:
    return "ok"


@app.get("/", response_class=HTMLResponse)
def index(case_id: Optional[int] = Query(default=None)) -> str:
    form = """
    <form method=\"get\" action=\"/\"> 
      <label for=\"case_id\">Case ID</label><br />
      <input id=\"case_id\" name=\"case_id\" type=\"number\" min=\"1\" required />
      <button type=\"submit\">Laden</button>
    </form>
    """

    if case_id is None:
        return _layout(form)

    row: Optional[tuple[Any, ...]] = None
    try:
        conn = _db_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, slug, file_number, date, court_json, content_html
                FROM cases_raw
                WHERE id = %s
                LIMIT 1
                """,
                (int(case_id),),
            )
            row = cur.fetchone()
        conn.close()
    except Exception as exc:
        return _layout(
            form
            + f"<div class='warn'><strong>DB-Fehler:</strong> {escape(str(exc))}</div>",
            title="DB-Fehler",
        )

    if not row:
        return _layout(
            form
            + f"<div class='warn'>Kein Fall mit ID <strong>{int(case_id)}</strong> gefunden.</div>",
            title="Nicht gefunden",
        )

    cid, slug, file_number, decision_date, court_json, content_html = row
    court_name = None
    if isinstance(court_json, dict):
        court_name = court_json.get("name") or court_json.get("slug")

    meta = f"""
    <div class=\"card\">
      <div class=\"meta\">
        <div><strong>ID</strong></div><div>{escape(str(cid))}</div>
        <div><strong>Slug</strong></div><div>{escape(str(slug or ''))}</div>
        <div><strong>Aktenzeichen</strong></div><div>{escape(str(file_number or ''))}</div>
        <div><strong>Datum</strong></div><div>{escape(str(decision_date or ''))}</div>
        <div><strong>Gericht</strong></div><div>{escape(str(court_name or ''))}</div>
      </div>
    </div>
    """

    html_doc = content_html or "<p><em>Kein content_html vorhanden.</em></p>"
    body = form + meta + f"<div class='doc'>{html_doc}</div>"
    return _layout(body, title=f"Urteil {cid}")
