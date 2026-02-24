# etl/dump_reader.py
import gzip, json, sqlite3
from typing import Dict, Iterable, Any, Optional
import datetime as dt

from .timeutils import parse_iso

def detect_dump_type(path: str) -> str:
    p = path.lower()
    if p.endswith(".jsonl") or p.endswith(".ndjson"):
        return "jsonl"
    if p.endswith(".jsonl.gz") or p.endswith(".ndjson.gz") or p.endswith(".gz"):
        return "jsonl_gz"
    if p.endswith(".sqlite") or p.endswith(".db"):
        return "sqlite"
    raise ValueError(f"Unbekanntes Dump-Format: {path}")

def iter_dump_cases(path: str) -> Iterable[Dict[str, Any]]:
    t = detect_dump_type(path)
    if t == "sqlite":
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        try:
            for row in con.execute(
                "SELECT id, slug, court, file_number, date, created_date, "
                "updated_date, type, ecli, content FROM cases"
            ):
                yield dict(row)
            return
        except sqlite3.Error:
            pass
        try:
            for row in con.execute("SELECT data FROM cases"):
                yield json.loads(row["data"])
            return
        except sqlite3.Error as e:
            raise RuntimeError(
                "Dump-SQLite wird nicht erkannt – erwarte Tabelle 'cases' mit Spalten ODER Spalte 'data' (JSON)."
            ) from e
        finally:
            con.close()

    if t == "jsonl":
        with open(path, "rb") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line.decode("utf-8"))

    if t == "jsonl_gz":
        with gzip.open(path, "rb") as fh:
            for raw in fh:
                if raw.strip():
                    yield json.loads(raw.decode("utf-8"))

def max_updated_from_dump(path: str) -> Optional[dt.datetime]:
    latest = None
    for obj in iter_dump_cases(path):
        u = parse_iso(obj.get("updated_date") or obj.get("updated") or obj.get("modified") or "")
        if latest is None or u > latest:
            latest = u
    return latest
