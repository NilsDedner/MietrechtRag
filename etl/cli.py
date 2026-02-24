# etl/cli.py
import argparse
import datetime as dt
import json
import os
import sys
from typing import List, Dict, Any, Optional

from .config import PgConfig, LoaderConfig
from .pg_db import pg_connect, ensure_out_db, db_get_max_updated
from .dump_reader import max_updated_from_dump
from .timeutils import parse_iso, isoformat_utc
from .load_raw_pg import import_dump_to_db, upsert_cases_batch
from .api_cases import fetch_cases_since


def write_ndjson(path: str, objs: List[Dict[str, Any]]) -> int:
    if not path or not objs:
        return 0
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for obj in objs:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return len(objs)


def determine_since(conn, dump: Optional[str], since_arg: Optional[str]) -> dt.datetime:
    if since_arg:
        s = parse_iso(since_arg)
        if s != dt.datetime.min.replace(tzinfo=dt.timezone.utc):
            print(f"[1/5] Stichtag per --since: {isoformat_utc(s)}")
            return s
        print("WARN: --since unlesbar; ignoriere Parameter.")

    db_latest = db_get_max_updated(conn)
    if db_latest:
        print(f"[1/5] Stichtag aus Ziel-DB: {isoformat_utc(db_latest)}")
        return db_latest

    if dump:
        print(f"[1/5] Bestimme Stichtag aus Dump: {dump}")
        dump_latest = max_updated_from_dump(dump)
        s = dump_latest or dt.datetime(1900, 1, 1, tzinfo=dt.timezone.utc)
        print(f"[1/5] Stichtag (updated_date > since): {isoformat_utc(s)}")
        return s

    s = dt.datetime(1900, 1, 1, tzinfo=dt.timezone.utc)
    print(f"[1/5] Stichtag: {isoformat_utc(s)}")
    return s


def main():
    ap = argparse.ArgumentParser(description="Open Legal Data – Incremental Loader (V4, PostgreSQL) – fast batches")
    ap.add_argument("--dump", help="Pfad zum OLDP-Dump (JSONL(.gz) oder SQLite)")
    ap.add_argument("--import-dump", action="store_true", help="Dump in Postgres importieren (Erstbefüllung)")
    ap.add_argument("--import-limit", type=int, help="Max. Fälle beim Dump-Import (Debug/Test)")
    ap.add_argument("--dump-batch", type=int, default=2000, help="Batchgröße beim Dump-Import (Default 2000)")

    ap.add_argument("--out-ndjson", help="Optional: zusätzlich NDJSON anhängen (Delta) – Pfad")
    ap.add_argument("--since", help="ISO-Zeitstempel überschreibt Ermittlung aus DB/Dump")
    ap.add_argument("--page-size", type=int, default=200, help="API page_size (Default 200)")
    ap.add_argument("--server-filter", action="store_true", help="Serverseitige Datumsfilter versuchen (falls unterstützt)")
    ap.add_argument("--checkpoint", help="Pfad zur Resume-Datei (JSON). Optional zusätzlich zum DB-Resume")
    ap.add_argument("--max-pages", type=int, help="Optional: harte Seitenbegrenzung (Debug/Tests)")
    ap.add_argument("--no-progress", action="store_true", help="Progress-Bar abschalten")
    ap.add_argument("--full-initial-load", action="store_true", help="Alles laden (ignoriert since) – für Erstbefüllung")
    ap.add_argument("--no-ordering", action="store_true", help="Kein ordering=-updated_date setzen")

    # Performance: DB-Write batching
    ap.add_argument("--write-batch", type=int, default=1000, help="Batchgröße für DB-Upserts aus API (Default 1000)")

    args = ap.parse_args()

    # DB
    conn = pg_connect(PgConfig())
    ensure_out_db(conn)

    # Stichtag bestimmen
    since = determine_since(conn, args.dump, args.since)

    # Optional: Dump→DB importieren
    if args.import_dump:
        if not args.dump:
            print("ERROR: --import-dump erfordert --dump <Pfad>")
            sys.exit(2)
        print(f"[2/5] Importiere Dump in Postgres: {args.dump}")
        imported = import_dump_to_db(
            args.dump,
            conn,
            limit=args.import_limit,
            batch=args.dump_batch,
        )
        print(f"[2/5] Dump-Import abgeschlossen: {imported} Fälle.")

    lc = LoaderConfig()
    print(f"[3/5] Ziehe {'ALLE' if args.full_initial_load else 'neue/aktualisierte'} Fälle aus der API "
          f"(page_size={args.page_size}, write_batch={args.write_batch}) …")

    buf: List[Dict[str, Any]] = []
    ndjson_buf: List[Dict[str, Any]] = []
    total_written = 0

    def flush():
        nonlocal total_written, buf, ndjson_buf
        if not buf:
            return

        # 1) Cases upserten (Batch)
        wrote = upsert_cases_batch(conn, buf, page_size=args.write_batch)

        # 2) Optional NDJSON
        if args.out_ndjson:
            ndjson_buf.extend(buf)
            if len(ndjson_buf) >= 500:
                write_ndjson(args.out_ndjson, ndjson_buf)
                ndjson_buf.clear()

        # 3) EIN Commit: enthält Case-Upserts + loader_state-Updates aus fetch_cases_since()
        conn.commit()

        total_written += wrote
        buf.clear()

    try:
        for obj in fetch_cases_since(
            api_url=lc.api_cases_url,
            user_agent=lc.user_agent,
            timeout=lc.default_timeout,
            since=since,
            page_size=args.page_size,
            server_filter_hint=args.server_filter,
            checkpoint_json=args.checkpoint,
            conn_resume=conn,  # DB-Resume aktiv (state wird ohne commit gesetzt)
            max_pages=args.max_pages,
            progress=not args.no_progress,
            full_initial_load=args.full_initial_load,
            try_ordering=not args.no_ordering,
        ):
            buf.append(obj)
            if len(buf) >= args.write_batch:
                flush()

        flush()

        if args.out_ndjson and ndjson_buf:
            write_ndjson(args.out_ndjson, ndjson_buf)
            ndjson_buf.clear()

    finally:
        pass

    print(f"[4/5] Geschriebene Fälle: {total_written}")
    conn.close()
    print("[5/5] Fertig.")


if __name__ == "__main__":
    sys.exit(main())
