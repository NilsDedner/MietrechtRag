# etl/api_cases.py
import json, os, time, signal
import datetime as dt
from typing import Dict, Any, Iterable, Optional, Tuple

from tqdm import tqdm

from .http_client import build_session, try_fetch
from .timeutils import parse_iso, isoformat_utc
from .pg_db import db_state_get, db_state_set_many

def load_ckpt(path: Optional[str]) -> Dict[str, Any]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_ckpt(path: Optional[str], page: int, last_updated_iso: str):
    if not path:
        return
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"page": page, "last_updated": last_updated_iso}, f)
    os.replace(tmp, path)

def detect_server_filter(api_url: str, session, since: dt.datetime, timeout: int, full_initial_load: bool) -> Optional[Tuple[str, str]]:
    if full_initial_load:
        return None
    for key in ("updated_date__gt", "updated_after", "updated_min", "updated_date_after"):
        test_params = {"page_size": 1, key: isoformat_utc(since)}
        try:
            r = try_fetch(api_url, test_params, session, timeout=timeout)
            data = r.json()
            if isinstance(data, dict) and "results" in data:
                return (key, isoformat_utc(since))
        except Exception:
            continue
    return None

def fetch_cases_since(
    api_url: str,
    user_agent: str,
    timeout: int,
    since: dt.datetime,
    page_size: int = 100,
    server_filter_hint: bool = True,
    checkpoint_json: Optional[str] = None,
    conn_resume=None,
    max_pages: Optional[int] = None,
    progress: bool = True,
    full_initial_load: bool = False,
    try_ordering: bool = True,
) -> Iterable[Dict[str, Any]]:
    session = build_session(user_agent)

    # Resume-Start
    page = 1
    newest_seen = since

    if conn_resume is not None:
        p_db = db_state_get(conn_resume, "page")
        u_db = db_state_get(conn_resume, "last_updated")
        if p_db:
            try: page = max(1, int(p_db))
            except Exception: page = 1
        if u_db:
            dt_u = parse_iso(u_db)
            if dt_u > newest_seen:
                newest_seen = dt_u

    ckpt = load_ckpt(checkpoint_json)
    if ckpt:
        page = max(page, int(ckpt.get("page", page)))
        u_ck = parse_iso(ckpt.get("last_updated", isoformat_utc(newest_seen)))
        if u_ck > newest_seen:
            newest_seen = u_ck

    use_server_filter = None
    if server_filter_hint:
        use_server_filter = detect_server_filter(api_url, session, since, timeout, full_initial_load)

    # Progress
    pbar = tqdm(total=None, unit="case", dynamic_ncols=True, disable=not progress, desc="API")

    stop_flag = {"halt": False}
    def _graceful_exit(signum, frame):
        stop_flag["halt"] = True
        pbar.write("Signal empfangen – sauberer Stopp nach aktueller Seite …")

    old_int = signal.getsignal(signal.SIGINT)
    old_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)

    try:
        pages_done = 0
        seen_old_streak = 0

        while True:
            params = {"page": page, "page_size": page_size}
            if use_server_filter:
                params[use_server_filter[0]] = use_server_filter[1]
            if try_ordering:
                params["ordering"] = "-updated_date"

            r = try_fetch(api_url, params, session, timeout=timeout)
            data = r.json()
            results = data.get("results") or []
            if not results:
                break

            page_new, page_old = 0, 0
            for obj in results:
                u = parse_iso(obj.get("updated_date") or "")
                if full_initial_load or u > since:
                    yield obj
                    page_new += 1
                    if u > newest_seen:
                        newest_seen = u
                else:
                    page_old += 1
                pbar.update(1)

            # Update State (ohne Commit hier!)
            last_iso = isoformat_utc(newest_seen)
            if conn_resume is not None:
                db_state_set_many(conn_resume, {"page": str(page + 1), "last_updated": last_iso})
            save_ckpt(checkpoint_json, page + 1, last_iso)
            pbar.set_postfix(page=page, new=page_new, old=page_old)

            if (not full_initial_load) and try_ordering:
                if page_new == 0 and page_old > 0:
                    seen_old_streak += 1
                else:
                    seen_old_streak = 0
                if seen_old_streak >= 3:
                    pbar.write("Nur noch alte Fälle (<= since) – Abbruch.")
                    break

            page += 1
            pages_done += 1
            if max_pages and pages_done >= max_pages:
                pbar.write("Maximale Seitenanzahl erreicht – Stop.")
                break
            if stop_flag["halt"]:
                break
            time.sleep(0.2)

    finally:
        pbar.close()
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGTERM, old_term)
