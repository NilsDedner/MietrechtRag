# etl/timeutils.py
import datetime as dt

def parse_iso(s: str) -> dt.datetime:
    if not s:
        return dt.datetime.min.replace(tzinfo=dt.timezone.utc)
    s = s.strip()
    try:
        if s.endswith("Z"):
            return dt.datetime.fromisoformat(s[:-1]).replace(tzinfo=dt.timezone.utc)
        t = dt.datetime.fromisoformat(s)
        return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
    except Exception:
        try:
            d = dt.date.fromisoformat(s[:10])
            return dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc)
        except Exception:
            return dt.datetime.min.replace(tzinfo=dt.timezone.utc)

def isoformat_utc(ts: dt.datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tz.timezone.utc)
    return ts.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
