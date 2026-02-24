# etl/http_client.py
import random, time
from typing import Dict, Any
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ReadTimeout, ConnectionError, ChunkedEncodingError
from urllib3.util.retry import Retry
from urllib3.exceptions import ProtocolError

def build_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = user_agent
    s.headers["Accept-Encoding"] = "identity"

    retry_kwargs = dict(
        total=8, connect=8, read=8,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    try:
        retry = Retry(allowed_methods=["GET"], **retry_kwargs)
    except TypeError:
        retry = Retry(method_whitelist=["GET"], **retry_kwargs)

    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

def try_fetch(url: str, params: Dict[str, Any], session: requests.Session, timeout: int) -> requests.Response:
    backoff = 2.0
    for _ in range(8):
        try:
            r = session.get(url, params=params, timeout=timeout)
            if r.status_code in (200, 201):
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(backoff + random.uniform(0, 0.5))
                backoff = min(backoff * 1.7, 45)
                timeout = min(timeout + 10, 120)
                continue
            r.raise_for_status()
        except (ReadTimeout, ConnectionError, ChunkedEncodingError, ProtocolError):
            time.sleep(backoff + random.uniform(0, 0.5))
            backoff = min(backoff * 1.7, 45)
            timeout = min(timeout + 10, 120)
            continue
    return session.get(url, params=params, timeout=timeout)
