"""Bounded GET-only transport. Never log secrets, response bodies or signed URLs."""
from __future__ import annotations

from decimal import Decimal
import json
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import requests

SECRET_KEYS = {"apikey", "api_key", "authorization", "token", "password", "secret"}


class MassiveError(RuntimeError):
    pass


def safe_url(url: str, *, expected_path: str | None = None) -> str:
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.netloc != "api.massive.com" or parts.fragment
            or not parts.path.startswith(("/v2/aggs/", "/v3/trades/", "/v3/reference/tickers", "/stocks/v1/"))
            or (expected_path is not None and parts.path != expected_path)):
        raise MassiveError("Rejected unexpected pagination origin or endpoint")
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k.lower() not in SECRET_KEYS]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def clean(value, secret=""):
    if isinstance(value, dict):
        return {k:clean(v, secret) for k,v in value.items() if k.lower() not in SECRET_KEYS}
    if isinstance(value, list):
        return [clean(x, secret) for x in value]
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    return value


class MassiveClient:
    def __init__(self, key: str, *, max_requests=20, max_response_bytes=32*1024*1024, session=None):
        if not key:
            raise ValueError("Massive API key is required for network access")
        if max_requests < 1 or max_response_bytes < 1:
            raise ValueError("Positive request/response budgets are required")
        self._key = key
        self.session = session or requests.Session()
        self.session.trust_env = False
        self.max_requests, self.max_response_bytes = max_requests, max_response_bytes
        self.requests = 0

    def page(self, url):
        url = safe_url(url)
        if self.requests >= self.max_requests:
            raise MassiveError("Request budget exhausted; rerun to resume verified pages")
        self.requests += 1
        try:
            with self.session.get(url, headers={"Authorization":"Bearer " + self._key},
                                  timeout=(10, 45), allow_redirects=False, stream=True) as response:
                if response.status_code != 200:
                    raise MassiveError(f"Massive HTTP {response.status_code}; no automatic retry or provider fallback")
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > self.max_response_bytes:
                        raise MassiveError("Response exceeded memory budget")
                    chunks.append(chunk)
                payload = json.loads(b"".join(chunks), parse_float=Decimal)
        except MassiveError:
            raise
        except Exception:
            raise MassiveError("Massive transport/JSON failure; credentials and response omitted") from None
        if not isinstance(payload, dict) or payload.get("status") not in ("OK", "DELAYED"):
            raise MassiveError("Massive returned an unsuccessful or unknown response status")
        if "results" not in payload:
            # Some successful empty aggregate responses omit results entirely.
            if payload.get("resultsCount") == 0:
                payload["results"] = []
            else:
                raise MassiveError("Successful response is missing results")
        if not isinstance(payload["results"], list):
            raise MassiveError("Expected a results array")
        if payload.get("next_url"):
            payload["next_url"] = safe_url(payload["next_url"], expected_path=urlsplit(url).path)
        return clean(payload, self._key)
