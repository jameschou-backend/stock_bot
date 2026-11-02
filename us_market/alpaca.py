"""Historical SIP bars/trades only. No brokerage calls, IEX fallback or POC claims."""
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as day_time, timedelta, timezone
from decimal import Decimal
import hashlib
import json
import re
import time
from urllib.parse import parse_qsl, urlencode, urlsplit

import pyarrow as pa
import requests

from .client import clean
from .normalize import DECIMAL, number, trading_date
from .query import NY

ORIGIN = "https://data.alpaca.markets"
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class AlpacaError(RuntimeError):
    pass


def timestamp_ns(value):
    """Retain all nine fractional digits, without float/microsecond rounding."""
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})", str(value))
    if not match:
        raise ValueError("Alpaca timestamps require timezone-aware RFC3339 with at most nanoseconds")
    base, fraction, offset = match.groups()
    stamp = datetime.fromisoformat(base + ("+00:00" if offset == "Z" else offset))
    delta = stamp - EPOCH
    result = (delta.days * 86400 + delta.seconds) * 10**9 + int((fraction or "").ljust(9, "0"))
    if not 0 <= result <= 2**63 - 1:
        raise ValueError("Alpaca timestamp outside retained range")
    return result


def _parts(url):
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.netloc != "data.alpaca.markets" or parts.fragment
            or parts.path not in ("/v2/stocks/trades", "/v2/stocks/bars")):
        raise AlpacaError("Rejected non-historical Alpaca endpoint or origin")
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    params = dict(pairs)
    allowed = {"symbols", "start", "end", "feed", "asof", "sort", "limit", "page_token"}
    if parts.path.endswith("/bars"):
        allowed |= {"timeframe", "adjustment"}
        if params.get("timeframe") != "1Day" or params.get("adjustment") != "raw":
            raise AlpacaError("Only explicitly unadjusted daily bars are supported")
    if len(params) != len(pairs) or set(params) - allowed:
        raise AlpacaError("Unexpected or duplicate Alpaca query parameters")
    if (params.get("feed") != "sip" or params.get("asof") != "-"
            or params.get("sort") != "asc" or params.get("limit") != "10000"
            or not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,15}", params.get("symbols", ""))):
        raise AlpacaError("Alpaca requires explicit SIP, raw symbol identity and bounded ascending pages")
    start, end = timestamp_ns(params.get("start")), timestamp_ns(params.get("end"))
    if not 0 <= end - start < 368 * 86400 * 10**9:
        raise AlpacaError("Alpaca historical interval exceeds acquisition bounds")
    token = params.get("page_token")
    if token is not None and (not token or len(token) > 4096 or any(ord(c) < 32 for c in token)):
        raise AlpacaError("Invalid Alpaca pagination token")
    return parts.path, params


@dataclass(frozen=True)
class AlpacaQuery:
    dataset: str
    ticker: str
    start: str
    end: str
    revision: str = "v1"
    source_kind: str = "alpaca"
    error_type = AlpacaError

    def __post_init__(self):
        if self.dataset not in ("daily", "trades") or self.source_kind not in ("alpaca", "synthetic_fixture"):
            raise ValueError("Alpaca supports explicit daily/trades datasets and provenance only")
        if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,15}", self.ticker):
            raise ValueError("Expected an explicit US equity ticker")
        first, last = date.fromisoformat(self.start), date.fromisoformat(self.end)
        if str(first) != self.start or str(last) != self.end or not 0 <= (last - first).days <= 366:
            raise ValueError("Use an ISO date range of at most 367 calendar days")
        if self.dataset == "trades" and first != last:
            raise ValueError("Alpaca trades require one explicit New York date")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.revision):
            raise ValueError("Invalid immutable revision")

    def descriptor(self):
        return dict(schema="alpaca_us_query_v1", market="US", provider="alpaca", feed="sip",
                    adjusted=False, symbol_mapping="asof=-", **asdict(self))

    @property
    def key(self):
        return hashlib.sha256(json.dumps(self.descriptor(), sort_keys=True).encode()).hexdigest()

    def url(self, page_token=None):
        start = datetime.combine(date.fromisoformat(self.start), day_time(), NY).astimezone(timezone.utc)
        stop = datetime.combine(date.fromisoformat(self.end) + timedelta(days=1), day_time(), NY).astimezone(timezone.utc)
        # Alpaca's end is inclusive. Avoid leaking the next NY date into a page.
        end = (stop - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S") + ".999999999Z"
        params = dict(symbols=self.ticker, start=start.strftime("%Y-%m-%dT%H:%M:%SZ"), end=end,
                      feed="sip", asof="-", sort="asc", limit="10000")
        field = "trades"
        if self.dataset == "daily":
            field = "bars"
            params.update(timeframe="1Day", adjustment="raw")
        if page_token is not None:
            params["page_token"] = page_token
        return ORIGIN + "/v2/stocks/" + field + "?" + urlencode(params)

    def validate_url(self, url):
        path, params = _parts(url)
        expected_path, expected = _parts(self.url())
        if path != expected_path or {k:v for k,v in params.items() if k != "page_token"} != expected:
            raise AlpacaError("Alpaca pagination changed the requested symbol, feed or interval")
        return self.url(params.get("page_token"))

    def next_url(self, payload):
        token = payload.get("next_page_token")
        if token is None:
            return None
        if not isinstance(token, str) or not token:
            raise AlpacaError("Malformed Alpaca next_page_token")
        return self.validate_url(self.url(token))

    def normalize_page(self, payload):
        if (not isinstance(payload, dict) or "code" in payload or "message" in payload
                or "next_url" in payload or bool(payload.get("synthetic_fixture")) != (self.source_kind == "synthetic_fixture")):
            raise ValueError("Invalid Alpaca page or synthetic/provider provenance mismatch")
        field = "trades" if self.dataset == "trades" else "bars"
        data = payload.get(field)
        if field not in payload or (data is not None and not isinstance(data, dict)):
            raise ValueError("Alpaca page is missing the requested data map")
        data = data or {}
        if set(data) - {self.ticker}:
            raise ValueError("Alpaca returned an unexpected symbol")
        records = data.get(self.ticker)
        if records is None:
            records = []
        if not isinstance(records, list):
            raise ValueError("Expected an Alpaca record array")
        fields = [("ticker",pa.string()), ("session_date",pa.date32()), ("timestamp_ns",pa.int64())]
        if self.dataset == "daily":
            fields += [(name,DECIMAL) for name in ("open","high","low","close","volume")]
        else:
            # Alpaca uses alphabetic exchange/condition codes; never coerce to Massive enums.
            fields += [("price",DECIMAL),("shares",DECIMAL),("trade_id",pa.string()),
                       ("exchange",pa.string()),("conditions",pa.list_(pa.string())),("tape",pa.string())]
        rows = []
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Expected an Alpaca record object")
            ns = timestamp_ns(record["t"])
            day = trading_date(ns)
            if not self.start <= str(day) <= self.end:
                raise ValueError("Alpaca record outside requested New York dates")
            row = dict(ticker=self.ticker, session_date=day, timestamp_ns=ns)
            if self.dataset == "daily":
                row.update({name:number(record[key],positive=name!="volume") for key,name in
                            (("o","open"),("h","high"),("l","low"),("c","close"),("v","volume"))})
                if not row["low"] <= min(row["open"],row["close"]) <= max(row["open"],row["close"]) <= row["high"]:
                    raise ValueError("Invalid Alpaca OHLC envelope")
            else:
                ident = record.get("i")
                if type(ident) not in (int,str) or not re.fullmatch(r"\d+",str(ident)):
                    raise ValueError("Alpaca trade identity missing or invalid; pagination cannot be certified")
                exchange, tape, conditions = record.get("x"), record.get("z"), record.get("c")
                if not isinstance(exchange,str) or len(exchange) != 1 or (tape is not None and tape not in ("A","B","C")):
                    raise ValueError("Unknown Alpaca exchange/tape identity")
                if isinstance(conditions,str):
                    conditions = [conditions]
                if conditions is not None and (not isinstance(conditions,list) or any(not isinstance(c,str) or len(c)!=1 for c in conditions)):
                    raise ValueError("Invalid Alpaca condition codes")
                row.update(price=number(record["p"],positive=True),shares=number(record["s"]),
                           trade_id=str(ident),exchange=exchange,conditions=conditions,tape=tape)
            rows.append(row)
        return pa.Table.from_pylist(rows,schema=pa.schema(fields))

    def row_identity(self, row):
        if self.dataset == "daily":
            return (row["timestamp_ns"],)
        # Same-time/same-price prints with different IDs remain independent.
        return (row["session_date"],row["exchange"],row["tape"],row["trade_id"])

    def publication_fields(self):
        return dict(schema="alpaca_us_snapshot_v1",feed="sip",symbol_mapping="asof=-",
                    session_scope="new_york_calendar_day_unfiltered",poc_ready=False,
                    trade_conditions_reconciled=False,corrections_reconciled=False)


class AlpacaClient:
    def __init__(self, key, secret, *, max_requests=20, max_response_bytes=32*1024*1024,
                 session=None, now=None):
        if any(not value or any(c.isspace() for c in value) for value in (key,secret)):
            raise ValueError("Alpaca key and secret are required")
        if max_requests < 1 or max_response_bytes < 1:
            raise ValueError("Positive request/response budgets are required")
        self._key, self._secret = key, secret
        self.session = session or requests.Session()
        self.session.trust_env = False
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.max_requests, self.max_response_bytes = max_requests, max_response_bytes
        self.requests, self._last_request = 0, None

    def page(self, url):
        _, params = _parts(url)
        cutoff = self.now() - timedelta(minutes=15)
        if cutoff.tzinfo is None:
            raise ValueError("Alpaca credential checks require an aware UTC clock")
        if timestamp_ns(params["end"]) >= timestamp_ns(cutoff.isoformat()):
            raise AlpacaError("Alpaca historical SIP end must be older than 15 minutes; no recent-data request sent")
        if self.requests >= self.max_requests:
            raise AlpacaError("Request budget exhausted; rerun to resume verified pages")
        if self._last_request is not None:
            time.sleep(max(0, .35 - (time.monotonic() - self._last_request)))
        self._last_request = time.monotonic()
        self.requests += 1
        try:
            with self.session.get(url, headers={"APCA-API-KEY-ID":self._key,"APCA-API-SECRET-KEY":self._secret},
                                  timeout=(10,45),allow_redirects=False,stream=True) as response:
                if response.status_code != 200:
                    raise AlpacaError(f"Alpaca HTTP {response.status_code}; no automatic retry, IEX or provider fallback")
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > self.max_response_bytes:
                        raise AlpacaError("Alpaca response exceeded memory budget")
                    chunks.append(chunk)
                payload = json.loads(b"".join(chunks),parse_float=Decimal)
        except AlpacaError:
            raise
        except Exception:
            raise AlpacaError("Alpaca transport/JSON failure; credentials and response omitted") from None
        if not isinstance(payload,dict) or "code" in payload or "message" in payload:
            raise AlpacaError("Alpaca returned an unsuccessful or unknown response")
        return clean(clean(payload,self._key),self._secret)
