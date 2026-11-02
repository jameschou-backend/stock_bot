"""Explicit US symbols, dates and raw-price policy; no Taiwan ID fallback."""
from dataclasses import dataclass, asdict
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo
import hashlib
import json
import re
from urllib.parse import urlencode, quote

DATASETS = ("daily", "trades", "tickers", "splits", "dividends")
NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Query:
    dataset: str
    ticker: str
    start: str
    end: str
    revision: str = "v1"
    source_kind: str = "massive"

    def __post_init__(self):
        if self.dataset not in DATASETS:
            raise ValueError("Unsupported Massive dataset")
        if self.source_kind not in ("massive", "synthetic_fixture"):
            raise ValueError("Unknown data provenance")
        # US symbols are a separate contract; never loosen the Taiwan four-digit validator.
        if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,15}", self.ticker):
            raise ValueError("Expected an explicit US equity ticker, e.g. AAPL or BRK.B")
        first, last = date.fromisoformat(self.start), date.fromisoformat(self.end)
        if str(first) != self.start or str(last) != self.end or not 0 <= (last-first).days <= 366:
            raise ValueError("Use an ISO date range of at most 367 calendar days")
        if self.dataset in ("trades", "tickers") and first != last:
            raise ValueError("Trades and ticker snapshots require one explicit New York date")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.revision):
            raise ValueError("Invalid immutable revision")

    def descriptor(self):
        return dict(schema="massive_us_query_v1", market="US", provider="massive", adjusted=False, **asdict(self))

    @property
    def key(self):
        return hashlib.sha256(json.dumps(self.descriptor(), sort_keys=True).encode()).hexdigest()

    def url(self):
        symbol = quote(self.ticker, safe="")
        if self.dataset == "daily":
            path = f"/v2/aggs/ticker/{symbol}/range/1/day/{self.start}/{self.end}"
            params = dict(adjusted="false", sort="asc", limit=50000)
        elif self.dataset == "trades":
            day = date.fromisoformat(self.start)
            start = int(datetime.combine(day, time(), NY).timestamp()) * 10**9
            end = int(datetime.combine(day + timedelta(days=1), time(), NY).timestamp()) * 10**9
            path = f"/v3/trades/{symbol}"
            params = {"timestamp.gte":start, "timestamp.lt":end, "sort":"timestamp", "order":"asc", "limit":50000}
        elif self.dataset == "tickers":
            path = "/v3/reference/tickers"
            # Omitting active allows active/inactive results; this is a dated symbol
            # lookup, NOT a complete historical-universe certificate.
            params = dict(ticker=self.ticker, date=self.start, market="stocks", locale="us", limit=1000)
        else:
            path = f"/stocks/v1/{self.dataset}"
            field = "execution_date" if self.dataset == "splits" else "ex_dividend_date"
            params = {"ticker":self.ticker, f"{field}.gte":self.start, f"{field}.lte":self.end, "limit":5000, "sort":f"{field}.asc"}
        return "https://api.massive.com" + path + "?" + urlencode(params)
