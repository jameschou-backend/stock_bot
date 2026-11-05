"""Immutable historical universe, calendar and daily bars for US research only."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import json
from pathlib import Path
import re
import threading
import time
from urllib.parse import parse_qsl, urlencode, urlsplit

import pyarrow as pa
import pyarrow.parquet as pq
import requests

from .alpaca import AlpacaError, AlpacaQuery, timestamp_ns
from .client import MassiveClient, clean, safe_url
from .config import alpaca_credentials, api_key
from .normalize import number, trading_date
from .store import Store, encoded, sha

PROTOCOL_PATH = Path(__file__).with_name("research_protocol.json")
SYMBOL = re.compile(r"[A-Z][A-Z0-9.-]{0,15}")


def protocol():
    return json.loads(PROTOCOL_PATH.read_text())


class Acquisition:
    """A bounded, thread-safe acquisition run with verified offline replay."""
    def __init__(self, store, *, fetch=False, max_requests=2500):
        self.store = store
        self.fetch = fetch
        self.max_requests = max_requests
        self.requests = 0
        self._guard = threading.Lock()
        self._write_guard = threading.Lock()
        self._last = {"alpaca": 0.0, "massive": 0.0}
        self._local = threading.local()
        self._pair = alpaca_credentials() if fetch else None
        self._massive = MassiveClient(api_key(), max_requests=max_requests) if fetch else None

    def put(self, path, data):
        with self._write_guard, self.store.lock():
            return self.store.put(path, data)

    def _reserve(self, provider, url):
        with self._guard:
            if self.requests >= self.max_requests:
                raise AlpacaError("Research request budget exhausted; cached pages can resume")
            interval = 12.5 if provider == "massive" else 0.4
            time.sleep(max(0, interval - (time.monotonic() - self._last[provider])))
            self._last[provider] = time.monotonic()
            self.requests += 1
            self.put(f"attempts/{time.time_ns()}.json", encoded({"provider": provider, "query_sha256": sha(url.encode())}))

    @staticmethod
    def validate_url(url, provider):
        if provider == "massive":
            return safe_url(url, expected_path="/v3/reference/tickers")
        parsed = urlsplit(url)
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
        params = dict(pairs)
        if parsed.scheme != "https" or parsed.fragment or len(pairs) != len(params):
            raise ValueError("Invalid research data URL")
        if parsed.netloc == "paper-api.alpaca.markets" and parsed.path == "/v2/calendar":
            if set(params) != {"start", "end"}:
                raise ValueError("Calendar request parameters differ")
            return url
        if parsed.netloc == "data.alpaca.markets" and parsed.path == "/v1/corporate-actions":
            required = {"types", "start", "end", "limit", "sort"}
            if (not required.issubset(params) or set(params) - required - {"page_token"}
                    or params["limit"] != "1000" or params["sort"] != "asc"
                    or params["types"] != "cash_merger,stock_merger,stock_and_cash_merger,name_change,worthless_removal"):
                raise ValueError("Unexpected corporate-action query")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", params["start"]) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", params["end"]):
                raise ValueError("Invalid corporate-action interval")
            if not "2018-01-01" <= params["start"] <= params["end"] <= protocol()["data_end"]:
                raise ValueError("Corporate-action dates exceed the frozen study")
            return url
        if parsed.netloc != "data.alpaca.markets" or parsed.path != "/v2/stocks/bars":
            raise ValueError("Only historical bars and market calendar are allowed")
        allowed = {"symbols", "start", "end", "feed", "asof", "sort", "limit", "timeframe", "adjustment", "page_token"}
        if (set(params) - allowed or params.get("feed") != "sip" or params.get("asof") != "-"
                or params.get("timeframe") != "1Day" or params.get("adjustment") not in ("raw", "split", "all")
                or params.get("sort") != "asc" or params.get("limit") != "10000"):
            raise ValueError("Unexpected research bars parameters")
        symbols = params.get("symbols", "").split(",")
        if not 1 <= len(symbols) <= 25 or len(set(symbols)) != len(symbols) or any(not SYMBOL.fullmatch(x) for x in symbols):
            raise ValueError("Invalid research symbol batch")
        start, end = timestamp_ns(params["start"]), timestamp_ns(params["end"])
        if not 0 < end - start < 10 * 366 * 86400 * 10**9:
            raise ValueError("Research date window exceeds ten years")
        if end >= int((datetime.now(timezone.utc).timestamp() - 900) * 10**9):
            raise ValueError("Historical data must be at least 15 minutes old")
        return url

    def get(self, url, provider="alpaca"):
        url = self.validate_url(url, provider)
        request_path = f"requests/{sha(url.encode())}.json"
        if self.store.has(request_path):
            receipt = self.store.read_json(request_path)
            if receipt["url"] != url:
                raise ValueError("Cached request identity conflict")
            return json.loads(gzip.decompress(self.store.read_object(receipt["source"])))
        if not self.fetch:
            raise ValueError("Research source missing; explicit --fetch required")
        self._reserve(provider, url)
        if provider == "massive":
            payload = self._massive.page(url)
        else:
            session = getattr(self._local, "session", None)
            if session is None:
                session = self._local.session = requests.Session()
                session.trust_env = False
            key, secret = self._pair
            try:
                with session.get(url, headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
                                 timeout=(10, 60), stream=True, allow_redirects=False) as response:
                    if response.status_code != 200:
                        raise AlpacaError(f"Alpaca research HTTP {response.status_code}; resume only after resolving the error")
                    parts, size = [], 0
                    for chunk in response.iter_content(65536):
                        size += len(chunk)
                        if size > 64 * 1024**2:
                            raise AlpacaError("Research response exceeds 64 MiB")
                        parts.append(chunk)
                    payload = json.loads(b"".join(parts), parse_float=Decimal)
            except AlpacaError:
                raise
            except Exception:
                raise AlpacaError("Research transport failed; sensitive response omitted") from None
            payload = clean(clean(payload, key), secret)
        body = gzip.compress(encoded(payload), mtime=0)
        source = self.put(f"sources/{sha(body)}.json.gz", body)
        self.put(request_path, encoded({"url": url, "provider": provider, "source": source,
                                       "retrieved_at": datetime.now(timezone.utc).isoformat()}))
        return payload


def universe(acquisition, spec):
    params = dict(date=spec["universe_date"], market="stocks", type="CS", active="true", limit=1000, sort="ticker", order="asc")
    url = "https://api.massive.com/v3/reference/tickers?" + urlencode(params)
    records, visited = [], set()
    for _ in range(50):
        if url in visited:
            raise ValueError("Universe pagination loop")
        visited.add(url)
        payload = acquisition.get(url, "massive")
        records.extend(payload["results"])
        url = payload.get("next_url")
        if not url:
            break
    else:
        raise ValueError("Universe pagination incomplete")
    selected, excluded = [], []
    for row in records:
        if row.get("primary_exchange") not in spec["universe_exchanges"]:
            continue
        if row.get("type") != "CS" or row.get("locale") != "us" or row.get("market") != "stocks":
            raise ValueError("Historical universe contains unexpected security type")
        ticker = row["ticker"]
        if re.fullmatch(r"[A-Z][A-Z0-9.-]*w", ticker):
            excluded.append({"ticker": ticker, "reason": "when-issued variant; no verified cross-provider symbol mapping"})
            continue
        if not SYMBOL.fullmatch(ticker):
            raise ValueError("Historical symbol is outside supported US syntax")
        selected.append(row)
    selected.sort(key=lambda row: row["ticker"])
    if not selected or len({r["ticker"] for r in selected}) != len(selected):
        raise ValueError("Empty or duplicated historical universe")
    result = {"date": spec["universe_date"], "reference_rows": len(records), "stocks": selected,
              "later_ipos_included": False, "present_day_survivor_filter": False,
              "excluded_security_variants": excluded}
    ref = acquisition.put("inputs/universe.json", encoded(result))
    return result, ref


def calendar(acquisition, spec):
    url = "https://paper-api.alpaca.markets/v2/calendar?" + urlencode({"start": spec["data_start"], "end": spec["data_end"]})
    rows = acquisition.get(url)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Calendar response is missing sessions")
    days = [row["date"] for row in rows]
    if days != sorted(set(days)) or days[0] < spec["data_start"] or days[-1] > spec["data_end"]:
        raise ValueError("Invalid calendar dates")
    for row in rows:
        if row["open"] >= row["close"]:
            raise ValueError("Invalid market hours")
    return rows, acquisition.put("inputs/calendar.json", encoded(rows))


def bars(acquisition, symbols, adjustment, spec, sessions):
    # Reuse exact NY midnight boundaries without changing existing raw queries.
    first = AlpacaQuery("daily", symbols[0], spec["data_start"], spec["data_start"])
    last = AlpacaQuery("daily", symbols[0], spec["data_end"], spec["data_end"])
    params = dict(parse_qsl(urlsplit(first.url()).query))
    params.update(symbols=",".join(symbols), end=dict(parse_qsl(urlsplit(last.url()).query))["end"], adjustment=adjustment)
    identity = {"symbols": symbols, "adjustment": adjustment, "start": spec["data_start"], "end": spec["data_end"]}
    path = f"bars/{sha(encoded(identity))}.json"
    if acquisition.store.has(path):
        receipt = acquisition.store.read_json(path)
        acquisition.store.read_object(receipt["parquet"])
        return receipt
    result, seen, tokens = [], set(), set()
    for _ in range(100):
        payload = acquisition.get("https://data.alpaca.markets/v2/stocks/bars?" + urlencode(params))
        if not isinstance(payload, dict) or "bars" not in payload or "next_page_token" not in payload:
            raise ValueError("Incomplete bars envelope")
        data = payload["bars"]
        if data is None:
            data = {}
        if not isinstance(data, dict) or set(data) - set(symbols):
            raise ValueError("Unexpected bars symbols")
        for ticker, records in data.items():
            if not isinstance(records, list):
                raise ValueError("Malformed bars array")
            for raw in records:
                day = str(trading_date(timestamp_ns(raw["t"])))
                if day not in sessions or (ticker, day) in seen:
                    raise ValueError("Duplicate or non-calendar daily bar")
                seen.add((ticker, day))
                values = {name: float(number(raw[key], positive=name != "volume")) for name, key in
                          (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c"), ("volume", "v"))}
                if not values["low"] <= min(values["open"], values["close"]) <= max(values["open"], values["close"]) <= values["high"]:
                    raise ValueError("Invalid research OHLC envelope")
                result.append({"ticker": ticker, "date": day, **values})
        token = payload["next_page_token"]
        if token is None:
            break
        if not isinstance(token, str) or not token or token in tokens:
            raise ValueError("Invalid bars continuation")
        tokens.add(token)
        params["page_token"] = token
    else:
        raise ValueError("Bars page limit reached")
    schema = pa.schema([("ticker", pa.string()), ("date", pa.string())] + [(x, pa.float64()) for x in ("open", "high", "low", "close", "volume")])
    table = pa.Table.from_pylist(result, schema=schema)
    buf = pa.BufferOutputStream()
    pq.write_table(table, buf, compression="zstd")
    ref = acquisition.put(path.replace(".json", ".parquet"), buf.getvalue().to_pybytes())
    receipt = {"query": identity, "rows": len(result), "parquet": ref, "complete_pagination": True}
    acquisition.put(path, encoded(receipt))
    return receipt


def acquire(store, *, fetch=False, max_requests=2500):
    spec = protocol()
    runner = Acquisition(store, fetch=fetch, max_requests=max_requests)
    runner.put("protocol.json", encoded(spec))
    names, names_ref = universe(runner, spec)
    days, days_ref = calendar(runner, spec)
    symbols = sorted({row["ticker"] for row in names["stocks"]} | {spec["benchmark"]})
    session_set = {row["date"] for row in days}
    groups = [symbols[i:i + spec["batch_symbols"]] for i in range(0, len(symbols), spec["batch_symbols"])]
    receipts = []
    print(json.dumps({"phase": "acquire", "universe_stocks": len(names["stocks"]), "calendar_sessions": len(days), "jobs": len(groups) * 2}), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(bars, runner, group, adjustment, spec, session_set)
                for group in groups for adjustment in spec["adjustments"]]
        for completed, future in enumerate(as_completed(jobs), 1):
            try:
                receipts.append(future.result())
            except Exception:
                for pending in jobs:
                    pending.cancel()
                raise
            if completed % 10 == 0:
                print(json.dumps({"completed_batches": completed, "requests_this_run": runner.requests}), flush=True)
    receipts.sort(key=lambda item: (item["query"]["symbols"], item["query"]["adjustment"]))
    bundle = {"schema": "us_three_strategy_bundle_v1", "protocol_sha256": sha(encoded(spec)),
              "universe": names_ref, "calendar": days_ref, "bars": receipts, "complete_acquisition": True}
    ref = runner.put("inputs/bundle.json", encoded(bundle))
    return {"bundle": ref, "requests_this_run": runner.requests, "stocks": len(names["stocks"]), "bar_rows": sum(r["rows"] for r in receipts)}
