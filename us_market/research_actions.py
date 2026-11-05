"""Explicit cash-merger settlement overlay; incomplete actions remain unresolved."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import gzip
import io
import json
from urllib.parse import urlencode

import numpy as np
import pyarrow.parquet as pq

from .research_data import Acquisition, bars, protocol
from .store import encoded, sha

POLICY = {
    "version": "cash_merger_settlement_v1",
    "purpose": "Accounting repair; strategy signals, ranking, costs and exits remain frozen.",
    "cash_date": "First market session on/after max(payable_date, process_date, effective_date). Explicit payable_date required.",
    "cash_amount": "Adjusted units times last matched all-adjusted/raw close ratio times provider cash rate; matched bar within 10 sessions required.",
    "duplicate_policy": "Same symbol, CUSIP and rate within seven calendar days coalesced at latest payable/process/effective date; conflicting nearby rates quarantined.",
    "availability": "Settlement cash cannot fund same-day purchases. Mandatory cash redemptions have no modeled trading fee.",
    "scope": "Cash mergers only. Noncash mergers, renames, missing payment dates and unsupported identities remain explicitly unresolved.",
    "qualification": "Retrospective provider records, no publication-time certification; diagnostic only.",
}


def source_actions(store):
    bundle = store.read_json("actions/bundle.json")
    if not bundle["complete_pagination"]:
        raise ValueError("Incomplete corporate-action pagination")
    grouped = {}
    identities = set()
    for page in bundle["pages"]:
        payload = json.loads(gzip.decompress(store.read_object(page["source"])))
        for kind, records in payload["corporate_actions"].items():
            for record in records:
                if record["id"] in identities:
                    raise ValueError("Duplicate corporate-action identity")
                identities.add(record["id"])
                grouped.setdefault(kind, []).append(record)
    return grouped


def acquire_actions(store, *, fetch=False, max_requests=500):
    spec = protocol()
    runner = Acquisition(store, fetch=fetch, max_requests=max_requests)
    if not store.has("actions/bundle.json"):
        params = {"types": "cash_merger,stock_merger,stock_and_cash_merger,name_change,worthless_removal",
                  "start": spec["data_start"], "end": spec["data_end"], "limit": 1000, "sort": "asc"}
        pages, tokens = [], set()
        for _ in range(30):
            url = "https://data.alpaca.markets/v1/corporate-actions?" + urlencode(params)
            payload = runner.get(url)
            if not isinstance(payload.get("corporate_actions"), dict) or "next_page_token" not in payload:
                raise ValueError("Invalid corporate-action envelope")
            pages.append(store.read_json(f"requests/{sha(url.encode())}.json"))
            token = payload["next_page_token"]
            if token is None:
                break
            if not isinstance(token, str) or not token or token in tokens:
                raise ValueError("Invalid corporate-action pagination token")
            tokens.add(token)
            params["page_token"] = token
        else:
            raise ValueError("Corporate-action page limit reached")
        runner.put("actions/bundle.json", encoded({"pages": pages, "complete_pagination": True,
                                                  "start": spec["data_start"], "end": spec["data_end"]}))
    actions = source_actions(store)
    universe = {s["ticker"] for s in store.read_json("inputs/universe.json")["stocks"]}
    targets = sorted({e["acquiree_symbol"] for e in actions.get("cash_mergers", [])
                      if e.get("payable_date") and e.get("acquiree_symbol") in universe})
    sessions = {r["date"] for r in store.read_json("inputs/calendar.json")}
    groups = [targets[i:i + 25] for i in range(0, len(targets), 25)]
    receipts = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(bars, runner, group, "raw", spec, sessions) for group in groups]
        for job in as_completed(jobs):
            receipts.append(job.result())
    receipts.sort(key=lambda r: r["query"]["symbols"])
    receipt = {"policy": POLICY, "source_bundle_sha256": sha(encoded(store.read_json("actions/bundle.json"))),
               "raw_bars": receipts, "target_symbols": targets}
    runner.put("actions/cash-inputs.json", encoded(receipt))
    return {"cash_action_symbols": len(targets), "requests_this_run": runner.requests}


def cash_events(store, panel):
    inputs = store.read_json("actions/cash-inputs.json")
    if inputs["policy"] != POLICY or inputs["source_bundle_sha256"] != sha(encoded(store.read_json("actions/bundle.json"))):
        raise ValueError("Cash-action policy or source hash conflict")
    actions = source_actions(store)
    raw = {}
    for receipt in inputs["raw_bars"]:
        frame = pq.read_table(io.BytesIO(store.read_object(receipt["parquet"]))).to_pandas()
        for symbol, rows in frame.groupby("ticker"):
            raw[symbol] = rows.set_index("date").close
    selected, rejected = [], []
    for event in actions.get("cash_mergers", []):
        symbol = event.get("acquiree_symbol")
        if symbol not in panel.universe:
            continue
        if not event.get("payable_date"):
            rejected.append({"ticker": symbol, "id": event["id"], "reason": "payment_date_missing"})
            continue
        if not event.get("acquiree_cusip"):
            rejected.append({"ticker": symbol, "id": event["id"], "reason": "identity_missing"})
            continue
        payment = max(event["payable_date"], event["process_date"], event.get("effective_date", event["process_date"]))
        date.fromisoformat(payment)
        selected.append({"ticker": symbol, "date": payment, "rate": float(event["rate"]),
                         "cusip": event["acquiree_cusip"], "ids": [event["id"]]})
    selected.sort(key=lambda e: (e["ticker"], e["date"], e["ids"]))
    coalesced = []
    for event in selected:
        if not np.isfinite(event["rate"]) or event["rate"] <= 0:
            raise ValueError("Invalid cash merger rate")
        if coalesced and event["ticker"] == coalesced[-1]["ticker"] and (date.fromisoformat(event["date"]) - date.fromisoformat(coalesced[-1]["date"])).days <= 7:
            previous = coalesced[-1]
            if event["cusip"] != previous["cusip"] or event["rate"] != previous["rate"]:
                previous["conflict"] = True
            previous["date"] = event["date"]
            previous["ids"].extend(event["ids"])
        else:
            coalesced.append(event)
    events = {}
    for event in coalesced:
        symbol = event["ticker"]
        if event.get("conflict"):
            rejected.append({"ticker": symbol, "ids": event["ids"], "reason": "conflicting_cash_events"})
            continue
        i = int(panel.dates.searchsorted(event["date"]))
        if i >= len(panel.dates):
            rejected.append({"ticker": symbol, "ids": event["ids"], "reason": "payment_after_study"})
            continue
        values = raw.get(symbol)
        matched = [] if values is None else [d for d in panel.dates[max(0, i - 10):i + 1]
                                            if d in values.index and np.isfinite(panel.fields["close"].loc[d, symbol])]
        if not matched:
            rejected.append({"ticker": symbol, "ids": event["ids"], "reason": "adjusted_units_unverified"})
            continue
        day = matched[-1]
        factor = float(panel.fields["close"].loc[day, symbol] / values.loc[day])
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Invalid cash merger share conversion")
        value = {**event, "date": panel.dates[i], "unit_factor": factor, "factor_date": day,
                 "adjusted_cash_per_unit": factor * event["rate"]}
        events.setdefault(i, {})[panel.symbols.index(symbol)] = value
    audit = {"policy": POLICY, "cash_inputs_sha256": sha(encoded(inputs)), "eligible_settlements": sum(map(len, events.values())),
             "cash_records_by_process_year": dict(Counter(e["process_date"][:4] for e in actions.get("cash_mergers", []))),
             "rejected_records": rejected, "event_types": {k: len(v) for k, v in actions.items()},
             "records_before_2020_available": any(e["process_date"] < "2020-01-01" for e in actions.get("cash_mergers", []))}
    return events, audit
