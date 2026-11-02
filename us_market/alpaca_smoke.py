"""Bounded, explicit historical SIP smoke test; never a POC qualification test."""
import argparse
from decimal import Decimal
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .alpaca import AlpacaClient, AlpacaError, AlpacaQuery
from .config import DEFAULT_ROOT, alpaca_credentials
from .pipeline import ingest
from .store import Store, encoded, sha


def diagnose(store, daily, trades):
    """Compare raw sums as a diagnostic, without inventing condition eligibility."""
    for snapshot, dataset in ((daily,"daily"),(trades,"trades")):
        if (snapshot["query"]["provider"] != "alpaca" or snapshot["query"]["dataset"] != dataset
                or not snapshot["complete_pagination"]):
            raise ValueError("Alpaca diagnostics require complete daily and trade snapshots")
    for key in ("ticker","start","end","feed","source_kind"):
        if daily["query"][key] != trades["query"][key]:
            raise ValueError("Diagnostic snapshots must describe the same symbol, date, feed and provenance")
    volume, bar_rows, trade_rows, unknown = Decimal(0), 0, 0, 0
    raw_volume, first, last = Decimal(0), None, None
    exchanges, tapes = set(), set()
    for snapshot in (daily,trades):
        for page in snapshot["pages"]:
            # Verify stored bytes before reading the normalized data.
            table = pq.read_table(pa.BufferReader(store.read_object(page["parquet"])))
            for row in table.to_pylist():
                if snapshot is daily:
                    bar_rows += 1
                    volume += row["volume"]
                else:
                    trade_rows += 1
                    raw_volume += row["shares"]
                    unknown += row["conditions"] is None
                    exchanges.add(row["exchange"])
                    if row["tape"] is not None:
                        tapes.add(row["tape"])
                    stamp = row["timestamp_ns"]
                    first = stamp if first is None else min(stamp,first)
                    last = stamp if last is None else max(stamp,last)
    return dict(schema="alpaca_sip_diagnostic_v1",provider="alpaca",feed="sip",
                source_kind=trades["query"]["source_kind"],ticker=trades["query"]["ticker"],date=trades["query"]["start"],
                snapshot_ids=[daily["snapshot_id"],trades["snapshot_id"]],complete_pagination=True,
                daily_rows=bar_rows,trade_rows=trade_rows,daily_volume=str(volume),raw_trade_shares=str(raw_volume),
                raw_volume_equal=(raw_volume == volume) if bar_rows == 1 and trade_rows else None,
                unknown_trade_conditions=unknown,exchanges=sorted(exchanges),tapes=sorted(tapes),
                first_timestamp_ns=first,last_timestamp_ns=last,
                comparison_scope="Unfiltered raw trades; bar volume eligibility differs by sale condition. Equality is not completeness proof.",
                calendar_verified=False,trade_conditions_reconciled=False,corrections_reconciled=False,
                poc_ready=False,strategy_ready=False,live_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker",required=True)
    parser.add_argument("--date",required=True)
    parser.add_argument("--revision",default="smoke-v1")
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT/"alpaca")
    parser.add_argument("--fetch",action="store_true")
    parser.add_argument("--max-requests",type=int,default=20)
    parser.add_argument("--mysql-config",type=Path)
    args = parser.parse_args()
    try:
        if not 1 <= args.max_requests <= 20:
            raise ValueError("Smoke acceptance allows 1 to 20 total HTTP requests")
        queries = [AlpacaQuery(d,args.ticker,args.date,args.date,args.revision) for d in ("daily","trades")]
        # Never create a landing directory when credentials are missing.
        client = AlpacaClient(*alpaca_credentials(),max_requests=args.max_requests) if args.fetch else None
        store = Store(args.root)
        engine = None
        if args.mysql_config:
            from .catalog import engine_from_file
            engine = engine_from_file(args.mysql_config)
        snapshots = []
        try:
            for query in queries:
                result, ref = ingest(query,store,client,max_pages=100)
                if ingest(query,store,max_pages=100) != (result,ref):
                    raise ValueError("Offline replay differs from acquired snapshot")
                if engine is not None:
                    from .catalog import record
                    record(engine,result,ref)
                    record(engine,result,ref)
                snapshots.append(result)
        finally:
            if engine is not None:
                engine.dispose()
        report = diagnose(store,*snapshots)
        report.update(requests=client.requests if client else 0,offline_replay_verified=True)
        data = encoded(report)
        with store.lock():
            ref = store.put(f"diagnostics/{sha(data)}.json",data)
        print(json.dumps(dict(report=report,receipt=ref),indent=2))
        return 0
    except Exception as exc:
        message = str(exc) if type(exc) in (ValueError,AlpacaError) else "Alpaca smoke failed; details suppressed to protect credentials"
        print(json.dumps(dict(error=message,error_type=type(exc).__name__)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
