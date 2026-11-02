"""Explicit isolated US commands. Importing this module does not load .env."""
import argparse
import json
from pathlib import Path

from .config import DEFAULT_ROOT, GIB, api_key
from .query import DATASETS, Query
from .store import Store
from .pipeline import ingest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("ingest","fixture","status","serve"))
    p.add_argument("--root",type=Path,default=DEFAULT_ROOT / "provider")
    p.add_argument("--dataset",choices=DATASETS)
    p.add_argument("--ticker")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--revision",default="v1")
    p.add_argument("--fetch",action="store_true",help="Explicitly enable paid provider requests; default is offline")
    p.add_argument("--api-key-file",type=Path)
    p.add_argument("--mysql-config",type=Path,help="Explicit optional metadata mirror; never auto-select a DB")
    p.add_argument("--max-requests",type=int,default=20)
    p.add_argument("--max-pages",type=int,default=100)
    p.add_argument("--max-gib",type=float,default=10)
    p.add_argument("--reserve-gib",type=float,default=40)
    p.add_argument("--port",type=int,default=8001)
    args = p.parse_args()
    try:
        if args.action == "ingest":
            if not all((args.dataset,args.ticker,args.start,args.end)):
                p.error("ingest requires --dataset, --ticker, --start and --end")
            query = Query(args.dataset,args.ticker,args.start,args.end,args.revision)
            # Fail on missing credentials BEFORE creating any data directory.
            if args.fetch:
                from .client import MassiveClient
                client = MassiveClient(api_key(args.api_key_file),max_requests=args.max_requests)
            else:
                client = None
        elif args.fetch or args.api_key_file:
            p.error("Provider credentials/fetch are accepted only for ingest")
        store = Store(args.root,max_bytes=int(args.max_gib*GIB),min_free_bytes=int(args.reserve_gib*GIB))
        if args.action == "serve":
            if args.port in (8000,3306,3307) or not 1024 <= args.port <= 65535:
                raise ValueError("Choose a separate US HTTP port (default 8001), never Taiwan port 8000")
            import uvicorn
            from .api import create_app
            uvicorn.run(create_app(store),host="127.0.0.1",port=args.port)
            return 0
        if args.action == "status":
            print(json.dumps(dict(market="US",snapshots=len(store.snapshots()),storage=store.capacity(),live_qualified=False)))
            return 0
        outputs = []
        if args.action == "fixture":
            from .fixtures import FixtureClient
            client = FixtureClient()
            queries = [Query(d,"AAPL","2026-10-29","2026-10-29","fixture-v1","synthetic_fixture") for d in DATASETS]
        else:
            queries = [query]
        engine = None
        if args.mysql_config:
            from .catalog import engine_from_file
            engine = engine_from_file(args.mysql_config)
        try:
            for query in queries:
                result, ref = ingest(query,store,client,max_pages=args.max_pages)
                if engine is not None:
                    from .catalog import record
                    record(engine,result,ref)
                outputs.append(dict(snapshot_id=result["snapshot_id"],dataset=query.dataset,rows=result["rows"],
                                    status=result["data_status"],manifest=ref["path"],live_qualified=False))
        finally:
            if engine is not None:
                engine.dispose()
        print(json.dumps(dict(market="US",results=outputs,requests=client.requests if client else 0),indent=2))
        return 0
    except Exception as exc:
        # Never serialize arbitrary exceptions (requests/SQLAlchemy can include secrets).
        if type(exc) in (ValueError,):
            message = str(exc)
        else:
            from .client import MassiveError
            message = str(exc) if isinstance(exc,MassiveError) else "US operation failed; inspect configuration/data (details suppressed to protect credentials)"
        print(json.dumps(dict(error=message,error_type=type(exc).__name__)))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
