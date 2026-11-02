#!/usr/bin/env python3
"""Fingerprint all Taiwan DB rows in a consistent, read-only transaction.

No row values or credentials are written to the receipt. This does not create
a recoverable backup. Run separately from file inventory because a concurrent
authorized daily update legitimately changes database fingerprints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fingerprint():
    from app.config import load_config
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url
    url = make_url(load_config().db_url)
    if (url.host not in ("127.0.0.1", "localhost") or url.port != 3307
            or url.database != "stock_bot"):
        raise ValueError("Unexpected Taiwan database target; stopped")
    engine = create_engine(url, connect_args={"connect_timeout":5, "read_timeout":120})
    result = {}
    try:
        with engine.connect() as conn:
            conn.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"))
            conn.execute(text("START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY"))
            tables = conn.execute(text("SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA=DATABASE() AND TABLE_TYPE='BASE TABLE' ORDER BY TABLE_NAME")).scalars().all()
            for table in tables:
                if not re.fullmatch(r"[A-Za-z0-9_]+", table):
                    raise ValueError("Unexpected table identifier")
                keys = conn.execute(text("SELECT COLUMN_NAME FROM information_schema.KEY_COLUMN_USAGE WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=:t AND CONSTRAINT_NAME='PRIMARY' ORDER BY ORDINAL_POSITION"), {"t":table}).scalars().all()
                if not keys or any(not re.fullmatch(r"[A-Za-z0-9_]+", k) for k in keys):
                    raise ValueError("Every protected table needs a deterministic primary key")
                ddl = conn.execute(text(f"SHOW CREATE TABLE `{table}`")).one()[1]
                order = ",".join(f"`{k}`" for k in keys)
                rows = conn.execution_options(stream_results=True).execute(text(f"SELECT * FROM `{table}` ORDER BY {order}"))
                h, count = hashlib.sha256(), 0
                h.update(json.dumps(list(rows.keys())).encode())
                for row in rows:
                    h.update(json.dumps(list(row), default=str, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode() + b"\n")
                    count += 1
                rows.close()
                result[table] = dict(rows=count, sha256=h.hexdigest(),
                                     schema_sha256=hashlib.sha256(ddl.encode()).hexdigest())
                print(json.dumps({"table":table,"rows":count}), file=sys.stderr, flush=True)
            conn.rollback()
    finally:
        engine.dispose()
    return dict(schema="tw_database_fingerprint_v1", target="127.0.0.1:3307/stock_bot",
                read_only=True, is_backup=False, tables=result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compare", type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error("Output already exists; choose a new receipt")
    result = fingerprint()
    if args.compare:
        before = json.loads(args.compare.read_text())
        result["unchanged"] = before["tables"] == result["tables"]
        result["changed_tables"] = sorted(t for t in set(before["tables"]) | set(result["tables"])
                                           if before["tables"].get(t) != result["tables"].get(t))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "tables"}))
    return 0 if result.get("unchanged", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
