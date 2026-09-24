"""Inspect TPEx's public MTH sample; never produce an authenticated replay tape.

The published 67-byte layout has a buy and a sell record for a trade.  This
inspector accepts only matching pairs, so adding both sides cannot double the
reported shares.  Its local format ID is not an official historical version.
The twelve time characters remain raw: their precision and historical session
mapping have not been independently established for every file version.
"""
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
import hashlib
import re


FORMAT_ID = "tpex_mth_67_v1"
_WIDTHS = (
    ("date", 8), ("stock_id", 6), ("side", 1), ("trade_type", 1),
    ("time_raw", 12), ("trade_number", 8), ("order_number_2", 5),
    ("price", 7), ("shares", 9), ("price_type", 1),
    ("time_restriction", 1), ("filler", 2), ("order_type", 1),
    ("investor_type", 1), ("order_number_1", 4),
)


def inspect_mth_sample(raw: bytes, *, format_id: str) -> dict:
    """Return paired-record diagnostics without inferring complete sessions.

Unknown formats, codes, malformed records and unmatched sides raise ValueError.
The four-digit scope includes the project's stocks and 0050; other security
codes require a separate, explicit extension rather than silent acceptance.
"""
    if format_id != FORMAT_ID:
        raise ValueError("Unsupported MTH inspection format")
    if not isinstance(raw, bytes) or not raw:
        raise ValueError("MTH inspection requires nonempty raw bytes")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("MTH records must be ASCII") from exc
    # Only LF or CRLF separators; splitlines() alone would also accept control
    # characters as delimiters and conceal a malformed fixed-width record.
    lines = text.replace("\r\n", "\n").split("\n")
    if lines[-1] == "":
        lines.pop()
    groups = defaultdict(list)
    raw_types = Counter()
    raw_side_shares = defaultdict(Counter)
    for line_number, line in enumerate(lines, 1):
        if len(line) != 67 or any(ord(c) < 32 or ord(c) > 126 for c in line):
            raise ValueError(f"MTH line {line_number}: expected 67 printable bytes")
        row, offset = {}, 0
        for key, width in _WIDTHS:
            row[key] = line[offset:offset + width]
            offset += width
        try:
            if not re.fullmatch(r"\d{8}", row["date"]):
                raise ValueError("Invalid date")
            datetime.strptime(row["date"], "%Y%m%d")
            if not re.fullmatch(r"\d{4}  ", row["stock_id"]):
                raise ValueError("Only four-digit securities are supported")
            if row["side"] not in ("B", "S") or row["trade_type"] not in "012":
                raise ValueError("Unknown side or trade type")
            if not re.fullmatch(r"\d{12}", row["time_raw"]):
                raise ValueError("Invalid raw time")
            datetime.strptime(row["time_raw"][:6], "%H%M%S")
            if not re.fullmatch(r"\d{8}", row["trade_number"]):
                raise ValueError("Invalid trade number")
            if not re.fullmatch(r"\d{4}\.\d{2}", row["price"]):
                raise ValueError("Unsupported price encoding")
            if Decimal(row["price"]) <= 0:
                raise ValueError("Nonpositive price")
            if not re.fullmatch(r"\d{9}", row["shares"]) or int(row["shares"]) <= 0:
                raise ValueError("Nonpositive or invalid shares")
            if (row["price_type"] not in "12" or row["time_restriction"] not in "034"
                    or row["filler"] != "  " or row["order_type"] not in "0123456"
                    or row["investor_type"] not in "MFIJ"):
                raise ValueError("Unknown MTH classification or filler")
        except ValueError as exc:
            raise ValueError(f"MTH line {line_number}: {exc}") from exc
        row["stock_id"] = row["stock_id"].strip()
        key = tuple(row[k] for k in ("date", "stock_id", "trade_type", "trade_number"))
        groups[key].append(row)
        raw_types[row["trade_type"]] += 1
        raw_side_shares[row["trade_type"]][row["side"]] += int(row["shares"])

    by_type = defaultdict(lambda: {"paired_trades": 0, "shares_once": 0, "value_cents_once": 0,
                                  "first_time_raw": None, "last_time_raw": None,
                                  "raw_clock_buckets": {}})
    for key, pair in groups.items():
        if (len(pair) != 2 or {r["side"] for r in pair} != {"B", "S"}
                or len({(r["time_raw"], r["price"], r["shares"]) for r in pair}) != 1):
            raise ValueError(f"MTH trade {key}: expected one matching buy/sell pair")
        row = pair[0]
        stats = by_type[row["trade_type"]]
        stats["paired_trades"] += 1
        stats["shares_once"] += int(row["shares"])
        value = int(Decimal(row["price"]) * 100) * int(row["shares"])
        stats["value_cents_once"] += value
        at = row["time_raw"]
        stats["first_time_raw"] = min(stats["first_time_raw"] or at, at)
        stats["last_time_raw"] = max(stats["last_time_raw"] or at, at)
        # A clock distribution only, not an authenticated intraday/after-hours
        # channel classification. No time is converted to replay microseconds.
        bucket = ("0910_to_1330" if "091000000000" <= at <= "133000000000"
                  else "143000000000" if at == "143000000000" else "other")
        count = stats["raw_clock_buckets"].setdefault(bucket, {
            "paired_trades": 0, "shares_once": 0, "value_cents_once": 0})
        count["paired_trades"] += 1
        count["shares_once"] += int(row["shares"])
        count["value_cents_once"] += value
    for kind, stats in by_type.items():
        stats["raw_side_shares"] = dict(sorted(raw_side_shares[kind].items()))
        stats["raw_shares_both_sides"] = sum(raw_side_shares[kind].values())
    return {
        "schema": FORMAT_ID, "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": len(raw), "raw_rows": len(lines),
        "raw_trade_type_rows": dict(sorted(raw_types.items())),
        "paired_trades": len(groups), "by_trade_type": dict(sorted(by_type.items())),
        "clock_buckets_inferred": True,
        "clock_buckets_note": "Raw clock ranges only; not verified historical session labels or time precision",
        "dates": sorted({key[0] for key in groups}),
        "stock_ids": sorted({key[1] for key in groups}),
        "historical_session_complete": False, "execution_tape_accepted": False,
        "source_authenticated": False, "live_qualified": False,
    }
