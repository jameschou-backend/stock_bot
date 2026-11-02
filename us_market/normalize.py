"""Lossless source landing plus typed analytical columns; not POC qualification."""
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import json

import pyarrow as pa

from .query import NY, Query

DECIMAL = pa.decimal128(38, 12)


def number(value, *, positive=False):
    try:
        value = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError("Invalid provider numeric value") from None
    if not value.is_finite() or value < 0 or (positive and value == 0) or value != value.quantize(Decimal("1e-12")):
        raise ValueError("Provider number is negative, non-finite or exceeds retained precision")
    return value


def integer(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Expected nonnegative integer timestamp/identifier")
    return value


def trading_date(ns):
    return datetime.fromtimestamp(ns // 10**9, timezone.utc).astimezone(NY).date()


def normalize(query: Query, payload):
    if payload.get("status") not in ("OK", "DELAYED") or not isinstance(payload.get("results"), list):
        raise ValueError("Invalid successful provider page")
    if bool(payload.get("synthetic_fixture")) != (query.source_kind == "synthetic_fixture"):
        raise ValueError("Synthetic/provider provenance mismatch")
    if payload.get("ticker", query.ticker) != query.ticker:
        raise ValueError("Provider envelope ticker mismatch")
    if query.dataset == "daily" and payload.get("adjusted") is not False:
        raise ValueError("Daily bars must explicitly be unadjusted")
    records = payload["results"]
    rows = []
    if query.dataset in ("daily", "trades"):
        fields = [pa.field("ticker",pa.string()),pa.field("session_date",pa.date32()),pa.field("timestamp_ns",pa.int64())]
        if query.dataset == "daily":
            fields += [pa.field(k,DECIMAL) for k in ("open","high","low","close","volume")]
        else:
            fields += [pa.field("price",DECIMAL),pa.field("shares",DECIMAL),pa.field("trade_id",pa.string()),
                       pa.field("exchange",pa.int32()),pa.field("conditions",pa.list_(pa.int32())),
                       pa.field("correction",pa.int32()),pa.field("sequence_number",pa.int64())]
        for record in records:
            if not isinstance(record,dict):
                raise ValueError("Provider result must be an object")
            ns = integer(record["t"]) * 10**6 if query.dataset == "daily" else integer(record["sip_timestamp"])
            day = trading_date(ns)
            if not query.start <= str(day) <= query.end:
                raise ValueError("Provider result lies outside requested New York dates")
            row = dict(ticker=query.ticker,session_date=day,timestamp_ns=ns)
            if query.dataset == "daily":
                row.update({name:number(record[key],positive=name!="volume") for key,name in
                            (("o","open"),("h","high"),("l","low"),("c","close"),("v","volume"))})
                if not row["low"] <= min(row["open"],row["close"]) <= max(row["open"],row["close"]) <= row["high"]:
                    raise ValueError("Invalid OHLC envelope")
            else:
                if not isinstance(record.get("id"),str) or not record["id"]:
                    raise ValueError("Trade identity is missing; cannot certify pagination")
                row.update(price=number(record["price"],positive=True),
                           shares=number(record.get("decimal_size",record.get("size"))),
                           trade_id=record["id"],exchange=integer(record["exchange"]),
                           conditions=record.get("conditions",[]),correction=record.get("correction"),
                           sequence_number=record.get("sequence_number"))
            rows.append(row)
        return pa.Table.from_pylist(rows,schema=pa.schema(fields))
    for record in records:
        if not isinstance(record,dict) or record.get("ticker") != query.ticker:
            raise ValueError("Reference result ticker differs from requested US ticker")
        if query.dataset == "tickers":
            if record.get("locale") != "us" or record.get("market") != "stocks":
                raise ValueError("Reference result is not a US stock")
            day = query.start
        else:
            field = "execution_date" if query.dataset == "splits" else "ex_dividend_date"
            day = record[field]
            if str(date.fromisoformat(day)) != day:
                raise ValueError("Invalid corporate action date")
            if not query.start <= day <= query.end:
                raise ValueError("Corporate event outside requested range")
            if query.dataset == "splits":
                number(record["split_from"],positive=True); number(record["split_to"],positive=True)
            else:
                number(record["cash_amount"])
        rows.append(dict(ticker=query.ticker,event_date=day,record_json=json.dumps(record,default=str,sort_keys=True)))
    return pa.Table.from_pylist(rows,schema=pa.schema([("ticker",pa.string()),("event_date",pa.string()),("record_json",pa.string())]))
