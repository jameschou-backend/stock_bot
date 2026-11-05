"""Calendar-aligned, close-known features for the frozen US study."""
from dataclasses import dataclass
import io
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .research_data import protocol
from .store import encoded, sha

ARMS = ("breakout_control", "rs_breakout", "obv_breakout", "contraction_breakout")


@dataclass
class Panel:
    dates: pd.Index
    symbols: list
    fields: dict
    universe: list
    audit: dict


def load_panel(store):
    spec = protocol()
    if store.read_json("protocol.json") != spec:
        raise ValueError("Frozen protocol differs from code; use the matching research version")
    bundle = store.read_json("inputs/bundle.json")
    if not bundle["complete_acquisition"] or bundle["protocol_sha256"] != sha(encoded(spec)):
        raise ValueError("Incomplete or mismatched research bundle")
    names = json.loads(store.read_object(bundle["universe"]))["stocks"]
    calendar = json.loads(store.read_object(bundle["calendar"]))
    dates = pd.Index([row["date"] for row in calendar], name="date")
    symbols = sorted({row["ticker"] for row in names} | {spec["benchmark"]})
    fields = {name: pd.DataFrame(np.nan, index=dates, columns=symbols)
              for name in ("open", "high", "low", "close", "volume", "turnover")}
    coverage, loaded, row_count = {}, set(), {"all": 0, "split": 0}
    for receipt in bundle["bars"]:
        if not receipt["complete_pagination"]:
            raise ValueError("Unfinished bars pagination")
        query = receipt["query"]
        frame = pq.read_table(io.BytesIO(store.read_object(receipt["parquet"]))).to_pandas()
        if len(frame) != receipt["rows"] or frame.duplicated(["ticker", "date"]).any():
            raise ValueError("Bar count or identity mismatch")
        adjustment = query["adjustment"]
        row_count[adjustment] += len(frame)
        for symbol in query["symbols"]:
            if (symbol, adjustment) in loaded or symbol not in symbols:
                raise ValueError("Duplicated or unexpected research symbol batch")
            loaded.add((symbol, adjustment))
            one = frame.loc[frame.ticker == symbol].set_index("date")
            if not one.index.isin(dates).all():
                raise ValueError("Bar date outside market calendar")
            coverage[symbol, adjustment] = tuple(sorted(one.index))
            if adjustment == "all":
                for field in ("open", "high", "low", "close", "volume"):
                    fields[field].loc[one.index, symbol] = one[field]
            else:
                fields["turnover"].loc[one.index, symbol] = one.close * one.volume
    if loaded != {(s, a) for s in symbols for a in spec["adjustments"]}:
        raise ValueError("Missing symbol/adjustment batch")
    for symbol in symbols:
        if coverage[symbol, "all"] != coverage[symbol, "split"]:
            raise ValueError("Paired adjustments have different calendar coverage")
    benchmark = spec["benchmark"]
    if fields["close"][benchmark].isna().any():
        raise ValueError("Benchmark calendar coverage is incomplete")
    stocks = [row["ticker"] for row in names]
    valid = fields["close"][stocks].notna()
    returns = fields["close"][stocks].pct_change(fill_method=None)
    audit = {
        "universe_date": spec["universe_date"], "cohort_stocks": len(stocks),
        "calendar_sessions": len(dates), "first_session": dates[0], "last_session": dates[-1],
        "rows_by_adjustment": row_count,
        "symbols_without_bars": [s for s in stocks if not valid[s].any()],
        "symbols_missing_final_session": [s for s in stocks if not valid[s].iloc[-1]],
        "large_one_day_moves_over_100pct": int((returns.abs() > 1).sum().sum()),
        "future_completeness_filter_used": False, "corporate_action_vintage": "current",
        "delisting_proceeds_verified": False, "symbol_continuity_verified": False,
        "live_qualified": False, "qualification": spec["qualification"],
    }
    return Panel(dates, symbols, fields, stocks, audit)


def features(panel, spec=None):
    spec = spec or protocol()
    close, high, low, volume = (panel.fields[k] for k in ("close", "high", "low", "volume"))
    valid = close.notna() & volume.gt(0)
    continuous = valid.rolling(spec["warmup_sessions"]).sum().eq(spec["warmup_sessions"])
    adv = panel.fields["turnover"].rolling(20).mean()
    eligible = continuous & close.gt(close.rolling(spec["warmup_sessions"]).mean()) & adv.ge(spec["min_adv20_usd"])
    eligible.loc[:, ~eligible.columns.isin(panel.universe)] = False
    rs = close.pct_change(spec["relative_strength_sessions"], fill_method=None)
    rs = rs.sub(rs[spec["benchmark"]], axis=0)
    breakout = close.gt(high.shift(1).rolling(spec["breakout_sessions"]).max())
    signed_volume = np.sign(close.diff()).mul(volume).where(valid)
    obv = signed_volume.fillna(0).cumsum().where(valid)
    obv_breakout = obv.gt(obv.shift(1).rolling(spec["obv_sessions"]).max())
    window = spec["bollinger_sessions"]
    width = 2 * spec["bollinger_std"] * close.rolling(window).std(ddof=0) / close.rolling(window).mean()
    threshold = width.shift(2).rolling(spec["contraction_history"]).quantile(spec["contraction_quantile"])
    contraction = width.shift(1).le(threshold)
    expansion = volume.ge(spec["contraction_volume_ratio"] * volume.shift(1).rolling(20).mean())
    raw = {
        "breakout_control": breakout,
        "rs_breakout": breakout & rs.gt(0),
        "obv_breakout": obv_breakout,
        "contraction_breakout": breakout & contraction & expansion,
    }
    signals = {name: condition & ~condition.shift(1, fill_value=False) & eligible for name, condition in raw.items()}
    return {"signals": signals, "eligible": eligible, "rs": rs, "adv": adv,
            "exit_low": low.shift(1).rolling(spec["exit_low_sessions"]).min()}


def ranked_indices(mask, relative_strength, adv, symbols):
    indices = np.flatnonzero(mask)
    return sorted(indices, key=lambda j: (-relative_strength[j], -adv[j], symbols[j]))
