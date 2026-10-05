"""Causal daily OHLCV research rules with explicit indicator conventions.

Sources are the chart vendor's own indicator documentation and John Bollinger's
own rules, linked in PUBLIC_CATALOG (read 2026-10-05). These are our fixed signal
combinations, not claims to reproduce any author's complete trading system.

All decisions use T-close information; execution is outside this module and can
start no earlier than T+1. Arrays must use the complete market calendar. Missing
or ineligible sessions break every window. EMA seeds with n consecutive inputs'
SMA, then alpha=2/(n+1); Wilder uses the same seed and alpha=1/n. An input gap
resets either smoother and requires a fresh seed. This convention is reproducible
but need not match a chart platform with a different history start. TR and price
changes require an observed prior session (the first bar is not imputed).

Zero denominators remain unknown except explicitly observed one-sided RSI/MFI
(0/100) and zero directional movement with positive TR (DX=0). OBV restarts at
an arbitrary zero origin after a gap and requires a whole new comparison window.
No future shifts, centered windows, backfills, holdings or account limits are used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


VERSION = "scanner_public_v1_20261005"
MIN_AMOUNT20 = 50_000_000
PRICE_ROUNDING_RTOL = 1e-12
_BASE = "https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/"
_IND = _BASE + "technical-indicators/"
_OVER = _BASE + "technical-overlays/"
_MA = _OVER + "moving-averages-simple-and-exponential"
_ATR = _IND + "average-true-range-atr"
_SOURCE = "skills/strategy_scanner/public_rules.py"


def _meta(sid, name, family, description, params, urls, regimes=("trend_up", "range")):
    return dict(
        id=sid, name=name, family=family, kind="entry", status="active",
        description=description + "；20 日平均估計成交值至少 5,000 萬，收盤確認後最早次交易日執行。",
        required_data=["adjusted_ohlc", "raw_volume", "raw_close_times_volume", "complete_market_calendar", "historical_eligibility"],
        preferred_regimes=list(regimes), version=VERSION, source_paths=[_SOURCE],
        source_urls=list(urls), variants=[], params=dict(params, amount20_min=MIN_AMOUNT20),
        data_gaps=["暖機不足、缺行情或資格未知時保留未知；技術指標不代表已證實獲利，也不辨識真實籌碼持有人。"],
        reusable_interfaces=["add_public_rules"], live_qualified=False,
        returns_inherited=False, source_checked_on="2026-10-05",
        signal_timing="T_close_confirmed__earliest_next_market_session",
        convention="SMA seeded EMA/Wilder; contiguous-session windows; no missing-value fill",
        strategy_attribution="Project-defined fixed combination of documented indicators; not the author's complete system",
    )


PUBLIC_CATALOG = [
    _meta("ma20_60_cross", "MA20／60 金叉", "momentum",
          "今日 MA20 高於 MA60，昨日 MA20 不高於 MA60；只在向上交叉日成立",
          {"fast_sma": 20, "slow_sma": 60}, [_MA]),
    _meta("macd12_26_cross", "MACD 訊號線金叉", "momentum",
          "EMA12 減 EMA26 的 MACD 線由不高於轉為高於自身 EMA9 訊號線；不是 EMA12／26 零軸交叉",
          {"fast_ema": 12, "slow_ema": 26, "signal_ema": 9, "seed": "SMA"},
          [_IND + "macd-moving-average-convergence-divergence-oscillator", _MA]),
    _meta("rsi14_reclaim30", "RSI14 收復 30", "mean_reversion",
          "Wilder RSI14 昨日低於 30、今日至少 30；全程無漲跌的零分母保持未知",
          {"period": 14, "threshold": 30, "smoothing": "Wilder_SMA_seed"},
          [_IND + "relative-strength-index-rsi"], ("range",)),
    _meta("stochastic14_3_cross", "慢速 KD 低檔金叉", "mean_reversion",
          "14 日快 K 經 3 日均線成慢 K，再 3 日均線成 D；昨日 K、D 均低於 20，今日 K 向上穿越 D",
          {"lookback": 14, "k_sma": 3, "d_sma": 3, "prior_oversold": 20},
          [_IND + "stochastic-oscillator-fast-slow-and-full"], ("range",)),
    _meta("bollinger_squeeze_breakout", "布林收斂後上破", "price_breakout",
          "昨日 20 日布林帶寬為當時近 120 日最低，今日收盤由軌內上穿上軌；20 日均線與兩倍母體標準差，120 日為本研究固定參數",
          {"period": 20, "std_multiplier": 2, "std_ddof": 0, "squeeze_lookback": 120},
          [_IND + "bollinger-bandwidth", "https://www.bollingerbands.com/bollinger-band-rules"]),
    _meta("keltner20_breakout", "Keltner 通道上破", "price_breakout",
          "收盤由通道內上穿 EMA20＋2×Wilder ATR20；ATR20 為本研究設定，並非宣稱採用看盤軟體預設 ATR10",
          {"ema_period": 20, "atr_period": 20, "atr_multiplier": 2, "seed": "SMA"},
          [_OVER + "keltner-channels", _ATR]),
    _meta("adx14_di_cross", "ADX 趨勢確認＋DI 金叉", "momentum",
          "+DI14 今日向上穿越 −DI14，且 Wilder ADX14 至少 25；ADX 衡量強度，不單獨決定方向",
          {"dm_period": 14, "adx_period": 14, "adx_min": 25, "smoothing": "Wilder_SMA_seed"},
          [_IND + "average-directional-index-adx"], ("trend_up",)),
    _meta("ichimoku_cloud_breakout", "一目均衡表過雲", "price_breakout",
          "收盤向上穿越當日雲頂且轉換線高於基準線；雲使用 26 日前已算好的先行帶，不使用回畫到過去的遲行線",
          {"conversion": 9, "base": 26, "span_b": 52, "forward_plot_shift": 26},
          [_OVER + "ichimoku-cloud"], ("trend_up",)),
    _meta("obv20_breakout", "OBV 突破 20 日高點", "volume_flow",
          "漲日加量、跌日減量的 OBV 今日高於此前 20 日最高，昨日尚未突破其前 20 日高點；這是量價指標，不是法人淨買超",
          {"lookback": 20, "segment_origin": 0}, [_IND + "on-balance-volume-obv"]),
    _meta("cmf20_cross", "CMF20 由負轉正", "volume_flow",
          "以 K 棒收盤位置乘成交量計算 20 日 CMF，由不高於零轉為正；同高低價的分母為零時保留未知，並非真實資金淨流入",
          {"period": 20, "threshold": 0, "zero_range": "unknown"}, [_IND + "chaikin-money-flow-cmf"]),
    _meta("mfi14_reclaim20", "MFI14 收復 20", "mean_reversion",
          "以典型價方向及成交量計算 14 日 MFI，昨日低於 20、今日至少 20；無方向性金流的零分母為未知",
          {"period": 14, "threshold": 20}, [_IND + "money-flow-index-mfi"], ("range",)),
    _meta("atr14_volatility_breakout", "超過一倍 ATR 的收盤漲幅", "price_breakout",
          "今日收盤嚴格高於昨收＋昨日 Wilder ATR14；ATR 只提供波動單位，本突破規則為研究自定，不代表 Wilder 完整系統",
          {"atr_period": 14, "atr_multiplier": 1, "threshold_asof": "previous_session"}, [_ATR]),
]
PUBLIC_IDS = tuple(row["id"] for row in PUBLIC_CATALOG)


def _smooth(frame: pd.DataFrame, period: int, alpha: float) -> pd.DataFrame:
    """Column-parallel SMA seed then recursive smoothing; missing resets state."""
    values = frame.to_numpy(dtype=float)
    out = np.full_like(values, np.nan)
    count = np.zeros(values.shape[1], dtype=np.int64)
    total = np.zeros(values.shape[1], dtype=float)
    state = np.full(values.shape[1], np.nan)
    for i, row in enumerate(values):
        good = np.isfinite(row)
        count[~good] = 0
        total[~good] = 0.0
        state[~good] = np.nan
        advancing = good & (count >= period)
        warming = good & ~advancing
        total[warming] += row[warming]
        count[warming] += 1
        seed = warming & (count == period)
        state[seed] = total[seed] / period
        state[advancing] += alpha * (row[advancing] - state[advancing])
        out[i] = state
    return pd.DataFrame(out, index=frame.index, columns=frame.columns)


def _ema(frame, period):
    return _smooth(frame, period, 2.0 / (period + 1))


def _wilder(frame, period):
    return _smooth(frame, period, 1.0 / period)


def _obv(close, volume):
    """Rebase each contiguous observed segment; first bar's level is arbitrary."""
    values, vols = close.to_numpy(float), volume.to_numpy(float)
    out = np.full_like(values, np.nan)
    prev = np.full(values.shape[1], np.nan)
    state = np.full(values.shape[1], np.nan)
    for i, (row, vol) in enumerate(zip(values, vols)):
        good = np.isfinite(row) & np.isfinite(vol)
        continuation = good & np.isfinite(prev)
        starting = good & ~continuation
        state[~good] = np.nan
        state[starting] = 0.0
        state[continuation] += np.sign(row[continuation] - prev[continuation]) * vol[continuation]
        out[i] = state
        prev = np.where(good, row, np.nan)
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def _positive_ratio(positive, negative):
    """100*p/(p+n), including observed one-sided 0/100, never fill 0/0."""
    denominator = positive + negative
    return 100 * positive / denominator.where(denominator.gt(0))


def _inputs(f):
    required = ("c", "h", "l", "v", "a", "valid", "eligible")
    if any(key not in f for key in required):
        raise ValueError("Public rules require aligned c/h/l/v/a/valid/eligible matrices")
    reference = f["c"]
    if not isinstance(reference, pd.DataFrame):
        raise ValueError("Public rule inputs must be wide DataFrames")
    for key in required:
        if (not isinstance(f[key], pd.DataFrame) or not reference.index.equals(f[key].index)
                or not reference.columns.equals(f[key].columns)):
            raise ValueError("Public rule matrices must share the complete session calendar and stock columns")
    if (not reference.index.is_unique or not reference.index.is_monotonic_increasing
            or not reference.columns.is_unique):
        raise ValueError("Public rule axes must be ordered and unique")
    good = f["valid"].eq(True) & f["eligible"].eq(True)
    for key in ("c", "h", "l", "v", "a"):
        good &= np.isfinite(f[key])
    # An equal raw high/close may differ by an ulp after raw_high*(adj/raw).
    # Permit only relative arithmetic precision, not a tick or price tolerance;
    # normalize these already-approved boundary contacts before computing ranges.
    c, h, l = f["c"], f["h"], f["l"]
    tolerance = np.maximum(c.abs(), np.maximum(h.abs(), l.abs())) * PRICE_ROUNDING_RTOL
    good &= (c.gt(0) & l.gt(0) & h.ge(c-tolerance) & l.le(c+tolerance)
             & f["v"].ge(0) & f["a"].ge(0))
    h = h.where(h.ge(c), c)
    l = l.where(l.le(c), c)
    return tuple(frame.where(good) for frame in (c, h, l, f["v"], f["a"]))


def add_public_rules(f, z, add):
    """Add features and 12 masks via add(id, match, fields, rule, known=...).

    ``fields`` contains both current and previous operands, so a comparison
    involving missing data cannot silently become a known negative signal.
    Inputs are never mutated. New feature names use ``pub_`` to avoid collisions.
    No external requests or indicator dependencies are needed.
    """
    c, h, l, v, amount = _inputs(f)
    features = {}
    def put(name, frame):
        key = "pub_" + name
        if key in z:
            raise ValueError("Public indicator feature already registered: " + key)
        features[key] = frame
        return frame
    def previous(name):
        return put("prior_" + name, features["pub_" + name].shift(1))
    amount20 = put("amount20", amount.rolling(20, min_periods=20).mean())
    liquid = amount20.ge(MIN_AMOUNT20)
    ma20 = put("ma20", c.rolling(20, min_periods=20).mean())
    ma60 = put("ma60", c.rolling(60, min_periods=60).mean())
    previous("ma20"); previous("ma60")
    put("c", c); previous("c")
    ema12, ema26 = _ema(c, 12), _ema(c, 26)
    macd = put("macd", ema12 - ema26)
    macd_signal = put("macd_signal", _ema(macd, 9))
    previous("macd"); previous("macd_signal")

    change = c.diff()
    rsi = put("rsi14", _positive_ratio(_wilder(change.clip(lower=0), 14),
                                      _wilder(-change.clip(upper=0), 14)))
    previous("rsi14")
    highest14, lowest14 = h.rolling(14, min_periods=14).max(), l.rolling(14, min_periods=14).min()
    width14 = highest14 - lowest14
    fast_k = 100 * (c - lowest14) / width14.where(width14.gt(0))
    slow_k = put("slow_k", fast_k.rolling(3, min_periods=3).mean())
    slow_d = put("slow_d", slow_k.rolling(3, min_periods=3).mean())
    previous("slow_k"); previous("slow_d")
    std20 = c.rolling(20, min_periods=20).std(ddof=0)
    bb_upper = put("bb_upper", ma20 + 2 * std20)
    bandwidth = put("bandwidth", 4 * std20 / ma20.where(ma20.gt(0)))
    put("prior_bandwidth_min120", bandwidth.rolling(120, min_periods=120).min().shift(1))
    previous("bandwidth"); previous("bb_upper")

    # np.maximum preserves unknown prior close; DataFrame.max(skipna=True) would not.
    prior_c = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prior_c).abs(), (l - prior_c).abs()))
    atr14 = put("atr14", _wilder(tr, 14)); previous("atr14")
    atr20 = _wilder(tr, 20)
    kc_upper = put("kc_upper", _ema(c, 20) + 2 * atr20)
    previous("kc_upper")
    up, down = h.diff(), -l.diff()
    movement_known = np.isfinite(up) & np.isfinite(down) & np.isfinite(tr)
    plus_dm = up.where(up.gt(down) & up.gt(0), 0.0).where(movement_known)
    minus_dm = down.where(down.gt(up) & down.gt(0), 0.0).where(movement_known)
    plus_di = put("plus_di14", 100 * _wilder(plus_dm, 14) / atr14.where(atr14.gt(0)))
    minus_di = put("minus_di14", 100 * _wilder(minus_dm, 14) / atr14.where(atr14.gt(0)))
    di_sum = plus_di + minus_di
    dx = (100 * (plus_di - minus_di).abs() / di_sum.where(di_sum.gt(0))).where(di_sum.ne(0), 0.0)
    adx = put("adx14", _wilder(dx, 14))
    previous("plus_di14"); previous("minus_di14")

    def mid(period):
        return (h.rolling(period, min_periods=period).max() + l.rolling(period, min_periods=period).min()) / 2
    conversion = put("conversion9", mid(9))
    base = put("base26", mid(26))
    span_a = put("cloud_a", ((conversion + base) / 2).shift(26))
    span_b = put("cloud_b", mid(52).shift(26))
    # Both cloud sides must exist; no skipna-based maximum or Chikou backshift.
    cloud_top = put("cloud_top", np.maximum(span_a, span_b))
    previous("cloud_top")
    obv = put("obv", _obv(c, v))
    put("obv_high20", obv.rolling(20, min_periods=20).max().shift(1))
    previous("obv"); previous("obv_high20")
    spread = h - l
    multiplier = (2 * c - h - l) / spread.where(spread.gt(0))
    vol20 = v.rolling(20, min_periods=20).sum()
    cmf = put("cmf20", (multiplier * v).rolling(20, min_periods=20).sum() / vol20.where(vol20.gt(0)))
    previous("cmf20")
    typical = (h + l + c) / 3
    money = typical * v
    typical_change = typical.diff()
    flow_known = np.isfinite(typical_change) & np.isfinite(money)
    positive = money.where(typical_change.gt(0), 0.0).where(flow_known)
    negative = money.where(typical_change.lt(0), 0.0).where(flow_known)
    mfi = put("mfi14", _positive_ratio(positive.rolling(14, min_periods=14).sum(),
                                       negative.rolling(14, min_periods=14).sum()))
    previous("mfi14")

    # Publish all shared features once, before registering rule operands.
    z.update(features)
    descriptions = {row["id"]: row["description"] for row in PUBLIC_CATALOG}
    def register(sid, match, names):
        fields = ["pub_amount20", *("pub_" + name for name in names)]
        add(sid, match & liquid, fields, descriptions[sid])
    p = lambda key: features["pub_prior_" + key]
    register("ma20_60_cross", ma20.gt(ma60) & p("ma20").le(p("ma60")),
             ["ma20", "ma60", "prior_ma20", "prior_ma60"])
    register("macd12_26_cross", macd.gt(macd_signal) & p("macd").le(p("macd_signal")),
             ["macd", "macd_signal", "prior_macd", "prior_macd_signal"])
    register("rsi14_reclaim30", rsi.ge(30) & p("rsi14").lt(30), ["rsi14", "prior_rsi14"])
    register("stochastic14_3_cross", slow_k.gt(slow_d) & p("slow_k").le(p("slow_d"))
             & p("slow_k").lt(20) & p("slow_d").lt(20),
             ["slow_k", "slow_d", "prior_slow_k", "prior_slow_d"])
    register("bollinger_squeeze_breakout", c.gt(bb_upper) & p("c").le(p("bb_upper"))
             & p("bandwidth").le(features["pub_prior_bandwidth_min120"]),
             ["c", "prior_c", "bb_upper", "prior_bb_upper", "prior_bandwidth", "prior_bandwidth_min120"])
    register("keltner20_breakout", c.gt(kc_upper) & p("c").le(p("kc_upper")),
             ["c", "prior_c", "kc_upper", "prior_kc_upper"])
    register("adx14_di_cross", plus_di.gt(minus_di) & p("plus_di14").le(p("minus_di14")) & adx.ge(25),
             ["plus_di14", "minus_di14", "prior_plus_di14", "prior_minus_di14", "adx14"])
    register("ichimoku_cloud_breakout", c.gt(cloud_top) & p("c").le(p("cloud_top")) & conversion.gt(base),
             ["c", "prior_c", "cloud_top", "prior_cloud_top", "conversion9", "base26"])
    register("obv20_breakout", obv.gt(features["pub_obv_high20"]) & p("obv").le(p("obv_high20")),
             ["obv", "obv_high20", "prior_obv", "prior_obv_high20"])
    register("cmf20_cross", cmf.gt(0) & p("cmf20").le(0), ["cmf20", "prior_cmf20"])
    register("mfi14_reclaim20", mfi.ge(20) & p("mfi14").lt(20), ["mfi14", "prior_mfi14"])
    register("atr14_volatility_breakout", c.gt(p("c") + p("atr14")) & p("atr14").gt(0),
             ["c", "prior_c", "prior_atr14"])
