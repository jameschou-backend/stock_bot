"""Causal filters for an unchanged, externally supplied entry-signal cohort.

These conditions never create candidate dates or read outcomes. The caller must
intersect them with its sealed original-red known-first stock/date coordinates.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from skills.smc_research import PRICE_VALIDATION_RTOL, _price_ge, compute_setups

CONDITION_IDS = ('high400', 'course400', 'rs20', 'smc_break5', 'fvg_form5',
                 'high400_and_rs20', 'high400_and_smc_break5')


def compute_conditions(f):
    """Return seven named filters with separate aligned ``matched``/``known``.

    Unknown always has matched=False. In a conjunction both constituent filters
    must be known, even if one already evaluates false. No zero-prefixed code
    can pass; 0050 remains available solely for relative-strength calculation.
    """
    required = ('c', 'h', 'l', 'open', 'close', 'volume', 'valid', 'eligible')
    if any(k not in f for k in required):
        raise ValueError('Conditional entries require prepared OHLC, volume, validity and eligibility frames')
    c = f['c']
    if not isinstance(c, pd.DataFrame) or not isinstance(c.index, pd.DatetimeIndex):
        raise ValueError('Conditional entries require DataFrames indexed by market dates')
    days, ids = c.index, c.columns
    if (days.hasnans or days.tz is not None or days.has_duplicates
            or not days.is_monotonic_increasing or not days.equals(days.normalize())):
        raise ValueError('Conditional entries require sorted unique naive market dates')
    if ids.has_duplicates or any(not isinstance(s, str) or not re.fullmatch(r'\d{4}', s) for s in ids):
        raise ValueError('Conditional entries require unique four-digit stock identifiers')
    for key in required:
        if (not isinstance(f[key], pd.DataFrame) or not f[key].index.equals(days)
                or not f[key].columns.equals(ids)):
            raise ValueError('Conditional entry frame coordinates differ: ' + key)

    raw, h, l, volume = (f[k] for k in ('close', 'h', 'l', 'volume'))
    adjusted_open = f['open'] * (c / raw)
    good = f['valid'].eq(True).fillna(False) & f['eligible'].eq(True).fillna(False)
    for values in (c, h, l, adjusted_open, raw, volume):
        good &= np.isfinite(values) & values.gt(0)
    good &= (_price_ge(h, c) & _price_ge(h, adjusted_open)
             & _price_ge(c, l) & _price_ge(adjusted_open, l))
    good = good.fillna(False).astype(bool)
    masked_close, masked_volume = c.where(good), volume.where(good)
    known400 = good.rolling(400, min_periods=400).sum().eq(400)
    known64 = good.rolling(64, min_periods=64).sum().eq(64)
    high400 = masked_close.ge(masked_close.rolling(400, min_periods=400).max()*.998)
    max_volume10 = masked_volume.rolling(10, min_periods=10).max()
    turnover20 = (raw*volume).where(good).rolling(20, min_periods=20).mean()
    course400 = high400 & volume.ge(max_volume10) & turnover20.ge(500_000_000)

    known21 = good.rolling(21, min_periods=21).sum().eq(21)
    rs_known = pd.DataFrame(False, index=days, columns=ids)
    relative = pd.DataFrame(np.nan, index=days, columns=ids)
    if '0050' in ids:
        returns = masked_close / masked_close.shift(20) - 1
        relative = returns.sub(returns['0050'], axis='index')
        rs_known = known21.mul(known21['0050'], axis='index').astype(bool)

    # Only causal entry-event dates are consumed. Zone terminal status, retest
    # outcomes and future lifecycle dates must never become entry filters.
    smc = compute_setups(f)
    event_matrices = {strategy: pd.DataFrame(False, index=days, columns=ids)
                      for strategy in ('smc_bull_break', 'fvg_form')}
    for event in smc['events']:
        if event['strategy_id'] in event_matrices:
            event_matrices[event['strategy_id']].at[pd.Timestamp(event['signal_date']), event['stock_id']] = True
    recent = {key: values.rolling(5, min_periods=5).max().eq(1)
              for key, values in event_matrices.items()}

    ordinary = pd.Series([not s.startswith('0') for s in ids], index=ids)
    conditions = {}

    def add(key, match, known, name, definition):
        available = known.fillna(False).astype(bool) & ordinary
        conditions[key] = dict(matched=(match.fillna(False).astype(bool) & available),
            known=available, name=name, definition=definition)

    add('high400', high400, known400, '接近400日收盤高點',
        '今日還原收盤≥含今日400日最高還原收盤的99.8%；400個市場日完整有效。')
    add('course400', course400, known400, '400日價量高點完整條件',
        'high400且今日量≥含今日10日最高量、含今日20日平均原始收盤×成交股數≥5億元。')
    add('rs20', relative.ge(.10), rs_known, '20日相對0050強勢',
        '個股20日還原收盤報酬減0050同期報酬≥10個百分點；兩者21個市場日都有效。')
    add('smc_break5', recent['smc_bull_break'], known64, '近5日出現SMC向上結構突破',
        'T-4至T至少一筆已確認smc_bull_break；64個連續有效市場日；只取訊號日，不讀事後區域狀態。')
    add('fvg_form5', recent['fvg_form'], known64, '近5日形成多頭FVG',
        'T-4至T至少一筆已確認fvg_form；64個連續有效市場日；區域之後失效不回刪形成訊號。')
    for key, other, name in (
        ('high400_and_rs20', 'rs20', '400日高點＋相對強勢'),
        ('high400_and_smc_break5', 'smc_break5', '400日高點＋近期SMC突破'),
    ):
        a, b = conditions['high400'], conditions[other]
        add(key, a['matched'] & b['matched'], a['known'] & b['known'], name,
            f'high400與{other}皆符合且兩者皆已知；任一資料不足都列未知。')
    return dict(conditions=conditions, definitions=dict(
        purpose='filters on unchanged original-red known-first candidates; this module creates no entry dates',
        signal_timing='all windows end at T close, no future outcome or lifecycle status is used',
        price_basis='adjusted daily OHLC; turnover uses raw close times reported share volume',
        missing='every market session retained; invalid, ineligible, nonfinite or nonpositive volume breaks the complete window',
        high400_window='400 market sessions including today',
        recent_event_window='T-4..T inclusive; 64 complete valid sessions required',
        conjunction='both inputs must be known, including when either condition is false',
        benchmark='0050 only for relative strength; missing benchmark leaves rs20 unknown',
        zero_prefixed_codes='never matched; candidate availability false',
        adjusted_ohlc_validation_relative_tolerance=PRICE_VALIDATION_RTOL,
        condition_threshold_tolerance=0,
        strategy_ids=list(CONDITION_IDS),
    ))
