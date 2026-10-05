"""Compute once, evaluate many strategies; never read positions or future returns.

This scanner produces close-confirmed setups, not executable orders. Unknown
observations stay unknown. Execution, capital allocation and exits are separate.
"""
from __future__ import annotations

from collections import Counter
import re

import numpy as np
import pandas as pd

BASE_IDS = ('original_breakout', 'original_red', 'poc_red_priority', 'poc_up_red',
              'momentum', 'risk_momentum', 'near_high', 'contraction_breakout',
              'donchian20', 'donchian55', 'bollinger_reclaim', 'ma_pullback')
from .public_rules import PUBLIC_IDS, add_public_rules
from .research_rules import RESEARCH_IDS, add_research_rules

ACTIVE_IDS = BASE_IDS + PUBLIC_IDS + RESEARCH_IDS
STATES = ('matched', 'not_matched', 'unknown', 'ineligible')


def _day(value):
    d = pd.Timestamp(value)
    if pd.isna(d) or d.tz is not None or d != d.normalize():
        raise ValueError('Dates must be naive market dates')
    return d


def _number(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def _prepare(bars, calendar, end):
    required = {'date', 'stock_id', 'open', 'high', 'low', 'close', 'volume',
                'amount', 'adjusted_close', 'quality', 'eligible'}
    if not required.issubset(bars.columns):
        raise ValueError('Missing scanner columns: '+', '.join(sorted(required-set(bars.columns))))
    days = pd.DatetimeIndex(calendar)
    if (days.hasnans or days.tz is not None or days.has_duplicates
            or not days.is_monotonic_increasing or not days.equals(days.normalize())):
        raise ValueError('Calendar must contain sorted unique market dates')
    days = days[days <= end]
    b = bars.copy()
    b['date'] = pd.to_datetime(b.date, errors='raise')
    if b.date.dt.tz is not None or not b.date.eq(b.date.dt.normalize()).all():
        raise ValueError('Bars require market dates without clock times')
    b = b[b.date <= end].copy()  # Future rows cannot influence any feature or universe.
    if b.duplicated(['date', 'stock_id']).any():
        raise ValueError('Duplicate stock/date observations')
    if not b.stock_id.map(lambda s: isinstance(s, str) and re.fullmatch(r'\d{4}', s) is not None).all():
        raise ValueError('Stock identifiers must be four-digit strings')
    if not b.date.isin(days).all():
        raise ValueError('Quote date is absent from supplied market calendar')
    for field in ('quality', 'eligible'):
        if not b[field].dropna().map(lambda v: isinstance(v, (bool, np.bool_))).all():
            raise ValueError(field+' must contain booleans or explicitly missing values')
    ids = sorted(set(b.stock_id))
    f = {}
    for field in sorted(required-{'date', 'stock_id'}):
        f[field] = b.pivot(index='date', columns='stock_id', values=field).reindex(index=days, columns=ids)
    for field in ('open', 'high', 'low', 'close', 'volume', 'amount', 'adjusted_close'):
        f[field] = f[field].astype(float)
    observed = b.assign(observed=True).pivot(index='date', columns='stock_id', values='observed')
    f['observed'] = observed.reindex(index=days, columns=ids).notna()
    valid = (f['quality'].eq(True) & f['close'].gt(0) & f['open'].gt(0)
             & f['volume'].ge(0) & f['amount'].ge(0) & f['adjusted_close'].gt(0)
             & f['high'].ge(f['close']) & f['high'].ge(f['open'])
             & f['low'].le(f['close']) & f['low'].le(f['open']) & f['low'].gt(0))
    for field in ('open', 'high', 'low', 'close', 'volume', 'amount', 'adjusted_close'):
        valid &= np.isfinite(f[field])
    if 'source_disagreement' in b:
        disagreement = b.pivot(index='date', columns='stock_id', values='source_disagreement').reindex(index=days, columns=ids)
        valid &= ~disagreement.eq(True)
    valid = valid.fillna(False)
    f['valid'] = valid
    # Ineligible or missing market sessions break windows, never compress them.
    mask = valid & f['eligible'].eq(True).fillna(False)
    c = f['adjusted_close'].where(mask)
    factor = c / f['close']
    f['c'], f['h'], f['l'] = c, f['high']*factor, f['low']*factor
    f['v'] = f['volume'].where(mask)
    f['a'] = f['amount'].where(mask)
    return f, days, ids


def _features(f):
    c, h, l, v = (f[k] for k in ('c','h','l','v'))
    z = {'c': c, 'red': f['close'].gt(f['open']), 'amount20': f['a'].rolling(20, min_periods=20).mean()}
    for n in (10, 20, 60, 120):
        z['ma'+str(n)] = c.rolling(n, min_periods=n).mean()
    z['prior_high20'] = h.rolling(20, min_periods=20).max().shift(1)
    z['prior_high55'] = h.rolling(55, min_periods=55).max().shift(1)
    z['volume20'] = v.rolling(20, min_periods=20).mean().shift(1)
    z['volume_ratio'] = v/z['volume20'].where(z['volume20'].gt(0))
    z['ret20'] = c/c.shift(20)-1
    z['ret63'] = c/c.shift(63)-1
    z['momentum'] = c.shift(21)/c.shift(126)-1
    z['vol126'] = c.pct_change(fill_method=None).rolling(126, min_periods=126).std()*np.sqrt(252)
    z['risk_momentum'] = z['momentum']/z['vol126'].where(z['vol126'].gt(0))
    z['high252'] = c.rolling(252, min_periods=252).max()
    recent = h.rolling(10, min_periods=10).max().shift(1)-l.rolling(10, min_periods=10).min().shift(1)
    earlier = h.rolling(10, min_periods=10).max().shift(11)-l.rolling(10, min_periods=10).min().shift(11)
    z['contraction_ratio'] = recent/earlier.where(earlier.gt(0))
    z['bb_lower'] = z['ma20']-2*c.rolling(20, min_periods=20).std(ddof=0)
    z['prior_c'], z['prior_bb_lower'] = c.shift(1), z['bb_lower'].shift(1)
    z['prior_ma20'] = z['ma20'].shift(1)
    z['prior_ma60'] = z['ma60'].shift(1)
    z['low'] = l
    z['uptrend'] = c.gt(z['ma20']) & z['ma20'].gt(z['ma60'])
    z['downtrend'] = c.lt(z['ma20']) & z['ma20'].lt(z['ma60'])
    z['regime_known'] = c.notna() & z['ma20'].notna() & z['ma60'].notna()
    return z


def _evidence_matrices(f, days, ids, original_signals, poc):
    origin = pd.DataFrame(False, index=days, columns=ids)
    priorities = pd.DataFrame(np.nan, index=days, columns=ids)
    seen = set()
    for event in original_signals:
        d = _day(event['signal_date'])
        if d > days[-1]:
            continue
        members = event.get('members', [event.get('stock_id')])
        if len(members) != 1 or not isinstance(members[0], str):
            raise ValueError('Original individual-stock candidates must have one member')
        sid = members[0]
        key = (d, sid)
        if key in seen:
            raise ValueError('Duplicate original candidate coordinate')
        seen.add(key)
        if d in days and sid in ids:
            origin.at[d, sid] = True
            priorities.at[d, sid] = event.get('priority', np.nan)
    pstate = pd.DataFrame('not_covered', index=days, columns=ids)
    before, after = (pd.DataFrame(np.nan, index=days, columns=ids) for _ in range(2))
    seen.clear()
    for row in poc or []:
        d, sid = _day(row['signal_date']), row['stock_id']
        if d > days[-1] or d not in days or sid not in ids:
            continue
        if (d, sid) in seen:
            raise ValueError('Duplicate daily POC coordinate')
        seen.add((d, sid))
        # A profile that reaches into its signal day changes the sealed rule.
        if row.get('source_date_end') is not None and _day(row['source_date_end']) >= d:
            raise ValueError('POC profile must end before its signal date')
        state = row.get('status')
        if state not in ('up', 'down', 'unknown', 'pending_data'):
            raise ValueError('Unknown POC status')
        if state in ('up', 'down'):
            if row.get('source_date_end') is None:
                raise ValueError('Known POC requires a dated source cutoff')
            i = days.get_loc(d)
            expected = [str(x.date()) for x in days[max(0, i-20):i]]
            if (len(expected) != 20 or row.get('prior_dates') != expected
                    or row['source_date_end'] != expected[-1]
                    or row.get('available') is not True):
                raise ValueError('Known POC requires the exact previous 20 market sessions')
            x, y = row.get('poc_before'), row.get('poc_after')
            if not all(isinstance(a, (int,float)) and np.isfinite(a) and a>0 for a in (x,y)):
                raise ValueError('Known POC needs finite before/after prices')
            if (y>x) != (state=='up'):
                raise ValueError('POC direction contradicts prices')
            before.at[d, sid], after.at[d, sid] = x, y
        pstate.at[d,sid] = state
    return origin, priorities, pstate, before, after


def _compile_rules(f, days, ids, *, original_signals=(), poc=None, provenance=None):
    """Shared causal rule matrices used by scanning and separate outcome studies."""
    from .catalog import get_catalog
    provenance = dict(provenance or {})
    z = _features(f)
    origin, priorities, pstate, before, after = _evidence_matrices(f, days, ids, original_signals, poc)
    catalog = get_catalog()
    active = {s['id']:s for s in catalog if s['status']=='active'}
    if set(active) != set(ACTIVE_IDS):
        raise ValueError('Active catalog and engine implementations differ')
    masks = {}
    def add(sid, match, fields, rule, *, known=None):
        if sid in masks:
            raise ValueError('Duplicate strategy evaluator: '+sid)
        available = f['valid'].copy()
        for field in fields:
            available &= z[field].notna() & np.isfinite(z[field])
        if known is not None:
            available &= known
        masks[sid] = (match, available, fields, rule)
    original_known = pd.DataFrame(bool(provenance.get('original_candidates_complete', False)), index=days, columns=ids)
    for field, op in (('original_signal_start', 'before'), ('original_signal_end', 'after')):
        if provenance.get(field):
            outside = days < _day(provenance[field]) if op == 'before' else days > _day(provenance[field])
            original_known.loc[outside, :] = False
    add('original_breakout', origin, [], '符合原封存突破量增候選規則', known=original_known)
    red = origin & z['red']
    add('original_red', red, [], '原突破候選且訊號日收紅 K', known=original_known)
    add('poc_red_priority', red, [], '原紅 K 候選；POC僅提供優先排序，不是進場硬門檻', known=original_known)
    # For non-candidates, POC is irrelevant: known false, not a fabricated profile.
    add('poc_up_red', red & pstate.eq('up'), [], '原紅 K 候選且先前20日POC上移（新增硬篩版）',
        known=original_known & (~red | pstate.isin(['up','down'])))
    liquid = z['amount20'].ge(50_000_000)
    trend = z['c'].gt(z['ma120']) & z['momentum'].gt(0) & liquid
    add('momentum', trend, ['momentum','ma120','amount20'], '近126至21日動能為正、站上120日線、20日均成交值估算≥5千萬')
    add('risk_momentum', trend & z['risk_momentum'].notna(), ['risk_momentum','ma120','amount20'],
        '中期動能條件成立；以動能／126日波動作獨立排序，不跨策略比原始分數')
    add('near_high', z['c'].ge(z['high252']*.95) & z['ma60'].gt(z['ma120']) & z['ret63'].gt(0) & liquid,
        ['high252','ma60','ma120','ret63','amount20'], '距252日高點≤5%、60日線在120日線上、63日漲幅為正、均成交值≥5千萬')
    breakout = z['c'].gt(z['prior_high20']) & z['volume_ratio'].ge(1.5)
    add('contraction_breakout', breakout & z['contraction_ratio'].le(.75) & liquid,
        ['prior_high20','volume_ratio','contraction_ratio','amount20'], '前10日區間縮至再前10日的75%以下，再突破20日高點且量≥1.5倍；流動性門檻5千萬')
    for n in (20,55):
        key='prior_high'+str(n)
        add('donchian'+str(n), z['c'].gt(z[key]) & liquid, [key,'amount20'],
            f'收盤突破前{n}日最高價（不含今天），20日均成交值估算≥5千萬')
    add('bollinger_reclaim', z['prior_c'].lt(z['prior_bb_lower']) & z['c'].ge(z['bb_lower']) & z['red'] & liquid,
        ['prior_c','prior_bb_lower','bb_lower','amount20'], '昨收在布林下軌外、今日收紅且收回下軌內；20日均成交值估算≥5千萬')
    add('ma_pullback', z['ma20'].gt(z['ma60']) & z['prior_ma20'].gt(z['prior_ma60'])
        & z['low'].le(z['ma20']*1.01) & z['c'].gt(z['ma20']) & z['red'] & liquid,
        ['ma20','ma60','prior_ma20','prior_ma60','low','amount20'], '20日線持續高於60日線，今日回測20日線附近後收紅站回；均成交值≥5千萬')
    add_public_rules(f, z, add)
    add_research_rules(f, z, add)
    if set(masks) != set(ACTIVE_IDS):
        raise ValueError("Missing or unregistered strategy evaluator")
    evidence = dict(priorities=priorities, pstate=pstate, before=before, after=after, red=red)
    return z, masks, catalog, evidence


def scan_market(bars, calendar, *, start, end, names=None, original_signals=(),
                poc=None, provenance=None, strategies=None):
    """Return one outcome for every stock/date/active strategy, no account state.

    `original_signals` must be the complete hash-bound candidate ledger, not
    trades. Without that evidence original adapters return unknown, not False.
    Caller supplies the full market calendar and at least 420 prior sessions.
    Publication timestamps for optional external evidence belong in its adapter.
    """
    start, end = _day(start), _day(end)
    if start>end:
        raise ValueError('Scan start must not exceed end')
    f, days, ids = _prepare(bars, calendar, end)
    if end not in days or start not in days:
        raise ValueError('Scan endpoints must be observed market sessions')
    provenance = dict(provenance or {})
    if provenance.get('source_end') and end>_day(provenance['source_end']):
        raise ValueError('Scan exceeds frozen source coverage')
    z, masks, catalog, evidence = _compile_rules(f, days, ids,
        original_signals=original_signals, poc=poc, provenance=provenance)
    active = {s['id']:s for s in catalog if s['status']=='active'}
    chosen = list(ACTIVE_IDS if strategies is None else strategies)
    if len(set(chosen)) != len(chosen) or not chosen or set(chosen)-set(active):
        raise ValueError('Select unique registered active strategies')
    # The matrices already share _prepare's market-calendar/stock axes. Convert
    # lookup views once; no indicator, eligibility or signal rule is recomputed.
    # In particular, retain the preceding market row for first-signal evidence.
    eligible = f['eligible'].eq(True).fillna(False).to_numpy(dtype=bool)
    ineligible = f['eligible'].eq(False).fillna(False).to_numpy(dtype=bool)
    regime_known, uptrend, downtrend = (z[k].to_numpy(copy=False) for k in
        ('regime_known', 'uptrend', 'downtrend'))
    priorities, pstate, before, after, red = (evidence[k].to_numpy(copy=False) for k in
        ('priorities', 'pstate', 'before', 'after', 'red'))
    numeric_fields = {field: z[field].to_numpy(copy=False)
                      for key in chosen for field in masks[key][2]}
    priority_keys = ('original_breakout', 'original_red', 'poc_red_priority', 'poc_up_red')
    poc_keys = ('poc_red_priority', 'poc_up_red')
    lookups = []
    for key in chosen:
        match, known, fields, rule = masks[key]
        lookups.append((key, match.to_numpy(copy=False), known.to_numpy(copy=False),
                        [(field, numeric_fields[field]) for field in fields], rule,
                        active[key].get('preferred_regimes', []), key in priority_keys, key in poc_keys))
    stock_coordinates = [(j, sid) for j, sid in enumerate(ids) if not sid.startswith('0')]
    benchmark_index = ids.index('0050') if '0050' in ids else None
    stock_names = names or {}
    dates = []
    for i in np.flatnonzero((days >= start) & (days <= end)):
        d = days[i]; stocks = []; counts = Counter()
        market = 'unknown'
        if benchmark_index is not None and regime_known[i, benchmark_index]:
            market = ('trend_up' if uptrend[i, benchmark_index] else
                      'trend_down' if downtrend[i, benchmark_index] else 'range')
        for j, sid in stock_coordinates:
            regime = 'unknown'
            if regime_known[i, j]:
                regime = 'trend_up' if uptrend[i, j] else 'trend_down' if downtrend[i, j] else 'range'
            results = {}
            for key, match, known, fields, rule, preferred, has_priority, has_poc in lookups:
                metrics = {field: _number(values[i, j]) for field, values in fields}
                first = None
                if ineligible[i, j]:
                    state = 'ineligible'; reasons = ['該日市場身分／原資料資格不符合個股掃描範圍']
                elif not eligible[i, j] or not known[i, j]:
                    state = 'unknown'; reasons = ['資料不足、品質衝突或歷史窗口不完整，未判定為不符合']
                else:
                    state = 'matched' if match[i, j] else 'not_matched'
                    reasons = [rule if state == 'matched' else '未同時符合：'+rule]
                    if state == 'matched' and i > 0 and known[i-1, j] and eligible[i-1, j]:
                        first = not bool(match[i-1, j])
                if has_priority:
                    metrics['original_priority'] = _number(priorities[i, j])
                if has_poc:
                    metrics.update(poc_before=_number(before[i, j]), poc_after=_number(after[i, j]),
                                   poc_status=pstate[i, j])
                    if red[i, j] and pstate[i, j] not in ('up', 'down'):
                        reasons.append('POC資料不足；不能宣稱籌碼成本已上移')
                results[key] = dict(status=state, reasons=reasons, metrics=metrics, first_signal=first,
                    regime_fit=None if regime == 'unknown' else (regime in preferred if preferred else True))
                counts[state] += 1
            stocks.append(dict(stock_id=sid, name=stock_names.get(sid, sid), regime=regime, results=results))
        dates.append(dict(date=str(d.date()), market_regime=market, stocks=stocks, counts=dict(counts)))
    return dict(schema='multi_strategy_scan_v1',start=str(start.date()),end=str(end.date()),
        source_end=provenance.get('source_end',str(days[-1].date())),strategies=catalog,days=dates,
        evaluated_strategy_ids=chosen,provenance=provenance,live_qualified=False,
        account_independent=True,signal_timing='T_close_confirmed__earliest_next_market_session',
        execution_model=None,portfolio_model=None,returns_inherited=False,
        context_policy='regime_fit_is_descriptive_not_a_validated_strategy_router')
