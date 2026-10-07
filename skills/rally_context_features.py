"""Causal, fixed context features for a descriptive first-signal study."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.study_rally_precursors import known_first
from skills.strategy_scanner.engine import _prepare, _compile_rules

COHORTS = ('original_red', 'legacy_course_breakout')
FILTERS = ('not_extended', 'contraction', 'moderate_volume', 'prior_quiet',
           'close_strong', 'rs20_positive', 'market_above60', 'market_breadth',
           'peer_breadth', 'peer_turnover', 'flow_positive_lag1',
           'flow_positive_lag3', 'poc_up', 'clean_price', 'quiet_breakout',
           'peer_and_flow')


def nullable_condition(values, predicate):
    """Unavailable numeric evidence cannot become a negative filter result."""
    values = np.asarray(values, dtype=float)
    return pd.array(np.where(np.isfinite(values), predicate(values), None), dtype='boolean')


def conjunction(*parts):
    result = pd.Series(parts[0], dtype='boolean').copy()
    known = result.notna()
    for part in parts[1:]:
        part = pd.Series(part, dtype='boolean')
        known &= part.notna()
        result &= part
    return result.where(known, pd.NA).array


def build_signal_features(bars, calendar, *, start, end, original_signals, poc, provenance):
    """Return only T-known events/contexts, plus matrices used by outcome stage."""
    f, days, ids = _prepare(bars, calendar, pd.Timestamp(end))
    z, masks, _, evidence = _compile_rules(f, days, ids,
        original_signals=original_signals, poc=poc, provenance=provenance)
    base = f['valid'] & f['eligible'].eq(True).fillna(False) & z['amount20'].ge(50_000_000)
    base.loc[(days < pd.Timestamp(start)) | (days > pd.Timestamp(end))] = False
    base.loc[:, [sid for sid in ids if sid.startswith('0')]] = False
    records = []
    for cohort in COHORTS:
        match, available, _, _ = masks[cohort]
        available = available & f['eligible'].eq(True).fillna(False)
        first = known_first(match, available) & base
        for i, j in zip(*np.where(first.to_numpy(bool))):
            day, sid = str(days[i].date()), ids[j]
            records.append(dict(cohort=cohort, event_id=f'{cohort}-{day}-{sid}',
                stock_id=sid, signal_date=day, signal_index=int(i), column_index=int(j)))
    events = pd.DataFrame(records, columns=['cohort','event_id','stock_id','signal_date','signal_index','column_index'])
    events = events.sort_values(['cohort','signal_index','stock_id']).reset_index(drop=True)
    rows, cols = events.signal_index.to_numpy(int), events.column_index.to_numpy(int)
    sample = lambda matrix: matrix.to_numpy()[rows, cols]
    # Free unrelated indicator matrices before computing additional features.
    keep = {key:z[key] for key in ('ma20','ma60','volume_ratio','contraction_ratio')}
    del z, masks
    numeric = {
        'distance_ma20': sample(f['c']/keep['ma20'] - 1),
        'contraction_ratio': sample(keep['contraction_ratio']),
        'volume_ratio': sample(keep['volume_ratio']),
        'prior_quiet_ratio': sample(f['v'].shift(1).rolling(5,min_periods=5).mean()
                                   /f['v'].shift(1).rolling(20,min_periods=20).mean().replace(0,np.nan)),
        'close_location': sample((f['close']-f['low'])/(f['high']-f['low']).where(f['high'].gt(f['low']))),
    }
    c = f['c']
    complete21 = c.rolling(21,min_periods=21).count().eq(21)
    ret20 = (c/c.shift(20)-1).where(complete21)
    numeric['relative_return20'] = sample(ret20.sub(ret20['0050'],axis=0))
    market = (c['0050']/keep['ma60']['0050']-1).to_numpy()
    numeric['market_distance_ma60'] = market[rows]
    stock_ids = [sid for sid in ids if not sid.startswith('0')]
    eligible = f['eligible'][stock_ids].eq(True).fillna(False)
    valid60 = c[stock_ids].notna() & keep['ma60'][stock_ids].notna()
    denominator = valid60.sum(axis=1)
    coverage = denominator/eligible.sum(axis=1).replace(0,np.nan)
    breadth = (c[stock_ids].gt(keep['ma60'][stock_ids]) & valid60).sum(axis=1)/denominator.replace(0,np.nan)
    breadth = breadth.where(denominator.ge(500) & coverage.ge(.8))
    numeric['market_breadth_value'] = breadth.to_numpy()[rows]
    numeric['market_breadth_coverage'] = coverage.to_numpy()[rows]
    for key, value in numeric.items():
        events[key] = value
    predicates = {
        'not_extended': ('distance_ma20', lambda v:v<=.10),
        'contraction': ('contraction_ratio', lambda v:v<=.75),
        'moderate_volume': ('volume_ratio', lambda v:(v>=1.5)&(v<=3)),
        'prior_quiet': ('prior_quiet_ratio', lambda v:v<=.8),
        'close_strong': ('close_location', lambda v:v>=.75),
        'rs20_positive': ('relative_return20', lambda v:v>0),
        'market_above60': ('market_distance_ma60', lambda v:v>0),
        'market_breadth': ('market_breadth_value', lambda v:v>=.5),
    }
    for key, (value, predicate) in predicates.items():
        events[key] = nullable_condition(events[value], predicate)
    states = sample(evidence['pstate'])
    events['poc_up'] = pd.array([s=='up' if s in ('up','down') else None for s in states],dtype='boolean')
    return events, f, days, ids


def attach_context(events, flow_rows, peer_rows, calendar):
    """Join frozen trailing features by coordinate; never read old outcomes."""
    result = events.copy()
    date_index = {str(d.date()):i for i,d in enumerate(pd.DatetimeIndex(calendar))}
    flow, peers = {}, {}
    for row in flow_rows:
        key = (row['signal_date'],row['stock_id'],row['lag'])
        if key in flow:
            raise ValueError('Duplicate flow feature coordinate')
        flow[key] = row
    for row in peer_rows:
        key = (row['signal_date'],row['stock_id'])
        if key in peers:
            raise ValueError('Duplicate peer feature coordinate')
        peers[key] = row
    peer_breadth, peer_turnover, peer_issues = [], [], []
    ratios = {lag:[] for lag in (1,3)}
    issues = {lag:[] for lag in (1,3)}
    for event in events.itertuples(index=False):
        key = (event.signal_date,event.stock_id)
        row = peers.get(key)
        issue = 'coordinate_not_covered' if row is None else row.get('feature_issue')
        if row is not None:
            if not row.get('group_cutoff_date') or row['group_cutoff_date']>=event.signal_date:
                raise ValueError('Peer membership reaches the signal date or future')
            if row.get('historical_industry_claimed') is not False:
                raise ValueError('Unexpected historical industry claim')
            if not issue and row.get('feature_available_at') != event.signal_date+' after completed close':
                raise ValueError('Known peer feature availability must be the signal close')
        values = [np.nan,np.nan] if issue else [row['peer_above_ma20_fraction'],row['peer_share_multiple']]
        if not issue and not np.isfinite(np.asarray(values,dtype=float)).all():
            raise ValueError('Known peer features are nonfinite')
        peer_breadth.append(values[0]);peer_turnover.append(values[1]);peer_issues.append(issue)
        for lag in (1,3):
            row = flow.get((*key,lag))
            issue = 'coordinate_not_covered' if row is None else row.get('flow_issue')
            if row is not None and (not issue or row.get('flow_end') is not None):
                i = date_index[event.signal_date]
                expected = str(pd.Timestamp(calendar[i-lag]).date()) if i>=lag else None
                if expected is None or row.get('flow_end')!=expected:
                    raise ValueError('Flow feature cutoff differs from declared lag')
                if not issue:
                    first = str(pd.Timestamp(calendar[i-lag-5]).date()) if i>=lag+5 else None
                    if first is None or row.get('price_start')!=first:
                        raise ValueError('Known flow feature start differs from declared five-session window')
            value = np.nan if issue else row['flow_ratio5']
            if not issue and not np.isfinite(value):
                raise ValueError('Known flow feature is nonfinite')
            ratios[lag].append(value);issues[lag].append(issue)
    result['peer_breadth_value'] = peer_breadth
    result['peer_turnover_multiple'] = peer_turnover
    result['peer_issue'] = peer_issues
    result['peer_breadth'] = nullable_condition(peer_breadth,lambda v:v>=.5)
    result['peer_turnover'] = nullable_condition(peer_turnover,lambda v:v>=1.2)
    for lag in (1,3):
        result[f'flow_ratio5_lag{lag}'] = ratios[lag]
        result[f'flow_issue_lag{lag}'] = issues[lag]
        result[f'flow_positive_lag{lag}'] = nullable_condition(ratios[lag],lambda v:v>0)
    result['clean_price'] = conjunction(result.not_extended,result.close_strong)
    result['quiet_breakout'] = conjunction(result.contraction,result.moderate_volume)
    result['peer_and_flow'] = conjunction(result.peer_breadth,result.peer_turnover,result.flow_positive_lag1)
    return result
