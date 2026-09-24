"""Versioned financial ratios; current observations never backdate publication.

FinMind amounts are TWD, income statements are single quarter, cash flow is
year-to-date. ROE uses consolidated profit and average consolidated equity.
OrdinaryShare is capital, not share count. No denominator is inferred from it.
"""
from datetime import timedelta
import hashlib
import json
import math

import numpy as np
import pandas as pd
from app.finmind import FinMindError

VERSION = 'quarterly_v2'
BS_MAP = {'TotalAssets':'assets', 'Liabilities':'liabilities', 'Equity':'equity'}
IS_MAP = {'Revenue':'revenue', 'OperatingIncome':'operating_income', 'IncomeAfterTaxes':'net_income'}
CF_MAP = {'CashFlowsFromOperatingActivities':'cfo_ytd', 'PropertyAndPlantAndEquipment':'capex_ytd'}
METRICS = ('roe_ttm','roa_ttm','debt_ratio','operating_margin','net_margin','fcf_ttm','fcf_per_share')


def pivot(raw, mapping):
    if raw.empty:
        return pd.DataFrame(columns=['stock_id','report_date',*mapping.values()])
    if not {'date','stock_id','type','value'}.issubset(raw):
        raise FinMindError('Quarterly source schema missing date/stock_id/type/value')
    x = raw.copy()
    x['stock_id'] = x.stock_id.astype(str)
    x = x[x.stock_id.str.fullmatch(r'\d{4}') & x.type.isin(mapping)].copy()
    if x.empty:
        raise FinMindError('No supported financial fields for ordinary stock IDs')
    x['report_date'] = pd.to_datetime(x.date, errors='raise')
    if not x.report_date.dt.is_quarter_end.all():
        raise FinMindError('Financial report date is not a calendar quarter end')
    x['value'] = pd.to_numeric(x.value, errors='raise')
    if not np.isfinite(x.value).all():
        raise FinMindError('Nonfinite financial amount')
    keys = ['stock_id','report_date','type']
    if x.groupby(keys).value.nunique().gt(1).any():
        raise FinMindError('Conflicting financial versions in one response')
    x = x.drop_duplicates(keys)
    return x.pivot(index=['stock_id','report_date'], columns='type', values='value').rename(columns=mapping).reset_index()


def calculate(bs, income, cashflow, *, observed_at, shares=None):
    observed = pd.Timestamp(observed_at)
    if observed.tzinfo is None:
        raise ValueError('Financial observation requires an explicit timezone')
    local_day = observed.tz_convert('Asia/Taipei').date()
    frames = [pivot(bs, BS_MAP), pivot(income, IS_MAP), pivot(cashflow, CF_MAP)]
    x = frames[0]
    for f in frames[1:]:
        x = x.merge(f, on=['stock_id','report_date'], how='outer', validate='one_to_one')
    if x.empty:
        return []
    for col in [*BS_MAP.values(), *IS_MAP.values(), *CF_MAP.values()]:
        if col not in x:
            x[col] = np.nan
    if x.report_date.dt.date.gt(local_day).any():
        raise FinMindError('Financial period is later than observation')
    share_map = {}
    for row in (shares or []):
        key = (row['stock_id'], pd.Timestamp(row['report_date']).to_period('Q'))
        if key in share_map or not row.get('source') or not math.isfinite(row['shares']) or row['shares'] <= 0:
            raise FinMindError('Invalid or unproven outstanding share denominator')
        # These are independently verified period-end outstanding ordinary shares.
        share_map[key] = row
    records = []
    for sid, group in x.groupby('stock_id'):
        group = group.set_index(group.report_date.dt.to_period('Q')).drop(columns=['stock_id','report_date']).sort_index()
        actual = set(group.index)
        group = group.reindex(pd.period_range(group.index.min(), group.index.max(), freq='Q'))
        for col in ('cfo','capex'):
            values = group[col+'_ytd']
            group[col+'_q'] = values.diff().where(group.index.quarter != 1, values)
        if group.capex_ytd.dropna().gt(0).any():
            raise FinMindError('PPE acquisition cash flow must be an outflow; verify provider sign')
        profit = group.net_income.rolling(4, min_periods=4).sum()
        fcf = (group.cfo_q + group.capex_q).rolling(4, min_periods=4).sum()
        def ratio(a,b):
            return float(a/b*100) if pd.notna(a) and pd.notna(b) and b > 0 else None
        for period in sorted(actual):
            row = group.loc[period]
            avg_equity = (group.equity.shift(4).at[period] + row.equity)/2
            avg_assets = (group.assets.shift(4).at[period] + row.assets)/2
            share = share_map.get((sid,period))
            cash = float(fcf.at[period]) if pd.notna(fcf.at[period]) else None
            metrics = dict(roe_ttm=ratio(profit.at[period], avg_equity),
                roa_ttm=ratio(profit.at[period],avg_assets), debt_ratio=ratio(row.liabilities,row.assets),
                operating_margin=ratio(row.operating_income,row.revenue), net_margin=ratio(row.net_income,row.revenue),
                fcf_ttm=cash, fcf_per_share=cash/share['shares'] if cash is not None and share else None)
            dependencies = group.loc[period-4:period].copy()
            dependencies.index = dependencies.index.astype(str)
            evidence = dict(version=VERSION, raw=dependencies.to_json(orient='split', double_precision=15), shares=share)
            fingerprint = hashlib.sha256(json.dumps(evidence,sort_keys=True,default=str).encode()).hexdigest()
            missing = [name for name,value in metrics.items() if value is None]
            records.append(dict(stock_id=sid, report_date=period.end_time.date(),
                observed_at=observed.tz_convert('UTC').tz_localize(None).to_pydatetime(),
                available_date=local_day+timedelta(days=1), source_sha256=fingerprint,
                definition_version=VERSION, timing_basis='first_observed_next_day',
                missing_metrics=','.join(missing), **metrics))
    return records


def align_prices(prices, snapshots):
    """Keep the newest known report, not a later-arriving older report revision."""
    targets = {'roe_ttm':'roe_raw','debt_ratio':'debt_ratio_raw','operating_margin':'operating_margin_raw'}
    pieces = []
    if snapshots.empty:
        return prices.assign(**{v:np.nan for v in targets.values()})
    snapshots = snapshots.copy()
    for col in ('available_date','report_date','observed_at'):
        snapshots[col] = pd.to_datetime(snapshots[col])
    observed_days = snapshots.observed_at.dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei').dt.tz_localize(None).dt.normalize()
    if ((snapshots.available_date <= observed_days).any()
            or snapshots.definition_version.ne(VERSION).any()
            or snapshots.timing_basis.ne('first_observed_next_day').any()):
        raise FinMindError('Quarterly snapshot has unverified timing or definition')
    for sid, prices_one in prices.groupby('stock_id',sort=False):
        known, events = {}, []
        rows = snapshots[snapshots.stock_id.eq(sid)].sort_values(['available_date','observed_at','report_date'])
        for row in rows.to_dict('records'):
            known[row['report_date']] = row
            newest = known[max(known)]
            events.append(dict(available_date=row['available_date'], **{k:newest[k] for k in targets}))
        if not events:
            pieces.append(prices_one.assign(**{v:np.nan for v in targets.values()}))
            continue
        timeline = pd.DataFrame(events).drop_duplicates('available_date',keep='last')
        for col in targets:
            timeline[col] = pd.to_numeric(timeline[col],errors='raise')
        merged = pd.merge_asof(prices_one.sort_values('trading_date'), timeline,
            left_on='trading_date',right_on='available_date',direction='backward')
        pieces.append(merged.drop(columns='available_date').rename(columns=targets))
    return pd.concat(pieces,ignore_index=True)
