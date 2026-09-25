"""Independent official daily aggregates, never a certification of tick completeness.

FinMind ticks include the 14:30 fixed-price session. Only positive-volume prints
inside the regular session (including delayed close) enter this comparison.
A quote-message row count is not the exchange's executed-transaction count.
"""
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
import json
import re

import pandas as pd

from skills.intraday_limit_replay import normalize_ticks


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def number(value, *, cents=False, missing=False):
    text = str(value).strip().replace(',', '')
    if missing and text in ('', '-', '--', '---', '----'):
        return None
    if isinstance(value, bool):
        raise ValueError('Boolean official number')
    try:
        n = Decimal(text) * (100 if cents else 1)
    except InvalidOperation as exc:
        raise ValueError('Invalid official number') from exc
    if not n.is_finite() or n < 0 or n != n.to_integral_value():
        raise ValueError('Nonnegative exact integer required')
    return int(n)


def _rows(table):
    fields = [s.strip() for s in table['fields']]
    if len(fields) != len(set(fields)):
        raise ValueError('Duplicate official field')
    if any(not isinstance(r, list) or len(r) != len(fields) for r in table['data']):
        raise ValueError('Official row width mismatch')
    for key in ('totalCount', 'total'):
        if key in table and number(table[key]) != len(table['data']):
            raise ValueError('Official row count mismatch')
    return [dict(zip(fields, r)) for r in table['data']]


def _payload(payload, day):
    date.fromisoformat(day)
    if payload.get('date') != day.replace('-', '') or str(payload.get('stat')).lower() != 'ok':
        raise ValueError('Official response date/status mismatch')


def _indexed(rows, id_field, *, repeated=False):
    result = {}
    for row in rows:
        sid = str(row[id_field]).strip()
        if not re.fullmatch(r'\d{4}', sid):
            continue
        if not repeated and sid in result:
            raise ValueError('Duplicate official stock row')
        if repeated:
            result.setdefault(sid, []).append(row)
        else:
            result[sid] = row
    return result


def parse_tpex(payload, day):
    _payload(payload, day)
    tables = payload.get('tables', [])
    if len(tables) != 1:
        raise ValueError('Ambiguous TPEx ordinary table')
    table = tables[0]
    d = date.fromisoformat(day)
    if (table.get('title') != '上櫃股票每日收盤行情(不含定價)'
            or table.get('date') != f'{d.year-1911}/{d.month:02d}/{d.day:02d}'):
        raise ValueError('TPEx regular-session title/date mismatch')
    output = {}
    for sid, r in _indexed(_rows(table), '代號').items():
        output[sid] = dict(shares=number(r['成交股數']), amount_cents=number(r['成交金額(元)'], cents=True),
            transaction_count=number(r['成交筆數']),
            open_cents=number(r['開盤'], cents=True, missing=True), high_cents=number(r['最高'], cents=True, missing=True),
            low_cents=number(r['最低'], cents=True, missing=True), close_cents=number(r['收盤'], cents=True, missing=True),
            scope='tpex_regular_excludes_fixed_price', components=['tpex_no1430'])
    return output


def parse_twse(parts, day):
    """No zero inference for a missing component; basket details are unsupported."""
    required = {'total', 'intraday_odd', 'after_odd', 'fixed', 'block_single', 'block_basket'}
    if set(parts) != required:
        raise ValueError('Every TWSE trading-session component is required')
    for p in parts.values():
        _payload(p, day)
    total = parts['total']
    tables = [t for t in total['tables'] if '證券代號' in t.get('fields', [])]
    if len(tables) != 1 or not any('含一般、零股、盤後定價、鉅額交易' in s for s in tables[0].get('notes', [])):
        raise ValueError('TWSE total-session scope not declared')
    expected_titles = {'intraday_odd': '盤中零股交易行情單', 'after_odd': '盤後零股交易行情單',
        'fixed': '盤後定價交易', 'block_single': '鉅額交易日成交資訊-單一證券', 'block_basket': '鉅額交易日成交資訊-股票組合'}
    for kind, title in expected_titles.items():
        if not parts[kind].get('title', '').endswith(title):
            raise ValueError('TWSE component title mismatch')
    if any(parts[k].get('type') != 'ALL' for k in ('intraday_odd', 'after_odd')):
        raise ValueError('Complete ALL odd-lot components required before zero inference')
    if parts['fixed'].get('selectType') != 'ALL' or '千股' not in ''.join(parts['fixed'].get('notes', [])):
        raise ValueError('Complete fixed-price table and explicit thousand-share unit required')
    if parts['block_single'].get('selectType') != 'S' or parts['block_basket'].get('selectType') != 'M':
        raise ValueError('TWSE block scope mismatch')
    if _rows(parts['block_basket']):
        raise ValueError('Basket-block constituent details missing; cannot infer zero')
    indexes = {k: _indexed(_rows(parts[k]), '證券代號', repeated=k == 'block_single')
               for k in ('intraday_odd', 'after_odd', 'fixed', 'block_single')}
    output = {}
    for sid, r in _indexed(_rows(tables[0]), '證券代號').items():
        shares, amount, count = number(r['成交股數']), number(r['成交金額'], cents=True), number(r['成交筆數'])
        component_values = {}
        count_available = True
        for kind, index in indexes.items():
            records = index.get(sid, []) if kind == 'block_single' else ([index[sid]] if sid in index else [])
            qty = sum(number(z['成交數量' if kind == 'fixed' else '成交股數']) * (1000 if kind == 'fixed' else 1) for z in records)
            amt = sum(number(z['成交金額'], cents=True) for z in records)
            if kind == 'block_single' and records:
                count_available = False
            elif kind != 'block_single':
                count -= sum(number(z['成交筆數']) for z in records)
            shares -= qty
            amount -= amt
            component_values[kind] = dict(shares=qty, amount_cents=amt)
        if min(shares, amount, count) < 0:
            raise ValueError('Negative ordinary-session residual')
        output[sid] = dict(shares=shares, amount_cents=amount,
            transaction_count=count if count_available else None,
            open_cents=number(r['開盤價'], cents=True, missing=True), high_cents=number(r['最高價'], cents=True, missing=True),
            low_cents=number(r['最低價'], cents=True, missing=True), close_cents=number(r['收盤價'], cents=True, missing=True),
            scope='twse_total_minus_all_other_sessions', components=component_values)
    return output


def summarize_ticks(raw, stock_id, day, market):
    ticks = normalize_ticks(raw, stock_id, day, market)
    regular = ticks.time.ge(pd.Timedelta('09:00:00')) & ticks.time.lt(pd.Timedelta('13:34:00'))
    fixed = ticks.time.eq(pd.Timedelta('14:30:00'))
    positive = ticks.shares.gt(0)
    unknown = positive & ~(regular | fixed)
    selected = ticks.loc[regular & positive]
    prices = [number(v, cents=True) for v in selected.price]
    shares = [int(v) for v in selected.shares]
    return dict(shares=sum(shares), amount_cents=sum(p*q for p,q in zip(prices, shares)),
        open_cents=prices[0] if prices else None, high_cents=max(prices) if prices else None,
        low_cents=min(prices) if prices else None, close_cents=prices[-1] if prices else None,
        regular_message_rows=len(selected), raw_rows=len(raw), normalized_rows=len(ticks),
        normalization_dropped_rows=len(raw)-len(ticks), raw_all_session_shares=int(pd.to_numeric(raw.volume).sum())*1000,
        zero_volume_rows=int((~positive).sum()),
        fixed_price_rows=int((fixed & positive).sum()), fixed_price_shares=int(ticks.loc[fixed & positive, 'shares'].sum()),
        unknown_session_rows=int(unknown.sum()),
        first_regular_time=str(selected.time.iloc[0]) if len(selected) else None,
        last_regular_time=str(selected.time.iloc[-1]) if len(selected) else None)


def reconcile(tape, official):
    keys = ('shares', 'amount_cents', 'open_cents', 'high_cents', 'low_cents', 'close_cents')
    differences = {k: dict(tape=tape[k], official=official[k]) for k in keys if tape[k] != official[k]}
    if tape['unknown_session_rows']:
        differences['unknown_session_rows'] = tape['unknown_session_rows']
    matched = not differences
    return dict(status='daily_aggregates_matched_count_unverified' if matched else 'daily_aggregate_conflict',
        same_scope_aggregate_matched=matched, differences=differences, tape=tape, official=official,
        transaction_count_comparison='not_comparable_message_rows_vs_execution_count',
        transaction_count_verified=False, tick_sequence_complete=False,
        quarantined=not matched, quarantine_reason='official_daily_aggregate_conflict' if not matched else None,
        accepted_for_strict_replay=False, own_order_fill_proven=False, live_qualified=False)


def verify_report(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path.with_suffix('.sha256').read_text().strip() != digest(path):
        raise ValueError('Reconciliation report hash mismatch')
    report = json.loads(path.read_text())
    if report.get('schema') != 'board_tape_reconciliation_v1':
        raise ValueError('Unsupported reconciliation schema')
    for mapping in ('input_sha256', 'code_sha256'):
        for name, expected in report[mapping].items():
            p = (root/name).resolve()
            if not p.is_relative_to(root) or digest(p) != expected:
                raise ValueError('Reconciliation input/code mismatch: '+name)
    if any(report.get(k) is not False for k in ('strict_data_ready', 'live_qualified', 'own_order_fill_proven')):
        raise ValueError('Daily totals cannot certify executable fills')
    return report


def assert_not_quarantined(report, day, stock_id):
    """Additional negative gate only; passing never grants replay qualification."""
    rows = [r for r in report['rows'] if (r['date'],r['stock_id']) == (day,stock_id)]
    if len(rows) != 1 or not rows[0]['same_scope_aggregate_matched']:
        raise ValueError('Independent daily aggregate evidence missing or conflicting')
    return None


def load_reconciled_tape(report_path, root, day, stock_id):
    """New research-only consumer; sealed replay engines are not modified.

    Independent daily mismatch or missing evidence is a hard refusal. A matched
    daily summary cannot certify the sequence, queue position or actual filling.
    """
    root = Path(root).resolve()
    report = verify_report(report_path,root)
    assert_not_quarantined(report,day,stock_id)
    item = next(r for r in report['rows'] if (r['date'],r['stock_id'])==(day,stock_id))
    path = (root/item['tape_path']).resolve()
    if not path.is_relative_to(root) or digest(path) != item['tape_sha256']:
        raise ValueError('Reconciled tape source hash mismatch')
    frame = normalize_ticks(pd.read_parquet(path),stock_id,day,item['market'])
    frame = frame.loc[frame.time.ge(pd.Timedelta('09:00:00'))
                      & frame.time.lt(pd.Timedelta('13:34:00')) & frame.shares.gt(0)].copy()
    return dict(frame=frame,stock_id=stock_id,date=day,market=item['market'],
                use='research_only_daily_aggregate_checked',tick_sequence_complete=False,
                accepted_for_strict_replay=False,own_order_fill_proven=False,live_qualified=False)
