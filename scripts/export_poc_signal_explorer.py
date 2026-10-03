#!/usr/bin/env python3
"""Publish sealed account evidence as an offline POC signal explorer.

No replay, data acquisition, or trade execution occurs here. Candidate evidence
and account observations have separate dates so a UI can hide future execution.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.export_signal_explorer import (PRICE_COLUMNS, digest, number,
    pack_prices, json_for_script, render_html)

BASE = Path('.cache/red-volume-exit-20261003')
BUNDLE = Path('.cache/market-input-repair-20261002/inputs-v2')
EXPECTED_MANIFEST = 'f62dbe32d263abb464c9f283879b7d6340b0667b6fa9e99404c6ece7e04663d6'
REPORTS = {
    'anchors-v2': '4d670409c617d2d562f60630dcafaef40c775598878a2b920415ba10c0a2aef2',
    'full-v1': '2b33236460533746c4fab4f2e65a9dd3df04efb2ab0bd5031b61a1c4d8e37a55',
}
CASES = {'poc_red': 'full-v1', 'poc_base': 'anchors-v2', 'original': 'anchors-v2'}
LABELS = {'poc_red': 'POC 優先＋訊號紅K', 'poc_base': 'POC 優先', 'original': '原始相對強勢排序'}
END = '2026-09-09'


def read(path):
    return json.loads(Path(path).read_text())


def merge_refs(refs, additions):
    for path, expected in additions.items():
        if path in refs and refs[path] != expected:
            raise ValueError('Conflicting source hashes: ' + path)
        refs[path] = expected


def verify_refs(refs, root=ROOT):
    """Every parent reference is checked, including raw receipts and snapshots."""
    def verify(item):
        name, expected = item
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or digest(path) != expected:
            raise ValueError('Missing or changed sealed source: ' + name)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(verify, refs.items()))


def load_evidence(root=ROOT):
    refs, reports, cases, profiles = {}, {}, {}, {}
    for folder, expected in REPORTS.items():
        path = BASE / folder / 'report.json'
        if digest(root / path) != expected:
            raise ValueError('Parent report differs from sealed study: ' + str(path))
        report = read(root / path)
        if report['input_bundle'] != str(BUNDLE) or report['end'] != END or report['live_qualified'] is not False:
            raise ValueError('Unexpected account input scope or qualification')
        merge_refs(refs, {str(path): expected})
        merge_refs(refs, report['source_sha256'])
        for source, snapshot in report['source_snapshots'].items():
            if report['source_sha256'].get(source) != snapshot['sha256']:
                raise ValueError('Unbound source snapshot: ' + source)
            merge_refs(refs, {snapshot['path']: snapshot['sha256']})
        profile_path = str(BASE / folder / 'profile-data/profile-features.json')
        if profile_path not in refs:
            raise ValueError('Profile features absent from parent closure')
        reports[folder] = report
    manifest_path = BUNDLE / 'manifest.json'
    if digest(root / manifest_path) != EXPECTED_MANIFEST:
        raise ValueError('Input manifest is not the sealed account bundle')
    manifest = read(root / manifest_path)
    merge_refs(refs, {str(manifest_path): EXPECTED_MANIFEST})
    merge_refs(refs, {str(BUNDLE / p): h for p, h in manifest['files_sha256'].items()})
    for arm, folder in CASES.items():
        item = reports[folder]['cases'][arm]
        if item['completed'] is not True or item['path'] != str(BASE / folder / (arm + '.json')):
            raise ValueError('Missing completed sealed case: ' + arm)
        merge_refs(refs, {item['path']: item['sha256']})
    verify_refs(refs, root)
    for arm, folder in CASES.items():
        case = read(root / reports[folder]['cases'][arm]['path'])
        if not case['completed'] or case['summary'] != reports[folder]['cases'][arm]['summary']:
            raise ValueError('Case differs from its completed parent summary')
        cases[arm] = case
        profiles[arm] = read(root / BASE / folder / 'profile-data/profile-features.json')
    return cases, profiles, refs


def unique_index(rows, key, label):
    result = {}
    for row in rows:
        value = row[key]
        if value in result:
            raise ValueError('Duplicate ' + label + ': ' + str(value))
        result[value] = row
    return result


def build_signal_rows(entries, adjusted, quotes, companies, year=2026, end=END):
    calendar = [str(d.date()) for d in adjusted.index]
    positions = {d: i for i, d in enumerate(calendar)}
    selected = [r for r in entries if f'{year}-01-01' <= r['signal_date'] <= end]
    unique_index(selected, 'event_id', 'candidate')
    grouped = defaultdict(list)
    for row in selected:
        if len(row['members']) != 1 or not math.isfinite(row['priority']):
            raise ValueError('Require one stock and finite original priority')
        grouped[row['signal_date']].append(row)
    raw = quotes.set_index(['date', 'stock_id'])
    if not raw.index.is_unique:
        raise ValueError('Duplicate raw quote identity')
    names = unique_index(companies.to_dict('records'), 'stock_id', 'company')
    prior60 = adjusted.shift(1).rolling(60, min_periods=60).max()
    result = []
    for day, group in sorted(grouped.items()):
        i = positions[day]
        expected_entry = calendar[i + 1] if i + 1 < len(calendar) else None
        for rank, row in enumerate(sorted(group, key=lambda e: (-e['priority'], e['event_id'])), 1):
            if row['entry_date'] != expected_entry or expected_entry is None:
                raise ValueError('Sealed entry must be the next observed market session')
            sid, stamp = row['members'][0], pd.Timestamp(day)
            q = raw.loc[(stamp, sid)]
            opened, close = number(q['open']), number(q['close'])
            if opened is None or close is None or min(opened, close) <= 0:
                raise ValueError('Invalid signal OHLC: ' + row['event_id'])
            evidence = row['leader_evidence']
            own, benchmark = evidence['leader_return20'], evidence['benchmark_return20']
            if not math.isclose(own - benchmark, row['priority'], rel_tol=0, abs_tol=1e-12):
                raise ValueError('Priority differs from frozen signal evidence')
            adjusted_close = number(adjusted.at[stamp, sid])
            prior = number(prior60.at[stamp, sid])
            company = names[sid]
            result.append(dict(signal_id=row['event_id'], stock_id=sid, name=company['name'],
                market=company['market'], signal_date=day, entry_date=expected_entry,
                daily_rank=rank, candidate_count=len(group), priority=row['priority'],
                stock_return20=own, benchmark_return20=benchmark,
                volume_ratio=evidence['leader_volume_ratio'],
                signal_open_raw=opened, signal_close_raw=close,
                signal_close_adjusted=adjusted_close, previous60_high_adjusted=prior,
                close_to_prior60_high_pct=adjusted_close / prior - 1 if adjusted_close and prior else None,
                turnover_mean20=None, turnover_median20=None, trend_state=None,
                candle='red' if close > opened else 'black' if close < opened else 'doji',
                available_at=day + ' 收盤資料完成後', source_scope='frozen_repaired_history'))
    return result


def profile_details(profile, signal):
    """Only a case's recorded query authorizes joining a precomputed profile."""
    for key, expected in [('event_id', signal['signal_id']), ('stock_id', signal['stock_id']),
                          ('signal_date', signal['signal_date'])]:
        if profile.get(key) != expected:
            raise ValueError('Profile identity mismatch: ' + key)
    if profile.get('available') is not True:
        if not profile.get('reason') or profile.get('poc_up') is not None:
            raise ValueError('Unknown profile must retain its reason and no flag')
        return dict(poc_status='unknown', poc_reason=profile['reason'])
    if type(profile.get('poc_up')) is not bool:
        raise ValueError('Known profile needs a boolean flag')
    dates = profile['prior_dates']
    if len(dates) != 20 or dates != sorted(set(dates)) or any(d >= signal['signal_date'] for d in dates):
        raise ValueError('Profile window must contain 20 strictly pre-signal sessions')
    p = profile['profile']
    if p['session_dates'] != dates or p['poc_up'] != profile['poc_up']:
        raise ValueError('Profile window or direction mismatch')
    return dict(poc_status='up' if profile['poc_up'] else 'down', poc_reason=None,
        poc_before=p['first_half']['poc_price'], poc_after=p['second_half']['poc_price'],
        poc_full=p['full']['poc_price'], poc_val=p['full']['val'], poc_vah=p['full']['vah'],
        poc_window_start=dates[0], poc_window_end=dates[-1], poc_price_basis='raw',
        poc_ordinary_daily_matched=profile['ordinary_daily_matched'])


def build_decisions(signals, case, profiles):
    gates = unique_index(case['entry_gate_decisions'], 'event_id', 'red gate')
    queries = unique_index(case['profile_queries'], 'event_id', 'profile query')
    profile_index = unique_index(profiles, 'event_id', 'profile feature')
    contexts = unique_index(case['account']['selection_decisions'], 'date', 'selection day')
    cert_rows = {}
    for day, context in contexts.items():
        for row in context.get('certificate', {}).get('decisions', []):
            if row['event_id'] in cert_rows:
                raise ValueError('Repeated selection certificate event')
            cert_rows[row['event_id']] = row
    orders, trades = defaultdict(list), defaultdict(list)
    for row in case['account']['orders']:
        if row['side'] == 'buy':
            orders[row['event_id']].append(row)
    for row in case['account']['trades']:
        if row['side'] == 'buy':
            trades[row['event_id']].append(row)
    red_required = case['family_rules']['red_gate']
    result = {}
    for s in signals:
        eid, day = s['signal_id'], s['entry_date']
        gate = gates.get(eid)
        if red_required:
            if gate is None or gate['signal_date'] != s['signal_date'] or gate['entry_date'] != day:
                raise ValueError('Missing dated red gate')
            if gate['passed'] != (s['candle'] == 'red') or gate['status'] != s['candle']:
                raise ValueError('Red gate disagrees with signal-day raw candle')
        rejected = red_required and not gate['passed']
        context, cert = contexts.get(day), cert_rows.get(eid, {})
        if not rejected and (context is None or eid not in context['original_event_ids']):
            raise ValueError('Kept candidate absent from account selection context')
        if rejected and eid in queries:
            raise ValueError('Rejected red candidate was queried for POC')
        fallback = bool(context and context.get('fallback_to_original'))
        reason = ('red_gate_rejected' if rejected else
                  'strategy_does_not_use_poc' if case['family_rules'].get('anchor') == 'original' else
                  cert.get('reason') or cert.get('profile_status') or 'not_queried')
        if fallback and eid not in queries and not rejected:
            reason = 'whole_day_quality_fallback_before_query'
        selection = ('red_gate_rejected' if rejected else 'fallback_original' if fallback else
                     cert.get('selection_status', 'original_unchanged'))
        row = dict(signal_id=eid, red_gate_required=red_required,
            red_gate=gate['passed'] if red_required else None,
            red_gate_status=gate['status'] if red_required else 'not_required',
            poc_status='not_evaluated', poc_reason=reason,
            poc_reconstructed=True, selection_status=selection,
            selection_reason=cert.get('reason'),
            candidate_rank_after_red=cert.get('original_rank'),
            fallback=fallback, fallback_reason=context.get('fallback_reason') if context else None,
            entry_date=day, selection_available_at=day + ' 盤前帳戶資源預約',
            selected_for_planning=bool(context and eid in context['selected_event_ids']) and not rejected,
            simulated_buy_qty=sum(t['qty'] for t in trades[eid]),
            simulated_buy_dates=sorted({t['date'] for t in trades[eid]}),
            order_count=len(orders[eid]),
            order_failures=sorted({o['failure'] for o in orders[eid] if o.get('failure')}))
        if eid in queries:
            query = queries[eid]
            if query['stock_id'] != s['stock_id'] or query['signal_date'] != s['signal_date']:
                raise ValueError('Query identity differs from candidate')
            if eid not in profile_index:
                raise ValueError('Queried profile is missing')
            row.update(profile_details(profile_index[eid], s))
            expected = {'up': 'known_true', 'down': 'known_false', 'unknown': 'unknown'}[row['poc_status']]
            if cert.get('profile_status') != expected:
                raise ValueError('Query/profile differs from selection certificate')
        elif cert.get('profile_status') in ('known_true', 'known_false', 'unknown'):
            raise ValueError('Evaluated certificate lacks its query record')
        result[eid] = row
    return result


def account_view(case, start, end, adjusted, raw_close):
    account = case['account']
    holdings = defaultdict(list)
    for row in account['holdings']:
        holdings[row['date']].append(deepcopy(row))
    dates = unique_index(account['daily'], 'date', 'account day')
    previous = max((d for d in dates if d < start), default=None)
    visible = {d: dict(deepcopy(row), holdings=holdings[d], holding_count=row['holdings'])
               for d, row in dates.items() if start <= d <= end}
    for d, row in visible.items():
        if row['holding_count'] != len(row['holdings']):
            raise ValueError('Daily holding count differs from ledger')
    trades = []
    for i, original in enumerate(account['trades']):
        if not start <= original['date'] <= end:
            continue
        row = deepcopy(original)
        stamp, sid = pd.Timestamp(row['date']), row['stock_id']
        a, raw = number(adjusted.at[stamp, sid]), number(raw_close.at[stamp, sid])
        if a is None or raw is None or min(a, raw) <= 0:
            raise ValueError('Cannot place modeled fill on missing candle basis')
        row.update(trade_id=str(i), adjusted_marker_price=row['reference_price'] * a / raw,
            price_basis='raw_execution_reference', execution_kind='research_simulated_fill',
            available_at=row['date'] + ' 收盤資料完成後')
        trades.append(row)
    return dict(account_days=visible, trades=trades,
        opening_inventory=holdings[previous] if previous else [],
        opening_account_day=deepcopy(dates[previous]) if previous else None,
        opening_inventory_date=previous)


def build_payload(entries, cases, profiles, adjusted, quality, quotes, eligibility, companies,
                  *, year=2026, end=END, warmup_sessions=120):
    calendar = pd.DatetimeIndex(adjusted.index)
    if not calendar.is_unique or not calendar.is_monotonic_increasing or str(calendar[-1].date()) != end:
        raise ValueError('Require exact ordered sealed market calendar')
    if not quality.index.equals(calendar) or not eligibility.index.equals(calendar):
        raise ValueError('Price quality/eligibility calendars differ')
    signals = build_signal_rows(entries, adjusted, quotes, companies, year, end)
    market = calendar[(calendar >= f'{year}-01-01') & (calendar <= end)]
    if market.empty:
        raise ValueError('No market sessions')
    start = str(market[0].date())
    first = calendar.get_loc(market[0])
    price_days = calendar[max(0, first - warmup_sessions):]
    raw_close = quotes.pivot(index='date', columns='stock_id', values='close')
    strategies = {}
    for arm, case in cases.items():
        view = account_view(case, start, end, adjusted, raw_close)
        if set(view['account_days']) != {str(d.date()) for d in market}:
            raise ValueError('Account view lacks a market session')
        strategies[arm] = dict(label=LABELS[arm], rules=deepcopy(case['family_rules']),
            summary=deepcopy(case['summary']), summary_scope='full_account_from_2024',
            year_summary=next(r for r in case['summary']['annual'] if r['year'] == str(year)),
            decisions=build_decisions(signals, case, profiles[arm]), **view)
    by_day, by_stock = {str(d.date()): [] for d in market}, defaultdict(list)
    for signal in signals:
        by_day[signal['signal_date']].append(signal['signal_id'])
        by_stock[signal['stock_id']].append(signal)
    stock_ids = set(by_stock)
    for strategy in strategies.values():
        stock_ids.update(t['stock_id'] for t in strategy['trades'])
        stock_ids.update(h['stock_id'] for h in strategy['opening_inventory'])
        stock_ids.update(h['stock_id'] for d in strategy['account_days'].values() for h in d['holdings'])
    names = unique_index(companies.to_dict('records'), 'stock_id', 'company')
    quote_groups = {sid: g for sid, g in quotes[quotes.stock_id.isin(stock_ids)].groupby('stock_id', sort=False)}
    stocks, invalid = {}, Counter()
    for sid in sorted(stock_ids):
        q = quote_groups.get(sid, pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume']))
        prices, issues = pack_prices(price_days, q, adjusted[sid], quality[sid], eligibility[sid])
        indexed = {r[0]: r for r in prices}
        for s in by_stock[sid]:
            if indexed[s['signal_date']][7] != s['signal_close_adjusted']:
                raise ValueError('Signal and candle marker bases differ')
        company = names[sid]
        stocks[sid] = dict(stock_id=sid, name=company['name'], market=company['market'],
            prices=prices, signal_ids=[s['signal_id'] for s in by_stock[sid]],
            invalid_bar_count=sum(issues.values()), quality_issue_counts=issues)
        invalid.update(issues)
    days = [dict(date=d, signal_count=len(ids), signal_ids=ids,
                 signal_status='not_generated_no_next_session' if d == end else 'complete')
            for d, ids in by_day.items()]
    if days[-1]['signal_ids']:
        raise ValueError('Unexpected terminal-day entry candidate')
    return dict(schema='offline_poc_account_signal_explorer_v1', default_strategy='poc_red',
        metadata=dict(year=year, date_start=start, date_end=end, data_as_of=end,
            account_start='2024-01-02', signal_last_date=max(by_day.keys() - {end}),
            price_start=str(price_days[0].date()), warmup_sessions=min(first, warmup_sessions),
            signal_count=len(signals), stock_count=len(stocks), market_day_count=len(days),
            zero_signal_days=sum(d['signal_count'] == 0 and d['signal_status'] == 'complete' for d in days),
            source_label='修復封存行情與 POC 帳戶研究；資料截止 2026/09/09',
            source_scope_labels={'frozen_repaired_history': '修復後封存歷史；非完整市场與成交認證'},
            price_basis='adjusted_ohlc_same_day_close_factor', poc_price_basis='raw',
            strategy_label='原始突破放量候選＋POC 優先＋可切換紅K門檻',
            score_definition='個股近20交易日報酬 − 0050同期報酬；不是獲利機率',
            marker_definition='訊號與帳戶模擬成交分開；所有成交都不是券商實際成交',
            invalid_bar_count=sum(invalid.values()), quality_issue_counts=dict(invalid),
            limitations=['帳戶自2024/01/02以100萬元開始，2026視圖延續既有持股和資產。',
                '訊號最後至9/8；9/9無下一執行日，封存資料未產生當日候選，不代表確認無訊號。',
                'POC僅顯示該版本真正查詢的20個前置交易日資料；未查不等於POC下降。',
                '必要POC遇永久品質未知時，當日全部合格候選回原排序；這是資料可用時的POC策略。',
                'POC數字是原始股價；K線是還原OHLC，兩者不可直接當成同一價格座標。',
                '一般盤使用日高低中點成交研究假設；普通盤可成交量未完整認證，非實盤成交。',
                '全期摘要屬2024至2026/09/09；當時資料模式應隱藏未来帳戶/成交與隔日預約。',
                '候選名單不是全市場歷史身分認證；名稱沿用封存名稱。成交值欄位未附於訊號檔，留空。'],
            live_qualified=False, cash_account=True, outcomes_included=True,
            new_backtest_executed=False), price_columns=PRICE_COLUMNS, days=days,
        signals=signals, stocks=stocks, strategies=strategies)


def run(args):
    cases, profiles, refs = load_evidence()
    bundle = ROOT / BUNDLE
    payload = build_payload(read(bundle / 'signals.json')['entries']['median50m'], cases, profiles,
        pd.read_parquet(bundle / 'close-official.parquet').set_index('date'),
        pd.read_parquet(bundle / 'close-quality.parquet').set_index('date'),
        pd.read_parquet(bundle / 'quotes-unmasked.parquet'),
        pd.read_parquet(bundle / 'eligibility.parquet').set_index('date'),
        pd.read_parquet(bundle / 'companies.parquet'))
    payload['metadata']['parent_report_sha256'] = {str(BASE / k / 'report.json'): v for k, v in REPORTS.items()}
    payload['metadata']['verified_source_count'] = len(refs)
    payload_path = Path(args.payload).resolve()
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json_for_script(payload), encoding='utf-8')
    outputs = {str(payload_path.relative_to(ROOT)): digest(payload_path)}
    for name in ('scripts/export_poc_signal_explorer.py', 'tests/test_poc_signal_explorer_data.py',
                 'scripts/export_signal_explorer.py'):
        refs[name] = digest(ROOT / name)
    if args.output:
        template = Path(args.template).resolve()
        refs[str(template.relative_to(ROOT))] = digest(template)
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_html(template.read_text(), payload), encoding='utf-8')
        outputs[str(output.relative_to(ROOT))] = digest(output)
    receipt = dict(schema='offline_poc_explorer_export_receipt_v1',
        created_at=datetime.now(timezone.utc).isoformat(), source_sha256=refs,
        output_sha256=outputs, parent_report_sha256=payload['metadata']['parent_report_sha256'],
        metadata=payload['metadata'], no_network=True, no_strategy_execution=True)
    path = Path(args.receipt).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(payload=str(payload_path), output=args.output, receipt=str(path),
                         signal_count=len(payload['signals']), sources=len(refs), hashes=outputs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Optional single-file HTML destination; omit for payload-only export')
    parser.add_argument('--template', default='ui/poc_signal_explorer.html')
    parser.add_argument('--payload', default='.cache/poc-explorer-20261003/payload.json')
    parser.add_argument('--receipt', default='.cache/poc-explorer-20261003/receipt.json')
    run(parser.parse_args())
