#!/usr/bin/env python3
"""Independently reconcile all 31 cash accounts before publishing a ranking."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from skills.all_entry_accounts import START, END, registry, write, digest
from scripts.summarize_poc_broker_account import Sources, account_metrics
from scripts.summarize_strategy_account_comparison import describe_cohorts, describe_benchmark, pending_cash_claims


class UnverifiedCertificateInventory(ValueError):
    """A full replay cannot be ranked until this instrument is represented."""


def holding_scope(account, active_slots):
    """Count actual symbols, including residual shares and delayed deliveries."""
    by_day = {}
    for row in account['holdings']:
        if row['qty'] > 0:
            by_day.setdefault(row['date'], set()).add(row['stock_id'])
    counts = [len(stocks) for stocks in by_day.values()]
    return dict(maximum_held_symbols=max(counts, default=0),
        days_above_active_slot_count=sum(n > active_slots for n in counts))


def guard_certificate_inventory(case):
    # The inherited runner has not separated these rights certificates from
    # ordinary stock inventory. A completed replay alone cannot certify them.
    boundaries = {('5876','2026-07-28'), ('6446','2026-07-09'), ('9941','2026-09-01'), ('2801','2026-08-05'), ('5904','2025-07-15')}
    for row in case['account'].get('corporate_actions',[]):
        if (row.get('stock_id'),row.get('date')) in boundaries and row.get('kind') == 'stock_dividend':
            if row['stock_id'] == '5904' and case.get('poya_certificate_inventory_audit'):
                from skills.all_entry_poya_completion import verify_inventory_proof
                try:
                    verify_inventory_proof(case)
                except ValueError as exc:
                    raise UnverifiedCertificateInventory('5904 獨立證書庫存核對失敗：'+str(exc)) from exc
                continue
            if row['stock_id'] == '5876' and case.get('certificate_inventory_audit'):
                from skills.all_entry_certificate_completion import verify_inventory_proof
                try:
                    verify_inventory_proof(case)
                except ValueError as exc:
                    raise UnverifiedCertificateInventory('5876 獨立證書庫存核對失敗：'+str(exc)) from exc
                continue
            if row['stock_id'] == '2801' and case.get('context_certificate_inventory_audit'):
                from skills.all_entry_context_distributions import verify_inventory_proof
                try:
                    verify_inventory_proof(case)
                except ValueError as exc:
                    raise UnverifiedCertificateInventory('2801 獨立證書庫存核對失敗：'+str(exc)) from exc
                continue
            if row['stock_id'] == '9941' and case.get('late_certificate_inventory_audit'):
                from skills.all_entry_late_certificate_completion import verify_inventory_proof
                try:
                    verify_inventory_proof(case)
                except ValueError as exc:
                    raise UnverifiedCertificateInventory('9941 期末證書庫存核對失敗：'+str(exc)) from exc
                continue
            raise UnverifiedCertificateInventory(
                f"{row['stock_id']} {row['date']} 配發新股權利證書：尚缺獨立庫存、"
                '轉換與可交易行情驗證，完整回放不等於有效績效，暫不排名。')


def summarize(report_path, output):
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    if digest(report_path) != report_path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Replay report checksum differs')
    definitions = registry()
    names = {x['id']: x['name'] for x in definitions} | {'benchmark':'0050 含息基準'}
    if report.get('schema') != 'all_entry_accounts_v1' or set(report['cases']) != set(names):
        raise ValueError('Publication requires an attempted result for every one of the 32 accounts')
    if (report['start'], report['end'], report['initial_cash']) != (START, END, 1_000_000):
        raise ValueError('Account comparison scope changed')
    sources = Sources(ROOT)
    sources.closure(report, str(report_path.relative_to(ROOT)))
    rows, calendar = {}, None
    for arm, name in names.items():
        descriptor = report['cases'][arm]
        case = json.loads(sources.bind(descriptor['path'], descriptor['sha256']).read_text())
        if case['completed'] != descriptor['completed'] or any(case.get(k) is not False for k in
                ('live_qualified','actual_fill_verified','unseen_validation')):
            raise ValueError('Invalid case completion or qualification')
        row = dict(id=arm, name=name, completed=case['completed'], path=descriptor['path'],
                   sha256=descriptor['sha256'], metrics=None, rank=None,
                   candidate_coverage=case.get('candidate_coverage'), live_qualified=False)
        if not case['completed']:
            if case.get('summary') is not None:
                raise ValueError('Incomplete account has a full-period return')
            row.update(reason=case.get('reason'), last_date=case.get('last_date'),
                       completed_sessions=case.get('completed_sessions'))
            rows[arm] = row
            continue
        try:
            guard_certificate_inventory(case)
        except UnverifiedCertificateInventory as exc:
            row.update(completed=False, replay_completed=True,
                validation_status='certificate_inventory_unverified', reason=str(exc),
                last_date=END, replayed_sessions=len(case['account'].get('daily', [])),
                completed_sessions=None)
            rows[arm] = row
            continue
        metrics = account_metrics(case)
        from skills.all_entry_poya_completion import settlement_notes
        evidence_notes = settlement_notes(case)
        if case.get('settlement_evidence_notes', []) != evidence_notes:
            raise ValueError('Settlement source limitations are absent or differ')
        metrics.update(holding_scope(case['account'], 1 if arm == 'benchmark' else 3))
        daily, trades = case['account']['daily'], case['account']['trades']
        dates = [r['date'] for r in daily]
        if len(dates) != 666 or (dates[0], dates[-1]) != (START, END) or calendar is not None and dates != calendar:
            raise ValueError('Completed accounts require the same 666 market sessions')
        calendar = dates
        years = (date.fromisoformat(END)-date.fromisoformat(START)).days/365.25
        metrics['cagr'] = (metrics['final_nav']/metrics['initial_cash'])**(1/years)-1
        metrics['mean_daily_cash_weight'] = sum(r['cash']/r['nav'] for r in daily)/len(daily)
        average_nav = sum(r['nav'] for r in daily)/len(daily)
        metrics['one_way_turnover_multiple'] = sum(t['gross'] for t in trades)/2/average_nav
        row.update(metrics=metrics, summary=case['summary'], completed_sessions=len(daily),
            cohorts=describe_benchmark(case) if arm == 'benchmark' else describe_cohorts(case),
            pending_cash_valuation=pending_cash_claims(case),
            execution_exclusion_reasons=dict(Counter(r['failure_reason'] for r in case.get('data_gap_exclusions',[]))),
            profile_quality_exclusions=len(case.get('profile_data_exclusions',[])),
            profile_queries=len(case.get('profile_queries',[])),
            annual=metrics['annual'], settlement_evidence_notes=evidence_notes)
        rows[arm] = row
    benchmark = rows['benchmark']['metrics']
    ranked = sorted((r for k,r in rows.items() if k != 'benchmark' and r['completed']),
                    key=lambda r: (-r['metrics']['total_return'], r['id']))
    for i,row in enumerate(ranked,1):
        row['rank'] = i
        row['metrics']['excess_total_return_pp'] = None if benchmark is None else 100*(row['metrics']['total_return']-benchmark['total_return'])
    result = dict(schema='all_entry_account_comparison_publication_v1', start=START, end=END,
        initial_cash=1_000_000, position_count=3, position_scope='active_stock_slots', strategy_count=31,
        completed_strategies=sum(r['completed'] for k,r in rows.items() if k != 'benchmark'),
        all_completed=all(r['completed'] for r in rows.values()), cases=rows,
        ranking=[r['id'] for r in ranked], best_completed_strategy=ranked[0]['id'] if ranked else None,
        best_is_global=all(r['completed'] for r in rows.values()),
        comparison_model=dict(ranking='signal_amount20_desc_stock_id_asc',
            entry='T_close_then_T_plus_1', exit='loss12_time63_post_entry_three_black',
            buy_fraction=.7, sell_fraction=.3, commission_each_side=.001425,
            minimum_commission_per_channel=20, friction_each_side=.0045,
            sell_tax_stock=.003, sell_tax_benchmark=.001,
            general_signal_policy='daily_matched', context_signal_policy='known_first_parent'),
        source_report=dict(path=str(report_path.relative_to(ROOT)),sha256=digest(report_path)),
        candidate_manifest=report['candidate_manifest'], checked_source_count=len(sources.used),
        live_qualified=False, actual_fill_verified=False, unseen_validation=False,
        historical_period_already_researched=True,
        qualification_gaps=[
            '未用來選策略或調參的期間驗證，並校正本輪31組比較的選擇偏差。',
            '固定規則後向前紙上追蹤，保存當日訊號及當時可得資料；歷史重建不算即時留存。',
            '核實真實委託、排隊、成交量與滑價；日內高低區間只是事後成交代理。',
            '完整歷史可交易名冊、還原價與公告可用時間仍需認證。',
            '完成成本加嚴及進出延遲壓力測試，檢查跨年度與回撤承受能力。'],
        limitations=[
            '排名僅描述同一組假設下的已見歷史；不是未來獲利機率或實戰推薦。',
            '每日符合條件可再次成為候選；不能直接套用首日訊號研究的勝率。',
            '一般訊號資料未知、POC品質未知及帳戶未完成分開揭露。',
            '未結清持股與應收權益列入資產，但不列入已結清批次勝率。'])
    result['limitations'].extend(sorted({note for row in rows.values()
        for note in row.get('settlement_evidence_notes', [])}))
    for name,sha in tuple(sources.used.items()):
        sources.bind(name,sha)
    h = write(output,result)
    Path(output).with_suffix('.sha256').write_text(h+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=summarize(args.report,args.output)
    print(json.dumps({k:result[k] for k in ('completed_strategies','all_completed','best_completed_strategy')},ensure_ascii=False))
