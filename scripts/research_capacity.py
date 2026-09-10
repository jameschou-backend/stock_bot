#!/usr/bin/env python
"""Separate position capacity from causal market-adjusted event priority."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from app.file_lock import file_lock
from app import regime_switch_research as parent_reader
from scripts import research_diffusion as source
from scripts.research_regime_switch import delay_events
from skills.diffusion_portfolio import simulate_baskets
from skills.regime_state import gate_events
from skills.residual_priority import rescore_events

CACHE = ROOT / '.cache/capacity-research'
PARENT = ROOT / '.cache/regime-switch-research'
SPEC = ROOT / 'docs/prereg_capacity_20260910.md'
CODE = ('scripts/research_capacity.py', 'skills/residual_priority.py')
RULES = {'control3': '原排序・3個部位', 'capacity6': '原排序・6個部位',
         'matched3': '可評分子集・原排序3個部位', 'residual3': '同子集・大盤校正排序3個部位'}
START, END = source.START, source.END


def code_hashes():
    return {name: source.sha(ROOT / name) for name in CODE}


def versions():
    return {'numpy': np.__version__, 'pandas': pd.__version__}


def parent_report():
    report = parent_reader.overview()
    if not report['available']:
        raise ValueError('The sealed regime-switch evidence is unavailable; restore its original inputs first.')
    return report


def case_name(rule, basis, scenario, delay):
    return f'case-{rule}-{basis}-{scenario}-{delay}.json'


def ranking_stats(original, matched, rescored, original_count, rejected):
    """Same-date order changes are opportunities, not proof of binding slots."""
    if {e['event_id'] for e in matched} != {e['event_id'] for e in rescored}:
        raise ValueError('Priority comparison requires exactly the same events')
    groups = defaultdict(list)
    new = {e['event_id']: e for e in rescored}
    for event in matched:
        other = new[event['event_id']]
        if any(event[k] != other[k] for k in ('signal_date', 'entry_date', 'members')):
            raise ValueError('Priority cannot change event identity or timing')
        groups[event['entry_date']].append(event)
    multiple = changed = 0
    examples = []
    for day, events in sorted(groups.items()):
        if len(events) < 2:
            continue
        multiple += 1
        order = [e['event_id'] for e in sorted(events, key=lambda e: (-e['priority'], e['event_id']))]
        reordered = sorted(order, key=lambda eid: (-new[eid]['priority'], eid))
        if order != reordered:
            changed += 1
            if len(examples) < 10:
                examples.append({'date': day, 'original_order': order, 'residual_order': reordered})
    return {'original_count': original_count, 'trend_count': len(original),
            'scoreable_trend_count': len(matched),
            'score_rejections': dict(Counter(r['reason'] for r in rejected)),
            'multiple_event_days': multiple, 'reordered_days': changed, 'reorder_examples': examples}


def prepare_signals():
    started = time.perf_counter()
    parent = parent_report()
    CACHE.mkdir(parents=True, exist_ok=True)
    code, protocol = code_hashes(), source.sha(SPEC)
    parent_hash = source.sha(PARENT / 'report.summary.json')
    prices = {basis: source.read_matrix(f'close-{basis}.parquet') for basis in ('official', 'snapshot')}
    files, stats = {}, {}
    for basis, close in prices.items():
        other = prices['snapshot' if basis == 'official' else 'official']
        raw = json.loads((source.CACHE / f'signals-{basis}.json').read_text())['entries']['leader_now']
        states = pd.read_parquet(PARENT / f'states-{basis}.parquet').set_index('date')
        states.index = pd.to_datetime(states.index)
        original, gate_rejected = gate_events(raw, states)
        scores = rescore_events(close, original, other)
        accepted_ids = {e['event_id'] for e in scores['events']}
        matched = [e for e in original if e['event_id'] in accepted_ids]
        for length in (600, 1000):
            subset = [e for e in original if pd.Timestamp(e['signal_date']) <= close.index[length-1]]
            prefix = rescore_events(close.iloc[:length], subset, other.iloc[:length])
            subset_ids = {e['event_id'] for e in subset}
            for key in ('events', 'rejections', 'diagnostics'):
                if prefix[key] != [e for e in scores[key] if e['event_id'] in subset_ids]:
                    raise ValueError('Future prices changed earlier residual evidence: ' + basis)
        stats[basis] = ranking_stats(original, matched, scores['events'], len(raw), scores['rejections'])
        payload = {'original_events': original, 'matched_events': matched, 'residual_events': scores['events'],
                   'gate_rejections': gate_rejected, 'score_rejections': scores['rejections'],
                   'score_diagnostics': scores['diagnostics'], 'stats': stats[basis]}
        filename = f'signals-{basis}.json'
        source.write_json(CACHE / filename, payload)
        files[filename] = source.sha(CACHE / filename)
        print(basis, json.dumps(stats[basis], ensure_ascii=False), flush=True)
    if code != code_hashes() or protocol != source.sha(SPEC) or parent_hash != source.sha(PARENT / 'report.summary.json'):
        raise ValueError('Research implementation or parent changed during signal preparation')
    parent_report()
    source.write_json(CACHE / 'signals.json', {'schema': 1, 'code_sha256': code, 'protocol_sha256': protocol,
        'versions': versions(), 'parent_report_sha256': parent_hash,
        'parent_signal_manifest_sha256': parent['states']['diffusion_signal_manifest_sha256'],
        'files_sha256': files, 'stats': stats, 'prefix_invariance_passed': True,
        'elapsed_seconds': round(time.perf_counter()-started, 3), 'finmind_requests': 0})
    print('已封存候選與分數，尚未計算新收益。', flush=True)


def verify_signals():
    parent = parent_report()
    manifest = json.loads((CACHE / 'signals.json').read_text())
    if (manifest['schema'] != 1 or manifest['code_sha256'] != code_hashes()
            or manifest['protocol_sha256'] != source.sha(SPEC) or manifest['versions'] != versions()
            or manifest['parent_report_sha256'] != source.sha(PARENT / 'report.summary.json')
            or manifest['parent_signal_manifest_sha256'] != parent['states']['diffusion_signal_manifest_sha256']
            or manifest['prefix_invariance_passed'] is not True
            or set(manifest['files_sha256']) != {'signals-official.json', 'signals-snapshot.json'}):
        raise ValueError('Signal provenance changed; explicitly run make prepare-capacity')
    for name, digest in manifest['files_sha256'].items():
        if source.sha(CACHE / name) != digest:
            raise ValueError('Frozen capacity signal changed: ' + name)
    return manifest, parent


def finalize_summary(sim):
    sim['summary']['annual_returns'] = source.annual_returns(sim['curve'])
    sim['summary']['mean_cash_weight'] = float(np.mean([row['cash']/row['nav'] for row in sim['curve']]))


def verify_control(sim, reference):
    if not reference.get('available'):
        raise ValueError('Original control ledger is unavailable')
    if any(sim[key] != reference[key] for key in ('summary', 'curve', 'executions', 'cohorts', 'rejections')):
        raise ValueError('Original three-position strategy did not reproduce exactly')


def save_case(sim, rule, basis, scenario, delay, signal, audit, files):
    name = case_name(rule, basis, scenario, delay)
    case = {'rule': rule, 'basis': basis, 'scenario': scenario, 'delay': delay, **sim,
            'valuation_audit': audit, 'gate_rejections': signal.get('gate_rejections', []),
            'score_rejections': signal.get('score_rejections', []),
            'score_diagnostics': signal.get('score_diagnostics', [])}
    source.write_json(CACHE / name, case)
    files[name] = source.sha(CACHE / name)
    return {'rule': rule, 'name': RULES.get(rule, '0050持有'), 'basis': basis,
            'scenario': scenario, 'delay': delay, 'case_file': name, 'summary': sim['summary'],
            'valuation_audit': audit, 'rejection_counts': dict(Counter(e['reason'] for e in sim['rejections']))}


def compare_cohorts(sims, basis, scenario, delay):
    rows = []
    for rule, reference in (('capacity6', 'control3'), ('matched3', 'control3'), ('residual3', 'matched3')):
        a = {e['event_id'] for e in sims[rule]['cohorts']}
        b = {e['event_id'] for e in sims[reference]['cohorts']}
        rows.append({'basis': basis, 'scenario': scenario, 'delay': delay, 'rule': rule,
                     'reference': reference, 'shared': len(a & b),
                     'only_rule': sorted(a-b), 'only_reference': sorted(b-a)})
    return rows


def run():
    started = time.perf_counter()
    manifest, parent = verify_signals()
    prices = {basis: source.read_matrix(f'close-{basis}.parquet') for basis in ('official', 'snapshot')}
    flags = source.read_matrix('trade-flags.parquet')
    results, baselines, files, charts, comparisons = [], [], {}, {}, []
    for basis, close in prices.items():
        signal = json.loads((CACHE / f'signals-{basis}.json').read_text())
        anomalies = source.price_anomalies(close, prices['snapshot' if basis == 'official' else 'official'])
        for scenario, slippage in (('base', .003), ('stress', .0045)):
            bm = simulate_baskets(close, flags, [], start=START, end=END, mode='benchmark', slippage=slippage)
            finalize_summary(bm)
            baselines.append(save_case(bm, 'benchmark', basis, scenario, 0, {}, source.valuation_audit(bm, anomalies), files))
            for delay in ((0, 1) if (basis, scenario) == ('official', 'stress') else (0,)):
                sims = {}
                for rule in RULES:
                    events_key = ('original_events' if rule in ('control3', 'capacity6')
                                  else 'matched_events' if rule == 'matched3' else 'residual_events')
                    events = delay_events(signal[events_key], close.index, delay)
                    sim = simulate_baskets(close, flags, events, start=START, end=END,
                                           slots=6 if rule == 'capacity6' else 3, slippage=slippage)
                    finalize_summary(sim)
                    if rule == 'control3':
                        row = next(row for row in parent['results'] if (row['rule'], row['basis'], row['scenario'], row['delay'])
                                   == ('entry_only', basis, scenario, delay))
                        verify_control(sim, parent_reader.load_case(parent, row))
                    sims[rule] = sim
                    results.append(save_case(sim, rule, basis, scenario, delay, signal,
                                             source.valuation_audit(sim, anomalies), files))
                    print(rule, basis, scenario, delay, '完成', flush=True)
                comparisons.extend(compare_cohorts(sims, basis, scenario, delay))
                if (basis, scenario, delay) == ('official', 'stress', 0):
                    charts = {rule: [{'date': row['date'], 'nav': row['nav']} for row in sim['curve']]
                              for rule, sim in {**sims, 'benchmark': bm}.items()}
    for row in results:
        key = (row['basis'], row['scenario'], row['delay'])
        reference = 'matched3' if row['rule'] == 'residual3' else 'control3'
        ref = next(r for r in results if (r['rule'], r['basis'], r['scenario'], r['delay']) == (reference, *key))
        bm = next(r for r in baselines if (r['basis'], r['scenario']) == key[:2])
        row.update(reference_rule=reference,
                   contrast_vs_reference=row['summary']['total_return']-ref['summary']['total_return'],
                   excess_vs_0050=row['summary']['total_return']-bm['summary']['total_return'])
    verify_signals()
    report = {'schema': 1, 'experiment': 'capacity_priority_20260910', 'research_only': True,
              'live_qualified': False, 'valid_strategy_evidence': False,
              'start': START, 'end': END, 'signal_end': '2025-12-31',
              'created_at': datetime.now(timezone.utc).isoformat(), 'code_sha256': code_hashes(),
              'protocol_sha256': source.sha(SPEC), 'signal_manifest_sha256': source.sha(CACHE/'signals.json'),
              'parent_report_sha256': manifest['parent_report_sha256'], 'signals': manifest,
              'signal_stats': manifest['stats'], 'case_files_sha256': files,
              'results': results, 'baselines': baselines, 'charts': charts, 'cohort_comparisons': comparisons,
              'elapsed_seconds': round(time.perf_counter()-started, 3),
              'preparation_elapsed_seconds': manifest['elapsed_seconds'], 'finmind_requests': 0,
              'prediction_model_training_runs': 0, 'control_reproduction_passed': True,
              'limitations': [
                  '全段歷史已使用，20組方法只是敏感度比較，沒有未見驗證；不得自動啟用策略。',
                  '大盤校正僅改既有領先事件的同日處理顺序，沒有重選群內股票，也沒有跨日排隊或換掉舊持倉。',
                  '排序變更也可能因同日先扣費而微幅改變投入比例；多事件日不代表實際名額不足。',
                  '新分數使用更長且完整的雙版本窗口；matched3分開資料筛選效果，不能直接將residual3與control3差額全歸因於排序。',
                  '增加部位可能稀釋贏家、改變後續事件與複利，較多股票不保證報酬提高或回撤降低。',
                  '沿用當前名冊與合成還原單位，仍有存活者及歷史價格對帳限制；完整漲跌停、整零股深度及最低手續費尚未建模。',
                  '0050為單一市場代理，20日殘差和不是學術多因子策略的完整複製；參數沒有搜尋。']}
    source.write_json(CACHE/'report.summary.json', report)
    source.write_json(ROOT/'docs/research_capacity_20260910.json', report)
    default = [r for r in results if (r['basis'], r['scenario'], r['delay']) == ('official', 'stress', 0)]
    bm = next(r for r in baselines if (r['basis'], r['scenario']) == ('official', 'stress'))
    lines = ['# 部位容量與大盤校正排序：固定比較結果', '',
             f'期間{START}～{END}，新訊號截止2025-12-31；官方參考價格，每邊滑價0.45%另扣稅費。',
             '四方法分開容量、資料篩選及同日排序，均屬歷史探索。', '',
             '| 方法 | 累積試算淨報酬 | 年化 | 最大回撤 | 買入事件 | 額滿拒絕 | 估值疑點 | 期末 |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for row in [*default, bm]:
        s = row['summary']
        liquidation = '已清算' if s['final_liquidation_complete'] else '含未平倉估值'
        lines.append(f"| {row['name']} | {s['total_return']:.2%} | {s['cagr']:.2%} | {s['max_drawdown']:.2%} | {s['entered_cohorts']} | {row['rejection_counts'].get('slots_full',0)} | {row['valuation_audit']['finding_count']} | {liquidation} |")
    lines += ['', *['- '+note for note in report['limitations']], '',
              f"候選準備{report['preparation_elapsed_seconds']}秒，20方法＋4基準含稽核{report['elapsed_seconds']}秒，研究FinMind0次。",
              '', '`make prepare-capacity` 先封存分數，`make research-capacity` 重現比較；逐筆原始帳目見 .cache/capacity-research/case-*.json。']
    (ROOT/'docs/research_capacity_20260910.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'comparisons': len(results), 'seconds': report['elapsed_seconds'], 'controls_reproduced': True}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-signals', action='store_true')
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    with file_lock(CACHE/'research.lock', timeout=0):
        if args.prepare_signals:
            prepare_signals()
        else:
            run()
