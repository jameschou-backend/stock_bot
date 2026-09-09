#!/usr/bin/env python
"""Freeze a two-symbol slice once; reproduce all official-guidance pilot contrasts offline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import duckdb
import pandas as pd

from app.file_lock import file_lock
from skills.guidance_signals import RULES, build_signals
from skills.guidance_research import simulate_core

CACHE = ROOT/'.cache/guidance-research'
SPEC = 'docs/prereg_guidance_20260910.md'
SOURCES = ('docs/guidance_quarterly_sources_20260910.json', 'docs/guidance_annual_sources_20260910.json')
CODE = ('scripts/research_guidance.py', 'skills/guidance_signals.py', 'skills/guidance_research.py')
START, END = '2023-01-03', '2026-06-23'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str))
    temp.replace(path)


def rows(path):
    value = json.loads(path.read_text())
    if isinstance(value, list):
        return value
    return value['events'] if 'events' in value else value['records']


def prepare():
    CACHE.mkdir(parents=True, exist_ok=True)
    if (CACHE/'inputs.json').exists():
        verify()
        print('重用已封存的兩檔行情；0 requests。', flush=True)
        return
    started = time.perf_counter()
    parent = ROOT/'.cache/event-group-research'
    meta = json.loads((parent/'signal-inputs.json').read_text())
    original = {}
    files = []
    frames = {}
    for name in ('close-official.parquet', 'close-snapshot.parquet', 'trade-flags.parquet'):
        path = parent/name
        if sha(path) != meta['files_sha256'][name]:
            raise ValueError('Frozen parent prices changed: '+name)
        original[str(path.relative_to(ROOT))] = sha(path)
        frames[name] = pd.read_parquet(path, columns=['date', '2330', '0050'])
        frames[name].to_parquet(CACHE/name, index=False)
        files.append(name)
    quote_path = ROOT/'.cache/growth-flow-research/quotes.parquet'
    quote_meta = json.loads((quote_path.parent/'inputs.json').read_text())
    if sha(quote_path) != quote_meta['files_sha256'][quote_path.name]:
        raise ValueError('Frozen raw quotes changed')
    original[str(quote_path.relative_to(ROOT))] = sha(quote_path)
    with duckdb.connect() as con:
        raw = con.execute("SELECT * FROM read_parquet(?) WHERE stock_id IN ('2330','0050') AND trading_date >= DATE '2021-01-01' ORDER BY trading_date,stock_id", [str(quote_path)]).df()
    raw.to_parquet(CACHE/'raw.parquet', index=False)
    files.append('raw.parquet')
    audit = {}
    a = frames['close-official.parquet'].set_index('date')
    b = frames['close-snapshot.parquet'].set_index('date')
    for sid in ('2330', '0050'):
        paired = pd.DataFrame({'official': a[sid], 'snapshot': b[sid]}).loc[START:END].dropna()
        diff = ((paired.official/paired.official.shift(1))/(paired.snapshot/paired.snapshot.shift(1))-1).abs().dropna()
        audit[sid] = {'paired_observed_returns': len(diff), 'max_difference_bp': float(diff.max()*10000),
                      'differences_over_50bp': int(diff.gt(.005).sum())}
        if diff.gt(.005).any():
            raise ValueError('Reconcile pilot price bases before comparing returns: '+sid)
    write_json(CACHE/'inputs.json', {'schema': 1, 'prepared_at': datetime.now(timezone.utc).isoformat(),
        'files_sha256': {n:sha(CACHE/n) for n in files}, 'parent_files_sha256': original,
        'parent_manifest_sha256': sha(parent/'signal-inputs.json'),
        'price_audit': audit, 'benchmark_split': meta['action_check']['benchmark_split'],
        'preparation_seconds': round(time.perf_counter()-started, 3), 'finmind_requests': 0,
        'note': '兩版本一致只排除大幅口徑差，不能取代逐筆官方原價／現金流對帳。'})
    print('兩檔行情封存完成；0 FinMind requests。', flush=True)


def verify():
    value = json.loads((CACHE/'inputs.json').read_text())
    expected = {'close-official.parquet', 'close-snapshot.parquet', 'trade-flags.parquet', 'raw.parquet'}
    if value['schema'] != 1 or set(value['files_sha256']) != expected:
        raise ValueError('Incomplete frozen inputs; run --prepare-inputs explicitly')
    for name, fingerprint in value['files_sha256'].items():
        if sha(CACHE/name) != fingerprint:
            raise ValueError('Frozen guidance inputs changed: '+name)
    return value


def matrix(name):
    return pd.read_parquet(CACHE/name).set_index('date')


def run():
    started = time.perf_counter()
    inputs = verify()
    source_hashes = {name:sha(ROOT/name) for name in SOURCES}
    code_hashes = {name:sha(ROOT/name) for name in CODE}
    prereg_hash = sha(ROOT/SPEC)
    quarterly, annual = (rows(ROOT/name) for name in SOURCES)
    raw = pd.read_parquet(CACHE/'raw.parquet')
    volume = raw[raw.stock_id.eq('2330')].set_index('trading_date').raw_volume
    flags = matrix('trade-flags.parquet')
    results, baseline_results, event_tables, charts = [], [], {}, {}
    for basis in ('official', 'snapshot'):
        close = matrix(f'close-{basis}.parquet')
        entries, events = build_signals(close, volume.reindex(close.index), quarterly, annual)
        event_tables[basis] = events
        for scenario, slippage in (('base', .003), ('stress', .0045)):
            baselines = {mode:simulate_core(close, flags, [], start=START, end=END, mode=mode, slippage=slippage)
                         for mode in ('benchmark', 'static_mix')}
            for mode, sim in baselines.items():
                baseline_results.append({'basis':basis,'scenario':scenario,'mode':mode,**sim})
            for delay in ((0, 1) if basis == 'official' and scenario == 'stress' else (0,)):
                for rule in RULES:
                    delayed = []
                    for entry in entries[rule]:
                        index = int(close.index.searchsorted(entry['entry_date'])) + delay
                        if index < len(close.index):
                            delayed.append({**entry, 'entry_date': str(close.index[index].date())})
                    sim = simulate_core(close, flags, delayed, start=START, end=END, slippage=slippage)
                    row = {'basis':basis, 'scenario':scenario, 'delay':delay, 'rule':rule, 'name':RULES[rule],
                        'timing_diagnostic': rule in ('guidance_up','combined'), 'signal_count': len(delayed),
                        'excess_vs_0050': sim['summary']['total_return']-baselines['benchmark']['summary']['total_return'],
                        'excess_vs_static_mix': sim['summary']['total_return']-baselines['static_mix']['summary']['total_return'], **sim}
                    results.append(row)
                    if basis == 'official' and scenario == 'stress' and delay == 0:
                        charts[rule] = sim['curve']
            if basis == 'official' and scenario == 'stress':
                charts.update({mode:sim['curve'] for mode,sim in baselines.items()})
    if source_hashes != {name:sha(ROOT/name) for name in SOURCES} or code_hashes != {name:sha(ROOT/name) for name in CODE} or prereg_hash != sha(ROOT/SPEC):
        raise ValueError('Sources, spec or code changed during the experiment')
    verify()
    limitations = [
        '單公司、12 個交易事件，不能推論全台股；台積電已是0050的重要持股，主動部位是額外集中。',
        '固定歷史已被過去研究使用，並非未見測試集；所有結果都屬於研究，沒有實盤資格。',
        '季度比較使用帶日期的官方發佈版本，但目前抓取未能證明每一文件當時第一版；來源失敗保留缺值。',
        '季營收可能已由月營收事先推估，因此實績超標不等於法說當天的新資訊或超過市場共識。',
        '年度逐字稿經編輯，延遲與版本月份屬保守假設；年度上修及組合數字僅供時序診斷。',
        '固定混合基準檢查被動持有2330的效果，並非精確匹配曝險的因果估計。',
        '調整價格為合成總報酬單位，尚未逐筆現金流驗證，不含最低手續費、零股成交與容量限制。',
        '本輪未建立全市場即時官方公告抓取或自動下單，資料截點固定於2026-06-23。']
    report = {'schema':1, 'experiment':'guidance_20260910', 'research_only':True, 'live_qualified':False,
        'valid_strategy_evidence':False, 'created_at':datetime.now(timezone.utc).isoformat(),
        'start':START,'end':END,'events':event_tables, 'inputs':inputs,'limitations':limitations,
        'elapsed_seconds':round(time.perf_counter()-started,3),'finmind_requests':0,'model_training_runs':0,
        'code_sha256':code_hashes,'source_sha256':source_hashes,'preregistration_sha256':prereg_hash,
        'input_manifest_sha256':sha(CACHE/'inputs.json'), 'results':results, 'baselines':baseline_results,
        'charts':{key:[{'date':r['date'],'nav':r['nav']} for r in curve] for key,curve in charts.items()}}
    write_json(CACHE/'report.full.json', report)
    # Fast UI reads never load 20 full curves or recompute a strategy.
    report['results'] = [{k:v for k,v in r.items() if k not in ('curve','executions')} for r in results]
    report['baselines'] = [{k:v for k,v in r.items() if k not in ('curve','executions')} for r in baseline_results]
    write_json(CACHE/'report.summary.json', report)
    write_json(ROOT/'docs/research_guidance_20260910.json', report)
    lines = ['# 官方指引與公告後確認：研究結果', '', f'共同期間 {START}～{END}。單公司連續季度試驗，並非全市場選股策略。',
        '', '## 官方價格、壓力成本（雙邊稅費另加每邊0.45%滑價）', '', '| 規則 | 累積淨報酬 | 年化 | 最大回撤 | 相對0050（百分點） | 相對固定混合（百分點） |',
        '|---|---:|---:|---:|---:|---:|']
    for r in results:
        if (r['basis'],r['scenario'],r['delay']) == ('official','stress',0):
            s = r['summary']
            lines.append(f"| {r['name']} | {s['total_return']:.2%} | {s['cagr']:.2%} | {s['max_drawdown']:.2%} | {r['excess_vs_0050']*100:.2f} | {r['excess_vs_static_mix']*100:.2f} |")
    for r in baseline_results:
        if (r['basis'],r['scenario']) == ('official','stress'):
            s=r['summary']; label={'benchmark':'0050 持有','static_mix':'70% 0050＋30%2330 固定持有'}[r['mode']]
            lines.append(f"| {label} | {s['total_return']:.2%} | {s['cagr']:.2%} | {s['max_drawdown']:.2%} | — | — |")
    lines += ['', '## 限制', '', *['- '+s for s in limitations], '', '## 效能與重現', '',
        f"行情準備 {inputs['preparation_seconds']} 秒；20 組比較 {report['elapsed_seconds']} 秒；本輪研究 0 FinMind requests、0 模型重訓。",
        '', '`make prepare-guidance` 首次切片，`make research-guidance` 完全離線重算。工作台「策略驗證」可切換成本、價格版本及延後成交，並檢查每場來源。',
        '', '來源：[季度實績及事前指引](guidance_quarterly_sources_20260910.json)、[年度展望](guidance_annual_sources_20260910.json)。所有20組及逐筆交易保留於同名JSON；完整逐日成交在本機快取 report.full.json。']
    (ROOT/'docs/research_guidance_20260910.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'comparisons':len(results),'seconds':report['elapsed_seconds'],'finmind_requests':0,
        'primary':[{'rule':r['rule'],'return':r['summary']['total_return'],'vs0050':r['excess_vs_0050'],'vs_static':r['excess_vs_static_mix']}
                   for r in results if (r['basis'],r['scenario'],r['delay'])==('official','stress',0)]},ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-inputs', action='store_true')
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    with file_lock(CACHE/'research.lock', timeout=0):
        prepare() if args.prepare_inputs else run()
