#!/usr/bin/env python3
"""Run one fixed, offline multi-strategy signal outcome comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.scan_market_strategies import DEFAULT_BUNDLE, DEFAULT_POC, digest, load_poc
from skills.strategy_scanner.data import load_bundle
from skills.strategy_scanner.outcomes import study_signals
from skills.trial_registry import append_trial_registry


def render_study(report):
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('&', '\\u0026').replace('>', '\\u003e')
    return '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>多策略訊號比較</title><style>body{font:16px system-ui;background:#f7f7f1;color:#16333b;margin:24px auto;padding:0 20px;max-width:1400px}h1{font-size:30px}a{color:#007e76}select{padding:10px;border:1px solid #bbb;border-radius:8px;background:white;margin:8px}table{border-collapse:collapse;width:100%;background:white;font-variant-numeric:tabular-nums}th,td{padding:10px;border-bottom:1px solid #dde4e2;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#e5f0eb}p{line-height:1.7}.notice{padding:16px;background:#e5f0eb;border-radius:12px}.table{overflow:auto}.bad{color:#8c4429}.good{color:#08726a}small{display:block;color:#53676c}</style>
<a href="multi_strategy_scanner.html">← 每日策略掃描</a><h1>訊號出現後，有沒有優勢？</h1>
<p id="period"></p><p class="notice">每檔不受持股名額限制，只取「昨日已知不成立、今日成立」的首日訊號。T 收盤確認，T+1 開盤買、持有固定市場日後收盤賣，以還原價格估算並扣假設成本。這是訊號結果研究，<b>不是帳戶複利報酬，也不是實際可成交回測</b>。</p>
<label>訊號年份 <select id="year"><option value="all">全部</option></select></label>
<label>持有日數 <select id="horizon"><option value="20">20 日</option><option value="5">5 日</option><option value="60">60 日</option></select></label>
<label>排列 <select id="sort"><option value="name">策略名稱</option><option value="mean_excess_vs0050">平均超額報酬</option><option value="mean_net_return">平均淨報酬</option><option value="win_rate">勝率</option><option value="evaluated">可評估筆數</option></select></label>
<p>超額報酬以<b>每筆完全相同進出日期的 0050</b>相減。缺資料與未滿持有期分列；沒有當成零報酬或提早出場。</p>
<div class="table"><table><thead><tr><th>策略</th><th>首日事件</th><th>未成熟</th><th>資料不足</th><th>可評估</th><th>勝率</th><th>平均淨報酬</th><th>中位淨報酬</th><th>同期0050</th><th>平均超額</th></tr></thead><tbody id="rows"></tbody></table></div>
<p id="counts"></p><details><summary>整段期間的訊號資料覆蓋（不隨年份篩選變動）</summary><div id="signal-coverage"></div></details><p>所有方法與窗口皆保留，沒有只顯示贏家。同股及不同策略的訊號會重疊，筆數不等於獨立樣本數；這段歷史已研究多次，尚未完成多重比較校正，不能以排行認定實戰有效。</p>
<p>成本假設：買賣各手續費 0.1425%、各滑價 0.1%；個股賣出稅 0.3%、0050 賣出稅 0.1%。未模擬最低手續費、逐筆／零股成交容量、漲跌停排隊及公司行動實際到帳。历史名冊、部分延伸行情及 POC 覆蓋仍有限制。所有策略尚未實戰認證。</p>
<script id="report" type="application/json">'''+payload+'''</script><script>
const data=JSON.parse(document.getElementById('report').textContent);const el=id=>document.getElementById(id);const pct=v=>v===null?'—':(100*v).toFixed(2)+'%';
for(const year of [...new Set(data.summary.map(r=>r.year).filter(y=>y!=='all'))].sort()){const option=document.createElement('option');option.value=year;option.textContent=year;el('year').append(option);}
el('period').textContent=data.start+' ～ '+data.end+' · '+data.strategies.length+' 套進場規則 · '+data.hypothesis_count+' 組固定比較';
for(const [id,c] of Object.entries(data.signal_counts)){const p=document.createElement('p');p.textContent=id+'：條件成立 '+c.matching_stock_days+' 股票日；已知首日 '+c.first_events+'；成立但前日未知 '+c.matched_prior_unknown+'；訊號資料不足 '+c.unknown_stock_days+' 股票日。';el('signal-coverage').append(p);}
function render(){const year=el('year').value,h=Number(el('horizon').value),sort=el('sort').value;let rows=data.summary.filter(r=>r.year===year&&r.horizon===h);rows.sort((a,b)=>sort==='name'?a.name.localeCompare(b.name,'zh-Hant'):(b[sort]??-Infinity)-(a[sort]??-Infinity));el('rows').replaceChildren();for(const r of rows){const tr=document.createElement('tr');const values=[r.name,r.events,r.immature,r.stock_path_missing+r.benchmark_path_missing,r.evaluated,pct(r.win_rate),pct(r.mean_net_return),pct(r.median_net_return),pct(r.mean_benchmark_net_return),pct(r.mean_excess_vs0050)];values.forEach((value,i)=>{const td=document.createElement('td');td.textContent=value;if(i===0){const small=document.createElement('small');small.textContent=r.strategy_id;td.append(small);}if(i===9&&r.mean_excess_vs0050!==null)td.className=r.mean_excess_vs0050>0?'good':'bad';tr.append(td);});el('rows').append(tr);}el('counts').textContent='此表每一列單獨統計，請勿把各列事件報酬相加或相乘。';}
['year','horizon','sort'].forEach(id=>el(id).addEventListener('change',render));render();</script></html>'''


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Choose a new empty output directory; studies are immutable')
    started = time.perf_counter()
    data = load_bundle(args.bundle, start=args.start, end=args.end)
    poc, poc_info = load_poc(args.poc_report, bundle=args.bundle,
        manifest_hash=data['provenance']['source_hashes']['manifest.json'])
    source = dict(data['provenance'], poc=poc_info, external_data_requests=0,
        source_code_sha256={str(p.relative_to(ROOT)): digest(p) for p in
            sorted((ROOT/'skills/strategy_scanner').glob('*.py'))+[Path(__file__).resolve(), ROOT/'scripts/scan_market_strategies.py']},
        prereg_sha256=digest(ROOT/'docs/prereg_scanner_expansion_20261005.md'))
    report, events = study_signals(data['bars'], data['calendar'], start=args.start, end=args.end,
        original_signals=data['original_signals'], poc=poc, provenance=source)
    report['elapsed_compute_seconds'] = round(time.perf_counter()-started, 3)
    events.to_parquet(output/'events.parquet', index=False)
    (output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    (output/'index.html').write_text(render_study(report))
    # Every tested strategy/horizon enters the registry, regardless of outcome.
    registry_counts=[]
    for row in report['summary']:
        if row['year']!='all': continue
        registry_counts.append(append_trial_registry(dict(
            timestamp=datetime.now(timezone.utc).isoformat(), source='strategy_scanner_event_study',
            command=' '.join(sys.argv), study_type=report['study_type'], sharpe=None,
            start=args.start, end=args.end, params=dict(strategy_id=row['strategy_id'],
                version=row['version'], horizon=row['horizon'], costs=report['costs'],
                first_signal_only=True), outcome=row, report_sha256=digest(output/'summary.json'))))
    receipt=dict(created_at=datetime.now(timezone.utc).isoformat(),
        files_sha256={p.name:digest(p) for p in sorted(output.iterdir())},
        source=source, trial_registry_records=len(registry_counts),
        trial_registry_last_count=registry_counts[-1] if registry_counts else None)
    (output/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(output), strategies=len(report['strategies']),
        hypotheses=report['hypothesis_count'], event_horizon_rows=len(events),
        elapsed_seconds=round(time.perf_counter()-started,3), live_qualified=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=DEFAULT_BUNDLE)
    parser.add_argument('--poc-report',type=Path,default=DEFAULT_POC)
    parser.add_argument('--start',default='2024-01-02')
    parser.add_argument('--end',default='2026-10-02')
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
