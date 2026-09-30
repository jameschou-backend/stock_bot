#!/usr/bin/env python3
"""Export every frozen post-2024 signal without portfolio cash/slot selection."""
from pathlib import Path
import argparse
import html
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.research_exit_scenarios import read, write, sha
from skills.independent_signals import SignalPath, observe
from skills.trial_registry import append_trial_registry

BASE = ROOT / '.cache/partial-risk-2019-20260929/inputs-final'
SIGNALS = ROOT / '.cache/stock-universe-2019-20260929/signals-v2.json'
LABELS = {'closed': '已出場', 'open': '仍持有', 'pending_exit': '已觸發待賣',
          'unknown': '資料待核對', 'profit': '獲利', 'loss': '虧損', 'flat': '損益兩平',
          'unrealized': '未實現', 'loss12': '收盤跌12%停損', 'time63': '63日到期'}
ISSUES = {'missing_or_invalid_price': '持有路徑行情缺漏',
          'historical_identity_or_eligibility': '歷史身分或資格待核對',
          'raw_ohlc_conflict': '原始高低收盤矛盾', 'missing_or_invalid_volume': '成交量資料缺漏',
          'no_volume_on_assumed_fill': '假設成交日無量',
          'daily_adjustment_conflict': '單日異動或雙來源還原差異',
          'cumulative_adjustment_conflict': '雙來源累計還原差異'}
COLUMNS = {
    'stock_id': '代碼', 'name': '名稱', 'signal_date': '訊號日', 'entry_date': '假設買進日',
    'first_signal_for_stock': '此股2024起首次訊號', 'status': '狀態',
    'exit_signal_date': '出場訊號日', 'exit_date': '假設賣出日', 'reason': '出場理由',
    'outcome': '已出場盈虧', 'raw_entry_price': '買進原始中點價', 'raw_exit_price': '賣出原始中點價',
    'gross_return': '還原報酬率', 'net_return': '估算扣費已實現報酬率',
    'peak_close_return': '持有期最高收盤漲幅', 'trough_close_return': '持有期最低收盤漲幅',
    'held_sessions': '持有交易日數含首尾', 'first_5pct_day': '首次漲5%第幾日',
    'first_10pct_day': '首次漲10%第幾日', 'first_20pct_day': '首次漲20%第幾日',
    'unrealized_net_return': '估算扣費未實現報酬率', 'raw_mark_price': '未出場原始收盤價',
    'observed_end_date': '觀察截止日', 'entry_single_price': '買進日全天單一價格',
    'exit_single_price': '賣出日全天單一價格', 'entry_volume': '買進日成交股數',
    'exit_volume': '賣出日成交股數', 'issue': '資料問題',
    'relative20_at_signal': '訊號日20日領先0050幅度', 'volume_ratio_at_signal': '訊號日量比',
    'event_id': '訊號識別碼',
}


def summary(rows):
    closed = [r for r in rows if r['status'] == 'closed']
    wins = [r for r in closed if r['outcome'] == 'profit']
    losses = [r for r in closed if r['outcome'] == 'loss']
    values = [r['net_return'] for r in closed]
    return dict(signals=len(rows), stocks=len({r['stock_id'] for r in rows}), closed=len(closed),
        winners=len(wins), losers=len(losses), flat=len(closed)-len(wins)-len(losses),
        win_rate=len(wins)/len(closed) if closed else None,
        mean_net_return=float(np.mean(values)) if values else None,
        median_net_return=float(np.median(values)) if values else None,
        mean_win=float(np.mean([r['net_return'] for r in wins])) if wins else None,
        mean_loss=float(np.mean([r['net_return'] for r in losses])) if losses else None,
        stop_exits=sum(r['reason'] == 'loss12' for r in closed),
        time_exits=sum(r['reason'] == 'time63' for r in closed),
        time_exit_losses=sum(r['reason'] == 'time63' for r in losses),
        profit_giveback20=sum(r['peak_close_return'] >= .2 for r in losses),
        profit_giveback10=sum(r['peak_close_return'] >= .1 for r in losses),
        open=sum(r['status'] == 'open' for r in rows),
        pending_exit=sum(r['status'] == 'pending_exit' for r in rows),
        unknown=sum(r['status'] == 'unknown' for r in rows),
        single_price_entry=sum(r.get('entry_single_price', False) for r in rows),
        single_price_exit=sum(r.get('exit_single_price', False) for r in rows))


def display_frame(rows):
    f = pd.DataFrame(rows).reindex(columns=COLUMNS)
    for c in ('status', 'outcome', 'reason'):
        f[c] = f[c].map(lambda x: LABELS.get(x, x))
    f['issue'] = f['issue'].map(lambda x: ISSUES.get(x, x))
    return f.rename(columns=COLUMNS)


def browser_report(rows, report):
    fields = ['stock_id','name','signal_date','entry_date','exit_date','status','reason','outcome',
              'raw_entry_price','raw_exit_price','net_return','peak_close_return',
              'unrealized_net_return','held_sessions','first_signal_for_stock','issue']
    payload = json.dumps([[r.get(k) for k in fields] for r in rows], ensure_ascii=False, separators=(',',':'))
    payload = payload.replace('<', '\\u003c')
    annual = ''.join('<tr>' + ''.join('<td>'+str(x)+'</td>' for x in (
        y, s['signals'], s['closed'], s['winners'], s['losers'],
        f"{s['win_rate']:.1%}" if s['win_rate'] is not None else '—',
        f"{s['mean_net_return']:.2%}" if s['mean_net_return'] is not None else '—',
        f"{s['median_net_return']:.2%}" if s['median_net_return'] is not None else '—',
        s['open']+s['pending_exit'],s['unknown']))+'</tr>' for y,s in report['annual'].items())
    headers = ''.join('<th>'+html.escape(COLUMNS[k])+'</th>' for k in fields)
    main = report['all_signals']
    first = report['first_per_stock']
    prefix = f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>2024年起選股逐筆結果</title><style>
body{{font-family:system-ui,sans-serif;color:#17324d;background:#f5f8fc;margin:24px;line-height:1.6}}
h1{{font-size:27px}}h2{{font-size:20px}}.card{{background:white;padding:20px;border:1px solid #dce4ef;border-radius:12px;margin:16px 0}}
.scroll{{overflow:auto;max-height:68vh}}table{{border-collapse:collapse;font-size:14px;width:100%}}th,td{{padding:9px 12px;border-bottom:1px solid #e4e9ef;text-align:right;white-space:nowrap}}
th{{position:sticky;top:0;background:#17324d;color:white}}td:first-child,td:nth-child(2){{text-align:left}}input,select,button{{padding:8px;margin:5px;border:1px solid #b6c5d5;border-radius:6px;background:white;font-size:15px}}.pos{{color:#086445}}.neg{{color:#b03624}}.note{{color:#465b71}}a{{color:#155ca0}}
</style><h1>2024 年起，每筆選股訊號最後賺了嗎？</h1>
<p>原版 liquid_universe · 訊號 2024/1/2～2026/9/8 · 行情截至 2026/9/9</p>
<div class="card"><b>{main['signals']:,} 筆訊號 / {main['stocks']:,} 檔股票</b><p>已出場 {main['closed']:,} 筆，獲利 {main['winners']:,} 筆，虧損 {main['losers']:,} 筆，估算扣費勝率 {main['win_rate']:.2%}。</p>
<p>每檔第一次入選：已出場 {first['closed']:,} 檔，勝率 {first['win_rate']:.2%}。這只是另一個固定觀察口徑，不是挑每檔最好的一次。</p></div>
<div class="card note"><b>讀表方式</b><p>每筆訊號獨立假設買入，沒有本金、名額或持股重疊限制。同股可重複入選，這些訊號不是獨立樣本；平均單筆報酬不是帳戶報酬。</p>
<p>收盤訊號隔交易日以高低價中點試算；入場日還原收盤跌 12% 後隔日退出，否則入場索引 +63 交易日退出。到期賣出也可能虧損。最高漲幅使用賣出前的最高還原收盤，並非最高點賣出。</p>
<p>估算比例成本：每邊手續費 0.1425%、滑價 0.45%，賣出稅 0.3%；不含最低手續費與股數捨入。還原價涵蓋價格調整，但不處理配股交付及股利入帳。未出場報酬按最後收盤假設清算，尚未實現。</p>
<p>這是理想成交的選股診斷，沒有驗證每筆中點可成交、零股成交、漲跌停與容量。原始買賣價不可直接相除取代還原報酬，除權息與分割可能改變價格尺度。資料待核對者不計勝率，也沒有刪掉。</p></div>
<div class="card"><h2>依訊號年度</h2><div class="scroll"><table><thead><tr><th>年度</th><th>訊號</th><th>已出場</th><th>獲利</th><th>虧損</th><th>勝率</th><th>平均單筆</th><th>中位單筆</th><th>未出場</th><th>資料待核對</th></tr></thead><tbody>{annual}</tbody></table></div></div>
<div class="card"><h2>全部訊號明細</h2><p><a href="signals.csv" download>下載全部訊號 CSV</a>　<a href="stocks.csv" download>下載每檔彙整 CSV</a></p>
<input id="q" placeholder="股票代碼或名稱" aria-label="股票代碼或名稱"><select id="year" aria-label="年度"><option value="">所有年度</option><option>2024</option><option>2025</option><option>2026</option></select>
<select id="status" aria-label="狀態"><option value="">所有狀態</option><option value="closed">已出場</option><option value="open">仍持有</option><option value="pending_exit">已觸發待賣</option><option value="unknown">資料待核對</option></select>
<select id="outcome" aria-label="盈虧"><option value="">所有盈虧</option><option value="profit">獲利</option><option value="loss">虧損</option></select>
<select id="reason" aria-label="出場理由"><option value="">所有出場理由</option><option value="loss12">跌12%停損</option><option value="time63">63日到期</option></select>
<label><input type="checkbox" id="first">只看每檔第一次入選</label>
<select id="sort" aria-label="排序"><option value="date">按訊號日</option><option value="profit">已出場報酬高到低</option><option value="loss">已出場報酬低到高</option><option value="peak">最高收盤漲幅高到低</option></select>
<p id="count"></p><button id="prev">上一頁</button><button id="next">下一頁</button><span id="page"></span>
<div class="scroll"><table><thead><tr>{headers}</tr></thead><tbody id="rows"></tbody></table></div></div>
<script>const data={payload}; const labels={json.dumps(LABELS,ensure_ascii=False)}; const issues={json.dumps(ISSUES,ensure_ascii=False)};
'''
    return prefix + r'''
let page=0, selected=[]; const $=id=>document.getElementById(id);
function filter(){let q=$('q').value.trim().toLowerCase();selected=data.filter(r=>(!q||(r[0]+' '+r[1]).toLowerCase().includes(q))&&(!$('year').value||r[2].startsWith($('year').value))&&(!$('status').value||r[5]===$('status').value)&&(!$('outcome').value||r[7]===$('outcome').value)&&(!$('reason').value||r[6]===$('reason').value)&&(!$('first').checked||r[14]));let mode=$('sort').value;if(mode!=='date'){let k=mode==='peak'?11:10;selected.sort((a,b)=>(a[k]===null?1:b[k]===null?-1:(mode==='loss'?a[k]-b[k]:b[k]-a[k])));}page=0;render();}
function render(){let pages=Math.max(1,Math.ceil(selected.length/100));page=Math.max(0,Math.min(page,pages-1));$('count').textContent='符合條件 '+selected.length.toLocaleString()+' 筆';$('page').textContent='第 '+(page+1)+' / '+pages+' 頁，每頁100筆';$('prev').disabled=page===0;$('next').disabled=page===pages-1;let body=$('rows');body.replaceChildren();for(let r of selected.slice(page*100,page*100+100)){let tr=document.createElement('tr');r.forEach((v,i)=>{let td=document.createElement('td');let text=v===null?'—':typeof v==='boolean'?(v?'是':'否'):v;if(v!==null&&[10,11,12].includes(i)){text=(v*100).toFixed(2)+'%';td.className=v>0?'pos':v<0?'neg':'';}else if(v!==null&&[8,9].includes(i))text=v.toLocaleString(undefined,{maximumFractionDigits:3});else if([5,6,7].includes(i))text=labels[v]||text;else if(i===15)text=issues[v]||text;td.textContent=text;tr.append(td);});body.append(tr);}}
for(let id of ['q','year','status','outcome','reason','first','sort'])$(id).addEventListener('input',filter);$('prev').onclick=()=>{page--;render();};$('next').onclick=()=>{page++;render();};filter();</script></html>'''


def run(output):
    if output.exists():
        raise ValueError('Preserve prior outputs; choose a new directory')
    output.mkdir(parents=True)
    refs = {}
    manifest = read(BASE / 'manifest.json')
    sealed = read(ROOT / '.cache/waiting-exit-20260930/final-a/report.json')['source_sha256']
    def bind(p, expected=None):
        h = sha(p)
        if expected is not None and h != expected:
            raise ValueError('Frozen input changed: '+str(p))
        refs[str(p.relative_to(ROOT))] = h
    for n in ['close-official.parquet','close-quality.parquet','eligibility.parquet',
              'quotes-unmasked.parquet','companies.parquet']:
        bind(BASE/n, manifest['files_sha256'][n])
    bind(BASE/'manifest.json')
    bind(SIGNALS, sealed[str(SIGNALS.relative_to(ROOT))])
    for p in [Path(__file__), ROOT/'skills/independent_signals.py', ROOT/'skills/exit_policy.py',
              ROOT/'skills/million_replay.py', ROOT/'docs/prereg_independent_signals_20261001.md']:
        bind(p)
    entries = sorted([e for e in read(SIGNALS)['entries']['liquid_universe']
                      if e['signal_date'] >= '2024-01-01'], key=lambda e:(e['signal_date'],e['event_id']))
    ids = sorted({e['members'][0] for e in entries})
    frames = [pd.read_parquet(BASE/n, columns=['date',*ids]).set_index('date') for n in
              ('close-official.parquet','close-quality.parquet','eligibility.parquet')]
    c, o, eligible = frames
    if eligible.isna().any().any() or any(t != np.dtype(bool) for t in eligible.dtypes):
        raise ValueError('Frozen eligibility must contain explicit nonmissing booleans')
    days = pd.DatetimeIndex(c.index)
    if str(days[-1].date()) != '2026-09-09' or any(not f.index.equals(c.index) for f in frames):
        raise ValueError('Frozen calendar differs')
    raw = pd.read_parquet(BASE/'quotes-unmasked.parquet')
    raw = raw[raw.stock_id.isin(ids)]
    raw_fields = {k: raw.pivot(index='date',columns='stock_id',values=k).reindex(index=days,columns=ids)
                  for k in ('close','high','low','volume')}
    names = pd.read_parquet(BASE/'companies.parquet').set_index('stock_id')['name'].to_dict()
    paths = {sid: SignalPath(days, c[sid].to_numpy(float), o[sid].to_numpy(float),
                            eligible[sid].to_numpy(bool), *[raw_fields[k][sid].to_numpy(float)
                            for k in ('close','high','low','volume')]) for sid in ids}
    rows, seen, event_ids = [], set(), set()
    for e in entries:
        sid = e['members'][0]
        if len(sid)!=4 or not sid.isdigit() or sid.startswith('0') or e['event_id'] in event_ids:
            raise ValueError('Non-stock or duplicate signal')
        event_ids.add(e['event_id'])
        index = days.get_loc(pd.Timestamp(e['entry_date']))
        if str(days[index-1].date()) != e['signal_date']:
            raise ValueError('Entry is not next-session execution')
        r = dict(event_id=e['event_id'],stock_id=sid,name=names[sid],
                 signal_date=e['signal_date'],entry_date=e['entry_date'],
                 first_signal_for_stock=sid not in seen,
                 relative20_at_signal=e['priority'],
                 volume_ratio_at_signal=e['leader_evidence']['leader_volume_ratio'])
        r.update(observe(paths[sid],int(index)))
        rows.append(r);seen.add(sid)
    report = dict(start_signal='2024-01-01',last_signal=entries[-1]['signal_date'],
                  data_end=str(days[-1].date()),all_signals=summary(rows),
                  first_per_stock=summary([r for r in rows if r['first_signal_for_stock']]),
                  annual={y:summary([r for r in rows if r['signal_date'].startswith(y)])
                          for y in sorted({r['signal_date'][:4] for r in rows})},
                  first_annual={y:summary([r for r in rows if r['first_signal_for_stock'] and r['signal_date'].startswith(y)])
                                for y in sorted({r['signal_date'][:4] for r in rows})},
                  source_sha256=refs,unseen_validation=False,live_qualified=False,
                  actual_fill_verified=False,mode='independent_adjusted_hl2_unit_return',
                  costs=dict(commission=.001425,slippage=.0045,sell_tax=.003,minimum_fee_modeled=False),
                  cash_account=False)
    display_frame(rows).to_csv(output/'signals.csv',index=False,encoding='utf-8-sig')
    display_frame([r for r in rows if r['first_signal_for_stock']]).to_csv(output/'first-signals.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rows).to_parquet(output/'signals.parquet',index=False)
    grouped = {}
    for r in rows:
        grouped.setdefault(r['stock_id'],[]).append(r)
    stocks = []
    for sid, group in grouped.items():
        s = summary(group)
        stocks.append(dict(代碼=sid,名稱=names[sid],首次訊號=group[0]['signal_date'],
            訊號次數=s['signals'],已出場=s['closed'],獲利=s['winners'],虧損=s['losers'],
            勝率=s['win_rate'],平均單筆估算扣費報酬=s['mean_net_return'],
            停損出場=s['stop_exits'],到期出場=s['time_exits'],
            尚未出場=s['open']+s['pending_exit'],資料待核對=s['unknown']))
    pd.DataFrame(stocks).to_csv(output/'stocks.csv',index=False,encoding='utf-8-sig')
    (output/'index.html').write_text(browser_report(rows,report),encoding='utf-8')
    report['exports_sha256']={p.name:sha(p) for p in sorted(output.iterdir())}
    write(output/'report.json',report)
    print(json.dumps({k:report[k] for k in ('all_signals','first_per_stock','annual')},ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve();output.relative_to(ROOT)
    trial=dict(source='independent_signals_20261001',timestamp=datetime.now(timezone.utc).isoformat(),
               command=' '.join(sys.argv),output=str(output.relative_to(ROOT)),live_qualified=False)
    try:
        run(output)
    except Exception as exc:
        append_trial_registry(dict(trial,completed=False,error=type(exc).__name__+': '+str(exc)))
        raise
    append_trial_registry(dict(trial,completed=True))


if __name__=='__main__':
    main()
