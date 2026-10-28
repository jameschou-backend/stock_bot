#!/usr/bin/env python3
"""Run the preregistered 24-arm, chronologically separated course study."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from skills.course_bundle import Config, configurations, prepare_features, signals, simulate
from scripts.prepare_course_bundle import OUT, SPEC, sha, save_json


def choose_validation(rows):
    if not rows:
        raise ValueError('No validation candidates')
    return sorted(rows, key=lambda r: (-r['summary']['total_return'],
        abs(r['summary']['max_drawdown']), r['summary']['traded_notional'], r['name']))[0]


def export_case(folder, name, run):
    run['curve'].to_csv(folder / (name + '-nav.csv'), index=False)
    pd.DataFrame(run['trades']).to_csv(folder / (name + '-trades.csv'), index=False)
    save_json(folder / (name + '-decisions.json'), run['decisions'])
    save_json(folder / (name + '.json'), {k:v for k,v in run.items() if k not in ('curve','decisions')})


def yearly(curve):
    x = curve.copy().set_index('date')
    returns = x.equity.pct_change(fill_method=None).dropna()
    return {str(year): float((1+values).prod()-1) for year,values in returns.groupby(returns.index.year)}


def render_report(folder, report, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from html import escape
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    chart_labels={'full':'Course bundle','benchmark_full':'0050 buy and hold','baseline_full':'Momentum baseline'}
    for name, curve in curves.items():
        nav = curve.set_index('date').equity
        axes[0].plot(nav.index, nav/nav.iloc[0], label=chart_labels[name], linewidth=1.5)
        axes[1].plot(nav.index, (nav/nav.cummax()-1)*100, label=name, linewidth=1)
    axes[0].set_ylabel('Net NAV / initial capital'); axes[0].legend()
    axes[1].set_ylabel('Drawdown %'); axes[1].set_xlabel('Date')
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('Course-derived momentum bundle | retrospective five-year daily model')
    fig.tight_layout(); fig.savefig(folder/'equity.png', dpi=160); plt.close(fig)
    def pct(v): return '—' if v is None else f'{v*100:.2f}%'
    cases=report['cases']; labels={'full':'五年回溯：選定組合','test':'獨立起算驗證段：2025 起',
        'benchmark_full':'0050 五年比較','benchmark_test':'0050 驗證段比較',
        'baseline_full':'固定動能基準：五年','baseline_test':'固定動能基準：驗證段','stress_full':'組合五年：滑價提高到每邊 0.3%','stress_test':'組合驗證段：滑價提高到每邊 0.3%'}
    rows=''.join('<tr><td>'+escape(labels.get(k,k))+'</td>'+''.join('<td>'+v+'</td>' for v in [pct(s['total_return']),pct(s['cagr']),pct(s['max_drawdown']),str(s['completed_trades']),pct(s['win_rate'])])+'</tr>' for k,s in cases.items())
    limits=''.join('<li>'+escape(v)+'</li>' for v in report['limitations_zh'])
    years=''.join(f'<tr><td>{y}</td><td>{pct(r)}</td><td>{pct(report["benchmark_yearly"].get(y))}</td></tr>' for y,r in report['yearly'].items())
    weights=report['winner_weights']; cfg=report['winner_config']; full=cases['full']
    conclusion=('本版未通過實用性檢驗：五年扣費後虧損，不能據此稱為最佳解。' if full['total_return']<0 else '這是有限候選中的選定版本，仍未取得實盤資格。')
    content=f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>愛德恩課程衍生組合：五年研究</title>
<style>body{{font:17px/1.65 system-ui;max-width:1120px;margin:40px auto;padding:0 24px;color:#16313e;background:#f5f8fa}}h1,h2{{color:#103f52}}table{{border-collapse:collapse;width:100%;background:white;margin:20px 0}}td,th{{padding:10px;border-bottom:1px solid #dce6eb;text-align:right}}td:first-child,th:first-child{{text-align:left}}.note{{background:#fff3d4;padding:18px;border-left:5px solid #c68a21}}img{{width:100%;background:white}}code{{overflow-wrap:anywhere}}a{{color:#087397}}</style>
<h1>課程衍生動能組合｜近五年回測</h1><p>日線多方；期間 {report['start']}～{report['end']}。這是自訂、可重現的研究版本，非作者公式重現。</p>
<div class="note">五年全期包含選參數期間，屬回溯績效。2025 起是本次研究保留的時間段，但專案曾研究過這段歷史，不宣稱全新未見資料。使用還原單位淨值模型，未證明真實成交。</div>
<p><strong>{conclusion}</strong>符合更多條件是排序依據，尚未校準成上漲機率。</p>
<h2>選定設定</h2><p><code>{escape(report['winner'])}</code>；動能權重 {escape(str(report['winner_weights']))}（20 日漲幅／20 日均線乖離／成交額相對強度）。固定 5 個持股名額、7% 停損、無槓桿。100 萬為標準化試算本金，並非使用者資金設定。</p>
<p>動能 M = {weights[0]:.0%} × 20 日報酬百分位 + {weights[1]:.0%} × MA20 乖離百分位 + {weights[2]:.0%} × 成交金額相對強度百分位。三項都在當日合格、資料完整且前日成交額超過 5 億的股票中排名，M 介於 0～100；不能對應作者的 7.5 分。</p>
<p>排序分數 = 0.7 × M + 當日投信買超 10 分 + 當日外資買超 10 分 + 量增達前 20 日均量 1.2 倍 10 分。各篩選器取聯集，不按重複上榜次數加分。</p>
<p>選定門檻為 {cfg['threshold']} 分；此門檻用於動能型篩選器，突破邊緣等其他篩選器依其規則納入。所有候選仍須 20 日報酬為正、站上 MA20，且 0050 站上 MA5、MACD 柱狀值增加。</p>
<p>選定進場：{'只做突破' if cfg['entries']=='breakout' else '突破或回檔'}。突破須收盤超過前 40 日最高價、紅 K、量增達 1.2 倍；隔日開盤跳空超過 ±3% 跳過。每筆上限為前日淨值 1/5，不補滿持倉。出場：固定 7% 停損立即生效，或收盤跌破 MA{cfg['exit_ma']} 後隔日開盤賣出；停損跳空按較差開盤價估算，無法成交則順延。</p>
<p>選參數方式：2021～2023 訓練 24 組，依年化報酬 − 0.5 × 最大回撤絕對值選前 5 組（至少 20 筆平倉）；2024 以扣費報酬選定一組；2025 起測試不參與選擇。較高滑價比較不另選參數。</p>
<p>24 組訓練中扣費獲利者 {report['positive_train_count']} 組；5 組驗證中獲利者 {report['positive_validation_count']} 組。「選定」只是依固定排序挑出比較對象，不表示通過獲利門檻，也不能據此判定作者原策略有效或無效。</p>
<h2>扣除成本後結果</h2><table><tr><th>案例</th><th>累積報酬</th><th>年化</th><th>最大回撤</th><th>已完成交易</th><th>勝率</th></tr>{rows}</table><p>勝率只計已平倉交易；期末部位按市值扣除預估清算成本，未假裝已成交。</p>
<img src="equity.png" alt="Five-year normalized net asset value and drawdown curves">
<p>成本假設：買賣手續費每邊 0.1425%（最低 20 元），股票賣出稅 0.3%、0050 賣出稅 0.1%，滑價每邊 0.1%。五年實付手續費與交易稅合計 {full['commissions_and_taxes']:,.0f} 元（滑價另內含於成交價），平均年單邊周轉 {full['annual_one_way_turnover']:.1f} 倍。0050 期末清算成本為估計，沒有捏造賣出。</p>
<h2>逐年報酬</h2><table><tr><th>年度</th><th>選定組合</th><th>0050</th></tr>{years}</table><p>2021、2026 為部分年度。年度數字由同一個連續五年帳戶計算。</p>
<h2>組合內容</h2><p>量大動能、投信剛買準突破、投信動能連買、外資剛大買、外資投信同買、突破邊緣，以及「價量創 400 日新高」的日線改編，共同產生候選。排序按動能、投信、外資、量能分組，重複上榜不重複加分。突破／回檔訊號收盤確認後，下一交易日開盤模型執行。</p>
<h2>資料與解讀限制</h2><ul>{limits}</ul>
<h2>來源與規則對照</h2><p><a href="methodology.md">完整策略規則與重跑方式</a>。動能三個構成參考 <a href="https://www.cmoney.tw/learn/course/edwintsai/topic/3692">CMoney 課程說明</a>，期間、權重及門檻由本研究明確定義。0050 分割停牌依 <a href="https://www.twse.com.tw/zh/ETFortune/announcement?company=A00005&amp;date=20250617&amp;fund=0050&amp;seq=1&amp;type=other">證交所公告</a>核對。</p>
<h2>可核對檔案</h2><p><a href="report.json">完整研究結果</a> · <a href="trials.json">全部訓練及驗證試驗</a> · <a href="coverage.csv">每日資料覆蓋</a> · <a href="full-nav.csv">五年每日淨值</a> · <a href="full-trades.csv">五年交易明細</a> · <a href="test-nav.csv">驗證段每日淨值</a> · <a href="test-trades.csv">驗證段交易明細</a> · <a href="spec.json">事前固定規格</a></p></html>'''
    (folder/'index.html').write_text(content)


def main():
    began=time.monotonic()
    spec=json.loads(SPEC.read_text())
    manifest=json.loads((OUT/'inputs.json').read_text())
    if manifest['spec_sha256'] != sha(SPEC):
        raise ValueError('Preregistered specification changed after input freeze')
    refs={str(SPEC.relative_to(ROOT)):sha(SPEC)}
    for name,expected in manifest['files_sha256'].items():
        if sha(OUT/name)!=expected: raise ValueError('Input hash mismatch: '+name)
        refs[str((OUT/name).relative_to(ROOT))]=expected
    from skills.strict_tick_inputs import full_session_halts
    identity_path=ROOT/spec['source_bundle']/'identity.json'
    source_manifest=json.loads((identity_path.parent/'manifest.json').read_text())
    if sha(identity_path)!=source_manifest['files_sha256']['identity.json']:
        raise ValueError('Historical identity source hash mismatch')
    identity=json.loads(identity_path.read_text())
    halts=[r for r in full_session_halts(identity) if r['end']>spec['start'] and r['start']<=spec['end']]
    refs[str(identity_path.relative_to(ROOT))]=sha(identity_path)
    refs['skills/strict_tick_inputs.py']=sha(ROOT/'skills/strict_tick_inputs.py')
    for name in {r['source_path'] for r in halts}:
        expected=identity['source_sha256'].get(name)
        if not expected or sha(ROOT/name)!=expected:
            raise ValueError('Official halt evidence missing or changed: '+name)
        refs[name]=expected
    extra_halts_path=ROOT/'docs/course_bundle_halt_evidence_20261028.json'
    refs[str(extra_halts_path.relative_to(ROOT))]=sha(extra_halts_path)
    for row in json.loads(extra_halts_path.read_text()):
        if not row['end_exclusive'] or sha(ROOT/row['source_path'])!=row['source_sha256']:
            raise ValueError('Supplemental official halt evidence changed')
        refs[row['source_path']]=row['source_sha256']
        halts.append(row)
    code=['skills/course_bundle.py','scripts/research_course_bundle.py','scripts/prepare_course_bundle.py']
    methodology=ROOT/'docs/course_bundle_research_20261028.md'
    refs[str(methodology.relative_to(ROOT))]=sha(methodology)
    refs.update({name:sha(ROOT/name) for name in code})
    digest=hashlib.sha256(json.dumps(refs,sort_keys=True).encode()).hexdigest()[:16]
    folder=OUT/('run-'+digest)
    if (folder/'report.json').exists():
        output_receipt=json.loads((folder/'outputs.json').read_text())
        for name,expected in output_receipt.items():
            if sha(folder/name)!=expected:raise ValueError('Research output changed: '+name)
        print('Existing immutable result verified:',folder);return
    folder.mkdir(exist_ok=True)
    (folder/'spec.json').write_bytes(SPEC.read_bytes())
    (folder/'methodology.md').write_bytes(methodology.read_bytes())
    print('Preparing causal features...',flush=True)
    f=prepare_features(pd.read_parquet(OUT/'bars.parquet'),pd.read_parquet(OUT/'institutional.parquet'),pd.read_parquet(OUT/'turnover.parquet'),halts=halts)
    f['coverage'].to_csv(folder/'coverage.csv',index=False)
    save_json(folder/'audit.json',f['audit']);print(f['audit'],flush=True)
    signal_cache={}; configs={c.name:c for c in configurations()}
    def sig(c):
        key=(c.weights,c.threshold,c.entries,c.baseline)
        if key not in signal_cache: signal_cache[key]=signals(f,c)
        return signal_cache[key]
    train=[]
    for n,c in enumerate(configs.values(),1):
        run=simulate(f,sig(c),c,*spec['train'])
        s=run['summary']; objective=s['cagr']-.5*abs(s['max_drawdown'])
        train.append(dict(name=c.name,config=asdict(c),objective=objective,summary=s))
        save_json(folder/'train-progress.json',train)
        print('train',n,c.name,round(s['total_return'],4),flush=True)
    shortlist=sorted([r for r in train if r['summary']['completed_trades']>=20],key=lambda r:(-r['objective'],r['name']))[:5]
    if not shortlist:
        save_json(folder/'no-winner.json',{'reason':'No training configuration with 20 completed trades','trials':train})
        raise ValueError('No configuration passes preregistered training sample minimum')
    validation=[]
    for row in shortlist:
        c=configs[row['name']];run=simulate(f,sig(c),c,*spec['validation'])
        validation.append(dict(name=c.name,config=asdict(c),summary=run['summary']))
    winner=choose_validation(validation);c=configs[winner['name']]
    save_json(folder/'selection-before-test.json',dict(winner=winner,train=train,validation=validation,
        final_test_used_for_selection=False))
    print('Selected before test:',c.name,flush=True)
    baseline=Config(baseline=True)
    runs={}
    for name,conf,period,slip,bench in [
        ('test',c,spec['test'],.001,False),('full',c,[spec['start'],spec['end']],.001,False),
        ('benchmark_test',c,spec['test'],.001,True),('benchmark_full',c,[spec['start'],spec['end']],.001,True),
        ('baseline_test',baseline,spec['test'],.001,False),('baseline_full',baseline,[spec['start'],spec['end']],.001,False),
        ('stress_test',c,spec['test'],.003,False),('stress_full',c,[spec['start'],spec['end']],.003,False)]:
        run=simulate(f,None if bench else sig(conf),conf,*period,slippage=slip,benchmark=bench)
        export_case(folder,name,run);runs[name]=run
        print(name,run['summary'],flush=True)
    report=dict(experiment=spec['experiment'],status='completed_daily_research',live_qualified=False,
        original_formula_replicated=False,start=spec['start'],end=spec['end'],winner=c.name,
        winner_config=asdict(c),winner_weights=list(__import__('skills.course_bundle',fromlist=['WEIGHTS']).WEIGHTS[c.weights]),
        cases={k:r['summary'] for k,r in runs.items()},yearly=yearly(runs['full']['curve']),
        benchmark_yearly=yearly(runs['benchmark_full']['curve']),data_audit=f['audit'],
        family_counts=sig(c)['family_counts'],source_sha256=refs,elapsed_seconds=time.monotonic()-began,
        selection_final_test_used=False,prior_project_period_exposure=True,
        positive_train_count=sum(r['summary']['total_return']>0 for r in train),
        positive_validation_count=sum(r['summary']['total_return']>0 for r in validation),
        limitations_zh=[
          '完整封存行情截至 2026-10-22；本次未把尚未整理核對的後續交易日接入。',
          '歷史股票母體、完整下市股票及逐日身分尚未全面認證，存在存活者偏誤風險。',
          '法人資料為目前保存的歷史值，未驗證每筆首次發布／修訂時間；缺值不補零。',
          '成交金額採實際 Trading_money；法人買超金額仍為淨買股數乘收盤價估算。',
          '還原價含公司行動效果，模型隱含再投入；不模擬股利入帳日、配股交付或真實整數股數。',
          '每日 OHLC 只提供成交估算；單一價位日保守不成交，仍不能證明排隊、停損成交或容量。',
          '當日價量完整性與單一價位限制是日線模型的成交假設，依賴事後可得的完整日 K；不等於開盤時可執行的即時委託紀錄。',
          '僅官方證據確認的整日停牌，以前次有效價格保留估值；停牌期間不成交，未知缺價仍中止。',
          '歷史停牌區間與實際有交易行情衝突的座標列為不可用；若持倉遇到仍中止，不自行改成可交易。',
          '大盤用 0050 的 MA5 與 MACD 代理，並非完整重現課程的加權／櫃買雙市場判讀。',
          '未納入當沖、放空及缺少歷史證據的新聞、題材、基本面主觀判斷；400 日新高為日線改編。',
          '24 組中的選定版本只是本次規格下的結果，無全域最佳或未來報酬保證；不自動替換原策略。'])
    save_json(folder/'trials.json',dict(train=train,validation=validation,selection=winner))
    render_report(folder,report,{k:runs[k]['curve'] for k in ('full','benchmark_full','baseline_full')})
    for name,expected in refs.items():
        if sha(ROOT/name)!=expected:raise ValueError('Source changed while running: '+name)
    save_json(folder/'report.json',report)
    save_json(folder/'outputs.json',{p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name!='outputs.json'})
    save_json(OUT/'latest.json',{'report':str((folder/'report.json').relative_to(ROOT)),
        'report_sha256':sha(folder/'report.json'),'status':report['status']})
    print('Complete:',folder,flush=True)


if __name__=='__main__':
    main()
