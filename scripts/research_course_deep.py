#!/usr/bin/env python3
"""Execute the frozen second-stage course study and publish complete evidence."""
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.prepare_course_bundle import sha, save_json
from skills.course_bundle import Config, prepare_features, signals, simulate, summarize
from skills.course_deep_research import (
    clean_signal_close, intervene, shift_institutional_availability, select_training,
    trade_attribution, overlap_diagnostics, forward_signal_study, paired_block_interval,
)
from skills.strict_tick_inputs import full_session_halts
from skills.trial_registry import append_trial_registry, registry_trial_count

OUT = ROOT/'.cache/course-deep-research-20261028'
SPEC = ROOT/'docs/prereg_course_deep_20261028.json'
PARENT = ROOT/'artifacts/course_bundle_20261028'
INPUT = ROOT/'.cache/course-bundle-20261028'
LABELS = {
    'reference':'第一版組合', 'exit_ma10_confirm2':'MA10 連續兩日出場',
    'exit_ma20':'MA20 出場', 'exit_ma20_confirm2':'MA20 連續兩日出場',
    'exit_ma50':'MA50 出場', 'market_ma60':'加大盤 MA60 濾網',
    'market_ma200':'加大盤 MA200 濾網', 'market_ma200_exit_ma20':'大盤 MA200 + 個股 MA20',
    'first_signal_10':'十日首個進場訊號', 'rank_momentum':'只按動能排序',
    'rank_membership_count':'先按符合策略數排序', 'momentum_60':'動能窗口改 60 日',
    'institutions_delayed1':'法人資料延後一日', 'family_volume_momentum':'只用量大動能',
    'family_trust_near_breakout':'只用投信剛買準突破', 'family_trust_consecutive':'只用投信連買',
    'family_foreign_first':'只用外資剛買', 'family_foreign_trust':'只用外資投信同買',
    'family_price_volume_400':'只用價量 400 日新高', 'family_near_breakout':'只用突破邊緣',
    'cash':'現金',
}


def export_run(folder, name, run):
    run['curve'].to_csv(folder/(name+'-nav.csv'), index=False)
    pd.DataFrame(run['trades']).to_csv(folder/(name+'-trades.csv'), index=False)
    save_json(folder/(name+'.json'), {k:v for k,v in run.items() if k not in ('curve','trades','decisions')})


def render(folder, report, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios':[3,1]})
    for key, label in [('reference','Original bundle'),('market_ma200_exit_ma20','Market MA200 + exit MA20'),
                       ('rank_membership_count','More conditions first'),('benchmark','0050 buy and hold')]:
        if key not in curves:
            continue
        curve = curves[key].set_index('date').equity
        axes[0].plot(curve.index, curve/curve.iloc[0], label=label, lw=1.4)
        axes[1].plot(curve.index, (curve/curve.cummax()-1)*100, lw=1)
    axes[0].legend(); axes[0].set_ylabel('Net NAV / initial capital')
    axes[1].set_ylabel('Drawdown %')
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('Course strategy diagnosis | previously seen historical data')
    fig.tight_layout(); fig.savefig(folder/'comparison.png', dpi=160); plt.close(fig)
    pct = lambda v: '—' if v is None else f'{v*100:.2f}%'
    rows = []
    for row in report['full_period']:
        title = escape(LABELS[row['arm']])
        if row['status'] != 'completed':
            rows.append(f'<tr><td>{title}</td><td colspan="6">中止：{escape(row["error"])}</td></tr>')
            continue
        s = row['summary']; ci = row['paired_difference_interval']
        rows.append(f'<tr><td>{title}</td><td>{pct(s["total_return"])}</td><td>{pct(s["cagr"])}</td><td>{pct(s["max_drawdown"])}</td><td>{s["completed_trades"]}</td><td>{s["annual_one_way_turnover"]:.1f}</td><td>{pct(ci["low"])}～{pct(ci["high"])}</td></tr>')
    overlap = ''.join(f'<tr><td>{escape(LABELS["family_"+r["family"]])}</td><td>{r["eligible_stock_days"]}</td><td>{r["unique_stock_days"]}</td></tr>' for r in report['overlap']['families'])
    forward = ''.join(f'<tr><td>{r["horizon"]}</td><td>{r["group"]}</td><td>{r["stock_days"]}</td><td>{pct(r["mean_net_return"])}</td><td>{pct(r["positive_fraction"])}</td><td>{pct(r["equal_date_mean_excess"])}</td></tr>' for r in report['forward_signals'])
    walk = ''.join(f'<tr><td>{r["year"]}</td><td>{escape(LABELS.get(r["selection"],r["selection"]))}</td><td>{pct(r.get("summary",{}).get("total_return"))}</td><td>{pct(r.get("summary",{}).get("max_drawdown"))}</td><td>{r["status"]}</td></tr>' for r in report['walk_forward'])
    costs = report['attribution']; notes=''.join('<li>'+escape(x)+'</li>' for x in report['limitations_zh'])
    trades_by_exit = ''.join(f'<tr><td>{escape(r["reason"])}</td><td>{r["trades"]}</td><td>{r["median_holding_sessions"]:.1f}</td><td>{r["net_pnl"]:,.0f}</td></tr>' for r in costs['by_exit'])
    stresses = ''.join(f'<li>{escape(LABELS[r["arm"]])}：'+(f'累積 {pct(r["summary"]["total_return"])}, 最大回撤 {pct(r["summary"]["max_drawdown"])}' if r['status']=='completed' else escape(r['error']))+'</li>' for r in report['stress'])
    content=f'''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>動能課程深入研究</title>
<style>body{{font:17px/1.65 system-ui;color:#193747;background:#f4f7f9;max-width:1200px;margin:40px auto;padding:0 22px}}h1,h2{{color:#124b60}}table{{border-collapse:collapse;width:100%;background:white;margin:18px 0;font-size:15px}}td,th{{border-bottom:1px solid #dbe6eb;padding:9px;text-align:right}}td:first-child,th:first-child{{text-align:left}}.callout{{background:#fff0c8;border-left:5px solid #bd8714;padding:18px}}img{{width:100%}}a{{color:#087c99}}.scroll{{overflow-x:auto}}</style>
<h1>愛德恩動能課程｜第二輪深入研究</h1><p>2021-10-25～2026-10-22；日線多方、成本後結果。20 組事前列明的機制比較；所有歷史期間已在先前研究看過。</p>
<div class="callout">這是診斷與探索，不是作者公式重現，也不是全新樣本外驗證。第一版逐筆交易與總結已原樣重現；本輪沒有修改原始回測引擎、行情或實盤策略。</div>
<p>資料修正：威剛 3260 於 2025-10-01 正式停牌、10-02 恢復，依 <a href="https://www.tpex.org.tw/storage/eb_data/11409/11402023641.html">櫃買公告</a>及期交所復牌公文補上當天持倉估值，不成交、不補進場訊號。修正前 103 次評估（8 次中止）完整保留；本輪規則未改。原始研究報告與證據雜湊記於 data_amendment。</p>
<h2>第一版為何虧損</h2><p>固定原來交易單位和進出時點，價差毛損益合計 <strong>{costs['same_units_gross_pnl']:,.0f} 元</strong>；手續費與稅 {costs['fee_tax']:,.0f} 元，滑價 {costs['slippage']:,.0f} 元；最後淨損益 <strong>{costs['net_pnl']:,.0f} 元</strong>。這是精確成本歸因，不能把毛損益當成另一個零成本複利回測。</p>
<p>持有期間中位數 {costs['median_holding_sessions']:.1f} 個交易日。不同出場原因：</p><table><tr><th>出場原因</th><th>筆數</th><th>持有日中位數</th><th>淨損益</th></tr>{trades_by_exit}</table>
<h2>條件數量不等於獨立證據</h2><p>以下為股票×訊號日，並非實際成交數。最後一欄是只符合該篩選器、移除它就會失去的訊號數。若為 0，代表它在這套聯集中沒有新增候選；不代表它沒有排序或研究價值。突破進場已要求放量，所以第一版量能 10 分對所有合格候選都相同。</p>
<table><tr><th>篩選器</th><th>合格訊號數</th><th>獨有訊號數</th></tr>{overlap}</table>
<h2>20 組完整比較</h2><p>固定 5 檔、7% 即時停損、隔日開盤估算。兩日確認只影響均線出場，不延後固定停損。大盤均線濾網只控制新買進，沒有事後強制出清。手續費每邊 0.1425%、股票賣出稅 0.3%、滑價每邊 0.1%；100 萬是統一試算本金。</p>
<div class="scroll"><table><tr><th>變化</th><th>累積報酬</th><th>年化</th><th>最大回撤</th><th>平倉筆數</th><th>年周轉</th><th>相對第一版年化日均差 95% 區間</th></tr>{''.join(rows)}</table></div>
<p>最後一欄由配對日報酬差的 20 日區塊重抽樣得到，並非 CAGR 差，也未做多重比較校正，不能當成獲利保證或正式顯著性結論。</p><img src="comparison.png" alt="Net value and drawdown comparisons">
<h2>較高滑價</h2><p>事前指定第一版與「大盤 MA200 + 個股 MA20」，每邊滑價改 0.3%，不重選參數：</p><ul>{stresses}</ul>
<h2>逐年只看過去再選擇</h2><p>每年以之前所有資料比較 19 組（法人延遲屬敏感度測試，不參與選擇），先要求正年化且至少 20 筆平倉，再以年化 − 0.5×回撤絕對值排序；沒有合格者就留現金。每年獨立本金起算、年末扣估計清算成本。這是已看過資料的時間順序重播。</p>
<table><tr><th>年度</th><th>前一年底選定</th><th>當年報酬</th><th>當年最大回撤</th><th>狀態</th></tr>{walk}</table>
<p>串接後研究淨值結果：{escape(str(report['walk_chain_summary']))}。若任一年遇未知持倉缺價而中止，整段不提供完整績效，亦不偷偷換成別的策略。</p>
<h2>符合更多條件，後續是否更容易漲</h2><p>每個收盤訊號從隔日開盤持有 5 或 20 個交易日至收盤；扣費、排除 ±3% 外開盤與缺資料／不可交易端點。未使用原策略途中停損或持倉名額，屬訊號診斷，不能與上面的帳戶報酬混用。positive_fraction 是這份歷史樣本的上漲比例，沒有校準成未來機率。相對 0050 欄按訊號日等權，其他報酬欄按股票日等權。</p>
<div class="scroll"><table><tr><th>持有日</th><th>組別</th><th>樣本</th><th>平均淨報酬</th><th>歷史正報酬比例</th><th>相對 0050 平均報酬</th></tr>{forward}</table></div>
<h2>限制與下一步</h2><ul>{notes}</ul><p>更完整的原課程轉譯，仍需處理箱型品質、初次發動、漲停家數、公司事件、產業與基本面。這些不是目前可從動能分數自動推導的資訊；不能因新結果較好就宣稱已重現作者。</p>
<h2>來源與查核</h2><p><a href="report.json">完整結果與來源雜湊</a> · <a href="spec.json">事前固定研究範圍</a> · <a href="trials.jsonl">每次成功／中止試驗</a> · <a href="trade-attribution.csv">逐筆成本歸因</a> · <a href="overlap.csv">條件交集</a> · <a href="forward-signals.csv">後續報酬診斷</a>。所有完成案例附 *-nav.csv 與 *-trades.csv。</p>
<p>研究方法參考 Bailey 等人的 <a href="https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf">The Probability of Backtest Overfitting</a>：反覆挑參數可能得到歷史假象，因此公開搜尋範圍及全部結果。本報告沒有計算 PBO，也不宣稱區塊重抽樣已消除選擇偏誤。</p></html>'''
    (folder/'index.html').write_text(content)


def main():
    began = time.monotonic()
    spec = json.loads(SPEC.read_text())
    freeze = json.loads((OUT/'plan-freeze.json').read_text())
    if freeze['spec_sha256'] != sha(SPEC):
        raise ValueError('Second-stage plan changed after freeze')
    parent = json.loads((PARENT/'report.json').read_text())
    refs = dict(parent['source_sha256'])
    refs[str((PARENT/'report.json').relative_to(ROOT))] = sha(PARENT/'report.json')
    for name in ('docs/prereg_course_deep_20261028.json','skills/course_deep_research.py',
                 'scripts/research_course_deep.py','tests/test_course_deep_research.py',
                 'docs/course_deep_data_amendments_20261028.json'):
        refs[name] = sha(ROOT/name)
    amendment=json.loads((ROOT/'docs/course_deep_data_amendments_20261028.json').read_text())
    for source in amendment['sources']:
        refs[source['path']]=source['sha256']
    refs[amendment['before_repair_report']]=amendment['before_repair_sha256']
    for name, expected in refs.items():
        if sha(ROOT/name) != expected:
            raise ValueError('Parent input/code changed: '+name)
    digest = hashlib.sha256(json.dumps(refs, sort_keys=True).encode()).hexdigest()[:16]
    folder = OUT/('run-'+digest)
    if (folder/'report.json').exists():
        for name, expected in json.loads((folder/'outputs.json').read_text()).items():
            if sha(folder/name) != expected: raise ValueError('Completed output changed: '+name)
        print('Existing result verified:',folder); return
    if (folder/'trials.jsonl').exists():
        raise ValueError('Partial run retained; inspect it before explicitly restarting')
    folder.mkdir(exist_ok=True)
    (folder/'spec.json').write_bytes(SPEC.read_bytes())
    identity = json.loads((INPUT.parent/'scanner-20261022-close/inputs-v1/identity.json').read_text())
    halts = [r for r in full_session_halts(identity) if r['end']>spec['period'][0] and r['start']<=spec['period'][1]]
    halts += json.loads((ROOT/'docs/course_bundle_halt_evidence_20261028.json').read_text())
    halts.append(amendment['halt'])
    bars = pd.read_parquet(INPUT/'bars.parquet'); inst = pd.read_parquet(INPUT/'institutional.parquet')
    money = pd.read_parquet(INPUT/'turnover.parquet')
    print('Preparing frozen features',flush=True)
    f = prepare_features(bars, inst, money, halts=halts); clean = clean_signal_close(bars)
    delayed = prepare_features(bars, shift_institutional_availability(inst, f['days']), money, halts=halts)
    config = Config(**spec['base_config'])
    evaluations, full, curves, runs = [], [], {}, {}
    global_start_count = registry_trial_count()

    def evaluate(arm, start, end, stage, *, slippage=.001, benchmark=False, export=False):
        try:
            base = delayed if arm == 'institutions_delayed1' else f
            feat, sig = (f, None) if benchmark else intervene(base, config, arm, clean)
            run = simulate(feat, sig, config, start, end, slippage=slippage, benchmark=benchmark)
            row = dict(arm=arm, stage=stage, start=start, end=end, slippage=slippage,
                status='completed', summary=run['summary'])
            if export: export_run(folder, stage+'-'+arm, run)
        except ValueError as exc:
            row = dict(arm=arm, stage=stage, start=start, end=end, slippage=slippage,
                status='blocked', error=str(exc)); run = None
        record = dict(timestamp=datetime.now(timezone.utc).isoformat(), source='course_deep_research',
            experiment=spec['experiment'], run_digest=digest, evaluation_index=len(evaluations),
            command='python scripts/research_course_deep.py', **row)
        append_trial_registry(record, folder/'trials.jsonl')
        append_trial_registry(record)
        evaluations.append(row)
        print(stage,arm,row['status'],round(row['summary']['total_return'],4) if run else row['error'],flush=True)
        return row, run

    for arm in spec['arms']:
        row, run = evaluate(arm, *spec['period'], 'full', export=True)
        full.append(row)
        if run: curves[arm] = run['curve']; runs[arm] = run
    if 'reference' not in runs:
        raise ValueError('Parent reference replay failed')
    original = json.loads((PARENT/'full.json').read_text())
    if runs['reference']['summary'] != original['summary'] or runs['reference']['trades'] != original['trades']:
        raise ValueError('Reference no longer exactly reproduces the prior research')
    for row in full:
        if row['status']=='completed':
            row['paired_difference_interval'] = paired_block_interval(curves[row['arm']], curves['reference'])
    benchmark_row, benchmark = evaluate('benchmark', *spec['period'], 'full', benchmark=True, export=True)
    if benchmark: curves['benchmark'] = benchmark['curve']
    sig = signals(f, config)
    attribution = trade_attribution(runs['reference']['trades'], f['days'], f['close']['0050'])
    attribution.to_csv(folder/'trade-attribution.csv', index=False)
    by_exit = attribution.groupby('reason').agg(trades=('net_pnl','size'),net_pnl=('net_pnl','sum'),
        median_holding_sessions=('holding_sessions','median')).reset_index().to_dict('records')
    by_regime = attribution.groupby('entry_market_regime').agg(trades=('net_pnl','size'),net_pnl=('net_pnl','sum'),
        mean_trade_return=('net_return','mean')).reset_index().to_dict('records')
    families, pairs, overlap_meta = overlap_diagnostics(sig,*spec['period'])
    pairs.to_csv(folder/'overlap.csv'); pd.DataFrame(families).to_csv(folder/'family-unique.csv', index=False)
    forward, censor = forward_signal_study(f,sig,*spec['period'])
    pd.DataFrame(forward).to_csv(folder/'forward-signals.csv',index=False)
    stress=[]
    for arm in ('reference','market_ma200_exit_ma20'):
        row, _ = evaluate(arm,*spec['period'],'stress',slippage=.003,export=True);stress.append(row)
    walk=[]; chain=[]; chain_nav=1_000_000.
    for year in spec['walk_forward']['years']:
        days = f['days']; past = days[days.year < year]; current = days[(days.year==year)&(days<=pd.Timestamp(spec['period'][1]))]
        train_end, start, end = (str(x.date()) for x in (past[-1],current[0],current[-1]))
        training=[]
        for arm in spec['arms']:
            if arm=='institutions_delayed1': continue
            row,_=evaluate(arm,spec['period'][0],train_end,'train'+str(year)); training.append(row)
        selected=select_training(training)
        save_json(folder/f'selection-{year}-before-replay.json',dict(training=training,selected=selected or 'cash',future_used_by_selector=False))
        if selected is None:
            curve=pd.DataFrame(dict(date=days[(days>=past[-1])&(days<=current[-1])],equity=1_000_000.,positions=0))
            row=dict(status='completed',summary={'total_return':0.,'max_drawdown':0.},arm='cash')
            curve.to_csv(folder/f'forward{year}-cash-nav.csv',index=False)
        else:
            row,run=evaluate(selected,start,end,'forward'+str(year),export=True)
            curve=run['curve'] if run else None
        walk.append(dict(year=year,selection=selected or 'cash',status=row['status'],
            **({'summary':row['summary']} if row['status']=='completed' else {'error':row['error']})))
        if curve is not None:
            linked=curve[['date','equity']].copy();linked['equity']*=chain_nav/1_000_000.
            if chain: linked=linked.iloc[1:]
            chain_nav=float(linked.equity.iloc[-1]);chain.append(linked)
    walk_benchmark_row, _ = evaluate('benchmark',str(f['days'][f['days'].year==2023][0].date()),spec['period'][1], 'walk',benchmark=True,export=True)
    if all(r['status']=='completed' for r in walk):
        linked=pd.concat(chain,ignore_index=True);linked.to_csv(folder/'walk-chain-nav.csv',index=False)
        nav=linked.equity.to_numpy();walk_summary=dict(total_return=float(nav[-1]/nav[0]-1),max_drawdown=float((nav/np.maximum.accumulate(nav)-1).min()))
    else:
        walk_summary={'status':'incomplete_due_to_blocked_year','total_return':None}
    report=dict(experiment=spec['experiment'],status='completed_exploratory_comparison',live_qualified=False,
        prior_period_exposure=True,reference_exactly_reproduced=True,source_sha256=refs,
        data_amendment=amendment,
        full_period=full,benchmark=benchmark_row,stress=stress,walk_forward=walk,walk_chain_summary=walk_summary,
        walk_benchmark=walk_benchmark_row,
        attribution=dict(**{k:float(attribution[k].sum()) for k in ('net_pnl','same_units_gross_pnl','fee_tax','slippage')},
            median_holding_sessions=float(attribution.holding_sessions.median()),by_exit=by_exit,by_entry_regime=by_regime,
            top_five_winning_trades_pnl=float(attribution.net_pnl.nlargest(5).sum())),
        overlap=dict(families=families,**overlap_meta),forward_signals=forward,forward_censoring=censor,
        evaluation_count=len(evaluations),blocked_count=sum(r['status']=='blocked' for r in evaluations),
        distinct_arms=len(spec['arms']),parent_distinct_configurations=24,
        global_registry_start=global_start_count,global_registry_end=registry_trial_count(),
        elapsed_seconds=time.monotonic()-began,
        limitations_zh=[
            '全部五年資料已在上一輪看過，本輪時間切割與重抽樣不會把它變成真正未見資料；未執行實盤或自動建立前瞻追蹤。',
            '歷史母體／下市完整性、法人首次發布時間與部分行情修訂尚未完全認證；法人延後一日只作敏感度檢查。',
            '使用還原單位淨值與日 K 成交模型，未處理真實整數股數、現金股利入帳、逐筆排隊及成交容量；完整日 K 可交易性檢查有事後觀察限制。',
            '慢速均線、長期大盤濾網與分數都是自訂研究假設。大盤 MA200 僅是 0050 代理，不是作者的完整大盤／櫃買／漲停家數判讀。',
            '只對已核證停牌延續估值；未知持倉缺價嚴格中止。中止案例不以部分淨值冒充完整五年，也不根據事後結果換股補洞。',
            '訊號報酬樣本重疊且受缺值、端點可交易性與跳空篩選影響；樣本上漲比例不是未來勝率，也不是扣除持倉限制後的帳戶績效。',
            '20 組比較與先前 24 組都增加選擇偏誤；區塊區間未經多重比較修正，沒有宣稱統計確認或全域最佳。'])
    render(folder,report,curves)
    for name,expected in refs.items():
        if sha(ROOT/name)!=expected:raise ValueError('Source changed during run: '+name)
    save_json(folder/'report.json',report)
    save_json(folder/'outputs.json',{p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name!='outputs.json'})
    save_json(OUT/'latest.json',{'report':str((folder/'report.json').relative_to(ROOT)),'report_sha256':sha(folder/'report.json')})
    print('Complete:',folder,flush=True)


if __name__=='__main__':
    main()
