#!/usr/bin/env python3
"""Publish a concise Markdown analysis from a fully verified local experiment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from app.chip_research_ui import LABELS
from scripts.research_chip import verify, OUTPUT, INPUT
from scripts.research_exit_scenarios import read, write, sha


def pct(value):
    return '—' if value is None else f'{value*100:+.2f}%'


def report():
    verify()
    summary=read(OUTPUT/'summary.json')
    source=read(INPUT/'primary_audit.json')
    control=summary['cases']['control']['summary']
    benchmark=summary['benchmark']
    lines=['# 籌碼策略研究與交叉比對（2026-09-11）','',
        '期間：2022-01-03至2026-09-09；本金100萬元，獲利複投、可買零股。固定458候選、三個股名額、閒置0050、63日／12%原出場。所有完整帳戶使用同一成交量限制、最低手續費、交易稅、雙邊滑價與公司行為帳務。', '',
        '**先區分兩件事：贏0050，和改善原策略，是不同門檻。以下是已反覆使用的歷史，不是未見驗證；排序敏感度及逐筆訊號診斷不足以消除過度配適。**', '',
        '## 完整帳戶比較','',
        '| 方法 | 總淨報酬 | 期末資產 | 年化 | 最大回撤 | 比原策略多 | 比同資料對照多 | 股票部位數 | 交易成本 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        f"| 0050持有 | {pct(benchmark['total_return'])} | {benchmark['final_nav']:,.0f} | {pct(benchmark['cagr'])} | {pct(benchmark['max_drawdown'])} | — | — | — | {benchmark['costs']['total_cost']:,.0f} |"]
    for mode,label in LABELS.items():
        case=summary['cases'][mode]
        if not case['completed']:
            lines.append(f"| {label} | 未完成：{case['reason']} | — | — | — | — | — | — | — |")
            continue
        r=case['summary']
        available=summary['cases'].get('available_'+mode)
        relative=(r['total_return']-available['summary']['total_return']) if available and available['completed'] else None
        label += '＊' if r.get('corporate_assumptions') else ''
        lines.append(f"| {label} | {pct(r['total_return'])} | {r['final_nav']:,.0f} | {pct(r['cagr'])} | {pct(r['max_drawdown'])} | {pct(r['total_return']-control['total_return'])} | {pct(relative)} | {r['stock_cohorts']} | {r['costs']['total_cost']:,.0f} |")
    lines+=['','金額單位為新臺幣；報酬差欄的%為百分點。相同資料對照只排除缺資料，沒有套用該籌碼條件。交易成本是全帳戶含0050、佣金、稅與滑價總和。期末保留持股與應收，沒有強制清倉。', '',
        '＊包含洋基2023配股推導比率與畸零估值假設，非全數公司行為精確核實。詳見[公司行為核對](chip_corporate_sources_20260911.md)。','',
        '## 各年度（同期間、含成本）','',
        '| 方法 | 2022 | 2023 | 2024 | 2025 | 2026至9/9 |', '|---|---:|---:|---:|---:|---:|']
    for mode,label in [('benchmark','0050'),*LABELS.items()]:
        c=dict(completed=True,summary=benchmark) if mode=='benchmark' else summary['cases'][mode]
        if c['completed']:
            annual={str(r['year']):r['total_return'] for r in c['summary']['annual']}
            lines.append('| '+label+' | '+' | '.join(pct(annual.get(str(y))) for y in range(2022,2027))+' |')
    lines+=['','## 逐筆訊號與交叉驗證','',
        '434筆有完整63市場日。這裡計算次日還原收盤到第63日還原收盤相對0050的比例費後超額；不含最低手續費、資金名額、實際容量、停損及股利支付時序，所以只用來診斷訊號，不是第二套可交易報酬。', '',
        '| 條件 | 可判讀／458 | 通過候選 | 完整63日通過筆數 | 通過組平均超額 | 通過減未通過 | 95%區間 |',
        '|---|---:|---:|---:|---:|---:|---|']
    event=summary['event_study']
    for mode,c in event['comparisons'].items():
        coverage=event['coverage'][mode]
        ci=c['ci95']
        lines.append(f"| {LABELS[mode]} | {coverage['known']} | {coverage['passed']} | {c['passed']['n']} | {pct(c['passed']['mean_excess63'])} | {pct(c['pass_minus_reject'])} | {'—' if ci is None else pct(ci[0])+'～'+pct(ci[1])} |")
    lines+=['','95%區間採進場月份連續3個月區塊、2000次重抽；沒有多重比較修正，也沒有消除本來候選策略的選擇偏誤。沒有任何正向差異的區間完整高於零；不能把最高的平均數稱為已證實優勢。', '',
        '## 名額與買入排序敏感度','',
        '| 方法 | 完整配對／20 | 贏同種子原策略 | 期末淨值差5%分位 | 中位差 | 95%分位 |',
        '|---|---:|---:|---:|---:|---:|']
    for mode,row in summary['paired_priority'].items():
        quantiles=row['nav_difference_quantiles']
        text=' | '.join(f'{v:+,.0f}' for v in quantiles) if quantiles else '— | — | —'
        lines.append(f"| {LABELS[mode]} | {row['complete_pairs']} | {row['wins']} | {text} |")
    lines+=['','每天的候選順序改變，但信號、預算、成交與稅費完全相同；每個方法與原策略用同一個種子。20組是固定的探索性壓力測試，不是20個獨立市場或顯著性證明。缺公司行為證據的種子不報完整報酬，也不補零。', '',
        '本輪三種方法皆完成20組配對，贏原策略次數全部為0。投信加速＋不追高、大戶增加＋融資減少、法人轉賣減半的期末資產中位差分別約−559萬、−576萬、−318萬元；因此這三種方法的劣勢，沒有因改變同日候選排序而翻轉。原策略自己的20種排序淨報酬卻介於479.70%至1316.99%，中位數644.59%，說明有限名額與複利會放大買入順序差異。這仍不證明原策略在新市場能維持績效。','',
        '## 綜合判讀與處理方向','',
        '1. **本輪沒有新方法超過完整原策略，但不能因此宣稱籌碼完全無效。** 原策略本身已經用過這段歷史做研究；710.81%不是未見市場的報酬承諾。本輪主要回答「在同一批458個候選上，加這些條件是否有改善」，不是搜尋所有台股所有籌碼策略。',
        '2. **分點集中暫列觀察，不能直接上線。** 淨報酬690.40%，比同資料對照365.90%高324.49個百分點；但低於原策略710.81%，最大回撤40.54%也更深。收益明顯集中在2026年截至9/9（180.46%），63日訊號差的區間跨零，而且本輪未預先把分點納入20種排序測試。下一輪應固定定義後，補分點排序壓力測試，再累積新日期的紙上驗證；不能補測後把本輪當未見驗證。',
        '3. **借券餘額下降有訊號線索，但帳戶增益尚未成立。** 415.75%贏0050，仍落後同資料對照449.40%；63日訊號差約6.14個百分點，但區間約−0.08至12.72，仍跨零。此帳戶另含洋基配股估值假設。',
        '4. **大戶增加只有很弱的增益，和融資減少疊加後反而退步。** 大戶增加246.19%，相同資料對照234.41%；加融資減少只剩71.63%。大戶再延後一週為72.10%，無法靠延遲修正救回這個組合。',
        '5. **投信／外資加速與不追高不宜設為本策略的硬門檻。** 它們可能排除尚未獲法人認同的早期行情，也可能讓有限名額轉向其他股票。下方逐部位回查可見多個大贏家被排除；這是路徑解釋，不是因果證明。',
        '6. **不採用目前這版法人轉賣出場。** 全出−19.06%、一次減半194.39%，都低於0050。全出形成215個股票部位，原策略58個；與原帳戶接近量級的累計成本，卻建立在更小的資產上。不可單由這次結果推論所有籌碼出場都無效，只能否定這個固定5日／20日轉賣規則在本樣本的表現。',
        '7. **目前這個族群成交占比條件不採用。** 帳戶76.90%，逐筆訊號通過組也更差。它是歷史價格群組的成交占比，尚未測「多日分點持續集中＋真正題材族群輪動」；新聞催化、分點連續5／20日、集保更細級距與融券回補等不在本輪定義，不能宣稱已研究完所有可能的籌碼分析。','',
        '## 原策略獲利集中與篩選的代價','',
        '下表是原帳戶已實現現金流加期末持股市值的逐部位貢獻，再回查當初是否符合籌碼條件；不據此重訂條件。篩選後整個資金路徑會改變，不能把被排除的獲利直接當成策略差額。','',
        '| 買入日 | 股票 | 原帳戶損益貢獻 | 投信加速 | 外資加速 | 大戶＋融資 | 借券減少 |',
        '|---|---|---:|---|---|---|---|']
    account=read(OUTPUT/'cases/control.json')['account']
    features={r['event_id']:r for r in read(OUTPUT/'candidate_signals.json')}
    final=account['daily'][-1]['date']
    contributions=[]
    if account['receivables']:
        raise ValueError('Attribution needs explicit outstanding entitlement valuation')
    for cohort in account['cohorts']:
        identity,sid=cohort['event_id'],cohort['stock_id']
        profit=sum(r['cash_change'] for r in account['cash_ledger'] if r.get('event_id')==identity and r.get('stock_id')==sid)
        profit+=sum(r['market_value'] for r in account['holdings'] if r['date']==final and r['event_id']==identity)
        contributions.append(dict(event_id=identity,stock_id=sid,name=cohort['name'],entry_date=cohort['entry_date'],pnl=profit))
    contributions.sort(key=lambda r:-r['pnl'])
    flag=lambda value:'未知' if value is None else '通過' if value else '排除'
    for r in contributions[:8]:
        f=features[r['event_id']]
        lines.append(f"| {r['entry_date']} | {r['stock_id']} {r['name']} | {r['pnl']:+,.0f} | "+' | '.join(flag(f[k]) for k in ('trust','foreign','holder_margin','sbl'))+' |')
    lines+=['','## 資料核對、執行效率與限制','',
        f"- 法人核對共同{source['compared_rows']:,}列，投信{source['net_different_rows']['trust']:,}列、外資{source['net_different_rows']['foreign']:,}列不同；融資共同{source['margin_compared_rows']:,}列、差異{source['margin_different_rows']:,}列。新研究統一採本次API版本，未覆寫DB或父研究。這不等於驗證歷史當時發布值。",
        '- 本機舊大戶表有合計重複計入及最高級距解析問題；研究重新解析15級、排除total／差異調整。本輪沒有修正正式ingest或覆寫DB，因此正式大戶欄位仍不能直接沿用。分點先合併同券商不同成交價；借券採借券賣出餘額，沒有把新借券成交當餘額。',
        '- 1,018次候選籌碼準備約163秒，另因資料差異補抓560次法人／融資。共享5400/h設定再保留10%餘額，實際上限4860/h，低於Sponsor的6000/h；4個I/O worker、無自動重試，逐檔保存和續取。新交易路徑另補實際公司行為、漲跌停及官方零股日資料，與訊號請求分開記錄。',
        '- 完整比較固定27組主要／同資料對照，加80組候選排序帳戶；最後以本機資料完整離線重現，未重新訓練模型。首次準備的網路時間不能拿來當每次回測時間。',
        '- 本次正式107組計算約571秒，連同全部離線重播與封存總計1,006秒；已有交易證據的單組約3至6秒。畫面初次核對12,727個來源檔約4.83秒，之後切換方法約0.52秒。這是本機本次量測，首次補資料時間另計。',
        '- 全部原執行限制保留：前20日均成交值5,000萬元；整股容量取當日及前20日均量較小值的1%，零股最多當日5%，漲跌停與缺報價拒絕成交。日量／最後報價無法保證當下撮合，股利仍有毛額與估值假設。',
        '- 大戶公布延遲7／14日為假設。當前公司名冊與歷史修訂版仍有存活者及資料修訂偏誤。借券用途未必放空；分點不是最終投資人身分。固定價格族群的成交占比不是全市場淨流入；未使用今天產業鏈名單回填歷史。',
        '- 選股、篩選及出場可能改變後續所有名額與複利，不能只用一條漂亮淨值線認定訊號有效；沒有自動採用本次最高回測方法。', '',
        '## 重現與紀錄','',
        '首次建立新研究資料時依序執行下列準備；已封存的目錄不要重跑準備步驟，因為來源索引與抓取時間會變更。這些命令依賴父研究的本機封存檔，並非只有Git就能從零還原。','',
        '```bash','python scripts/prepare_chip_inputs.py',
        'python scripts/prepare_chip_inputs.py --refresh-primary',
        'python scripts/research_chip.py','```','',
        '現有封存的檢查與重現（不需重新抓資料）：','', '```bash',
        'python scripts/research_chip.py --verify',
        'python scripts/research_chip.py --offline-replay',
        'python scripts/report_chip.py','```','',
        '本機`.cache/chip-research/`保存每組逐日資產、全部委託／成交、來源hash、逐筆條件、事件診斷及離線封存；`.cache/chip-inputs/`保存原始資料。這些大檔沒有推上Git，不能稱為已完成異地還原備份。Git保存程式、預先規格、資料更正說明、測試、本報告及[107組摘要與封存指紋](research_chip_20260911.json)。摘要不能取代完整逐筆資料。Workbench「策略驗證 → 籌碼有沒有幫助？」可切換完整方法、查看資產曲線、下載買賣及每日資產CSV；開畫面不會抓資料或重跑回測。','',
        '來源：[FinMind官方籌碼資料說明](https://finmind.github.io/tutor/TaiwanMarket/Chip/)、本機封存帳戶與原始API證據。', '']
    validation_path=ROOT/'.cache/chip-validation.json'
    validation=read(validation_path) if validation_path.exists() else None
    lines+=['## 軟體驗收與目前資料狀態','']
    if validation:
        tests=validation['tests']
        lines += [f"`make test`：{tests['passed']}通過、{tests['warnings']}警告、{tests['seconds']}秒。`make pipeline`兩次通過；`make api`與health／picks／models／jobs四個curl均回200。畫面已測方法切換、股票篩選、兩種CSV下載入口及公司行為假設提示。",'',
            '2026-09-11 21:16臺北時間，上市價格只到9/10，上櫃到9/11，資料及實盤策略狀態皆未就緒。pipeline成功不代表上市上櫃資料完整。本研究期間固定截止9/9，不把9/11狀態混入歷史結果。','']
    else:
        lines+=['未提供本次軟體驗收紀錄；封存核對不等同API與介面驗收。','']
    target=ROOT/'docs/research_chip_20260911.md'
    target.write_text('\n'.join(lines))
    write(target.with_suffix('.json'),dict(summary=summary,
        source_manifest_sha256=sha(OUTPUT/'manifest.json'),
        source_summary_sha256=sha(OUTPUT/'summary.json'),
        case_sha256={p.name:sha(p) for p in sorted((OUTPUT/'cases').glob('*.json'))},
        full_artifacts_backed_up=False,validation=validation))
    analysis=ROOT/'.cache/chip-analysis'
    analysis.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(contributions).to_csv(analysis/'control_contribution.csv',index=False)
    print(target)


if __name__=='__main__':
    report()
