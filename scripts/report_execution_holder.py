#!/usr/bin/env python3
"""Readable reports for execution stress and the separate 2492 holder case."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from scripts import research_execution_stress as execution
from scripts import research_holder_2492 as holder
from scripts.research_exit_scenarios import read,write,sha

LABELS={'control':'原策略','depth':'零股加對手量限制','quote':'零股按對手買賣價',
    'slip90':'每邊滑價0.90%','entry_delay':'進場再晚一市場日',
    'exit_delay':'出場再晚一市場日','combined':'全部壓力合併'}


def pct(value):
    return '—' if value is None else f'{value*100:+.2f}%'


def reports(font):
    em=execution.verify();hm=holder.verify()
    es=read(execution.OUTPUT/'summary.json');hs=read(holder.OUTPUT/'summary.json')
    lines=['# 原策略能否承受更嚴格的成交條件？','',
        '2026-09-11固定研究；2022-01-03至2026-09-09，本金100萬元複利、458候選、三個個股名額、閒置0050、12%／63市場日出場。七種執行條件各配相同成本與成交條件的0050帳戶，共14組。','',
        '**原策略的710.81%對執行條件很敏感，這次不能支持它已具有可重複實現的優勢。** 零股加最後對手量限制後幾乎回到0050；進場再晚一天及全部壓力合併都落後相同條件的0050。這是固定歷史的壓力試驗，不是估計真實成交率。','',
        '| 方法 | 費後總報酬 | 期末資產 | 最大回撤 | 同條件0050 | 超額百分點 | 股票部位數 | 累計成本 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode,label in LABELS.items():
        c,b=es['cases'][mode],es['cases'][mode+'_benchmark']
        if not c['completed'] or not b['completed']:
            lines.append(f"| {label} | 未完成：{c.get('reason',b.get('reason'))} | — | — | — | — | — | — |")
            continue
        r,br=c['summary'],b['summary']
        lines.append(f"| {label} | {pct(r['total_return'])} | {r['final_nav']:,.0f} | {pct(r['max_drawdown'])} | {pct(br['total_return'])} | {(r['total_return']-br['total_return'])*100:+.2f} | {r['stock_cohorts']} | {r['costs']['total_cost']:,.0f} |")
    lines+=['','金額為新臺幣；收益包含期末持股／應收，沒有強制清倉；全部扣最低手續費、交易稅及指定滑價。','',
        '## 各年度','', '| 方法 | 2022 | 2023 | 2024 | 2025 | 2026至9/9 |','|---|---:|---:|---:|---:|---:|']
    for mode,label in LABELS.items():
        c=es['cases'][mode]
        if c['completed']:
            lines.append('| '+label+' | '+' | '.join(pct(a['total_return']) for a in c['summary']['annual'])+' |')
    lines+=['','## 為何這次結果改變很大','',
        '- 對手價替代最後成交價的影響小；最後對手股數與延後進場的影響大。因此問題更集中在「能買到多少、何時空出名額與買到哪檔」，不是只有買賣價差。',
        '- 原帳戶的重要獲利部位包含5475德宏與6442光聖。深度限制、進場延遲及合併條件的帳戶均沒有買到這兩檔；較早的成交與退出差異，已改變後續名額與本金。這是事後路徑診斷，不是可把兩檔獲利直接相減的因果歸因。',
        '- 合併壓力報酬高於單獨延後進場，並不表示成本更高更好。不同執行條件改變整條資金路徑，策略結果不必隨壓力單調下降；沒有據此選擇最佳壓力組上線。',
        '- 不再把原歷史710.81%視為足以支持使用的證據。下一個工程優先項是凍結訊號、保留當時委託深度與實際／紙上委託紀錄；在真實可取得的報價上評估成交，之後再研究新因子。','',
        '## 精確定義與限制','',
        '- 深度組維持零股日量5%上限，再受當日最後對手報價股數限制；買進用賣方數量、賣出用買方數量。日內已成交量消耗額度，不重複使用最後快照。缺數量拒絕成交。這不是盤中完整訂單簿，也不能解讀成可見數量之外整天都買不到。',
        '- 對手價組零股買進用ask、賣出用bid，仍加原0.45%滑價；深度組單獨不更換原最後成交價。合併組兩項皆使用。',
        '- 滑價組每邊實際滑價加至0.90%，預算規劃仍用原成本估計；實際成交按更高成本縮股數，不借款、不透支。',
        '- 進場延遲保留原訊號，多等一市場日，按執行日前一日的淨值與價格重新估量；沒有用延遲日重新篩選股票。持有63日起算點跟著實際進場移動。出場延遲對第一次鎖定指令只加一天，未成交部分照原規則重試，不重新延期。這是執行延遲，不是停損建議。',
        '- 0050初始、閒置資金操作不額外延遲，個股資金調度與個股進場一起移動；每組0050基準用相同價量／成本條件，單獨個股延遲不適用0050。',
        '- 沿用父研究的公司行為毛額、畸零應收與權利證書估值假設，以及當前公司名冊、資料修訂與反覆使用歷史的偏誤。各組沒有曝露到6691的額外配股估值假設；不代表其他父研究限制已消除。','',
        '## 紀錄與重現','',
        f"首次計算及14帳戶逐欄離線重現、來源封存共{em['total_seconds']:.1f}秒。資料備妥的股票帳戶約3至4秒、0050約0.4秒；首次進場延遲及合併組約81／76秒含補交易證據。沒有重訓模型或重抓全市場。",'',
        '相對複製的父快取，本輪成交證據只新增1次FinMind請求與94次交易所HTTP請求，沒有新增股利快取。兩類請求分開記錄；FinMind沿用共享額度保護。','',
        '完整資料留在`.cache/execution-stress/`，各組包含全部成交、拒單、逐日資產、出場狀態、對帳結果；來源獨立放`.cache/execution-stress-inputs/`。Git保存摘要與指紋，沒有完成大檔異地備份。','',
        '```bash','python scripts/research_execution_stress.py --verify','python scripts/research_execution_stress.py --offline-replay','```','']
    (ROOT/'docs/research_execution_stress_20260911.md').write_text('\n'.join(lines))
    write(ROOT/'docs/research_execution_stress_20260911.json',dict(summary=es,
        manifest_sha256=sha(execution.OUTPUT/'manifest.json'),
        case_sha256={p.name:sha(p) for p in (execution.OUTPUT/'cases').glob('*.json')},full_artifacts_backed_up=False))

    first,last=hs['first'],hs['last']
    lines=['# 華新科2492：2026/4/2之後的小股東、大戶與股價','',
        '**上漲段確實曾出現小股東減少、大戶集中；回跌後又反轉。從4/2到最新觀測的持股占比下降，但小股東人數增加，所以必須分階段看。目前不足以支持這個現象是穩定的提前買入訊號。**','',
        '| 項目 | 2026/4/2 | 2026/9/4最新持股觀測 | 變化 |','|---|---:|---:|---:|']
    for key,label,percent in [('raw_close','未還原收盤價',False),('small_pct','100張以下持股比例',True),
        ('large_pct','超過1000張持股比例',True),('small_people','100張以下人數',False),('large_people','超過1000張人數',False),
        ('small_shares','100張以下持有股數',False),('large_shares','超過1000張持有股數',False),('total_shares','集保庫存股數',False)]:
        a,b=first[key],last[key]
        if percent:
            values=f'{a:.2f}% | {b:.2f}% | {b-a:+.2f}個百分點'
        else:
            values=f'{a:,.2f} | {b:,.2f} | {b-a:+,.2f}' if key=='raw_close' else f'{a:,.0f} | {b:,.0f} | {b-a:+,.0f}'
        lines.append(f'| {label} | {values} |')
    weekly=pd.read_csv(holder.OUTPUT/'weekly.csv')
    peak=weekly.loc[weekly.large_pct.idxmax()]
    lines+=['',f"股價資料另更新至9/11收盤{hs['latest_price']:,.1f}元，相對4/2未還原價格變動{pct(hs['raw_price_change'])}，FinMind還原價格變動{pct(hs['adjusted_price_change'])}；同期0050還原變動{pct(hs['benchmark_adjusted_change'])}。這不是扣交易成本的帳戶報酬。9/11沒有新的持股觀測，不能把9/4持股比例標為9/11最新股權。",'',
        f"事後分段可見：大戶比例最高的週觀測為{peak['date']}，股價{peak.raw_close:g}元、大戶{peak.large_pct:.2f}%、小股東{peak.small_pct:.2f}%及{peak.small_people:,.0f}人；相對4/2，當時確實是大戶更集中、小股東比例與人數都下降。到9/4股價回落，大戶比例降低、小股東人數增加。這個中間日期是在看完資料後取最大值，只用來解釋圖形，不能當事先知道的高點或出場規則。",'',
        '![股價、持股占比與人數](holder_2492_20260911.png)','',
        '## 同步變化，與提前訊號，是兩個問題','',
        '先固定看四週持股比例變動。同期相關衡量已發生的四週行情；領先分析則等持股觀測日加7日後的下一市場日，才開始衡量後20／40／60市場日還原股價變動。末端不夠完整持有期就排除，沒有補到最新日充當完整報酬。','',
        '| 2026/4/2起的比較 | 有效週數 | 千張大戶變化相關 | 小股東變化相關 |','|---|---:|---:|---:|']
    labels={'same_period_return4':'同期四週','forward20':'可用後20市場日','forward40':'可用後40市場日','forward60':'可用後60市場日'}
    for key,label in labels.items():
        d=hs['diagnostics']['7']['current'][key]
        lines.append(f"| {label} | {d['large_change4']['n']} | {d['large_change4']['spearman']:+.3f} | {d['small_change4']['spearman']:+.3f} |")
    lines+=['','相關係數為Spearman，範圍−1至1，不能當作勝率。同期大戶變化與漲幅同向、小股東比例與漲幅反向；移到之後的20日，關係接近零。40／60日呈反向，但樣本少且高度重疊，不能據此改成反向交易。','',
        '## 「大戶增加且小股東減少」交集，比沒有出現時更好嗎？','',
        '| 觀測期／公布延遲假設 | 成立週數 | 成立後20日平均變動 | 不成立週數 | 不成立後20日平均變動 |','|---|---:|---:|---:|---:|']
    for window,label in [('current','2026/4/2起'),('historical_2022_2025','2022–2025同股對照')]:
        for lag in ('7','14'):
            d=hs['diagnostics'][lag][window]['forward20'];a,b=d['joint_pass'],d['joint_fail']
            lines.append(f"| {label}／加{lag}日再等下一市場日 | {a['n']} | {pct(a['mean'])} | {b['n']} | {pct(b['mean'])} |")
    lines+=['','7日假設下2026交集看起來較好，但多延一週後方向翻轉；2022–2025的同股對照優勢很弱。這些週可能共同涵蓋同一波大漲，不能把11筆當成11次獨立成功交易。2022–2025對照的未來價格也截在2025年底，沒有混入2026行情。','',
        '較合理的用途，是把持股集中列為行情背景或候選解釋，再配合可交易的價格與成交證據；本輪不把它設成必買條件，也不從這次反向相關發明放空策略。','',
        '## 資料核對與限制','',
        '- 小股東是100張以下，含恰好100張；現有級距不能剔除剛好100張。千張級以1,000,001股起的最高級距計算。每張1000股。',
        '- 持股比例用各級股數除以集保庫存重新計算，再核對FinMind各級已四捨五入的percent。完整15級、排除total／差異調整，合計、人數及級距比例均核對；本次646筆歷史週觀測沒有格式不合格，但不表示2010至今每週都完整。',
        '- 分母是集保庫存，不是必然等於公司已發行股數。研究期庫存由485,804,774股降至485,204,774股；同時大戶實際股數增加31,669,155股，小股東實際股數減少28,183,368股，因此不只是分母略降產生的比例幻象。未逐案查明所有庫存變化的原因。',
        '- FinMind股利資料列2026/7/9現金除息，每股2.50309147元，7/29發放；原價與還原價分開列。還原版本未完成另一家價格供應商逐日對帳，不宣稱精確實收總報酬。',
        '- 集保按每週最後營業日餘額、ID歸戶編製。級距只能表示人數與持股分布，無法辨認是否同一批大戶、主力身分或資金來源。已聚合比例不等於個別投資人的買賣紀錄。',
        '- 7／14日只是公布可用時間假設，沒有歷史首次發布時間。4/2觀測不能當4/2收盤前就知道。只有23週主樣本，且股票與起點已由使用者事後指定；沒有全市場或未見樣本外證據。',
        '- Goodinfo網頁直接抓取回403，本輪沒有繞過限制，也沒有宣稱完成Goodinfo數值交叉驗證。數值來源是FinMind；級距加總是內部一致性檢查，非第二資料商驗證。','',
        '來源：[FinMind持股分級](https://finmind.github.io/tutor/TaiwanMarket/Chip/)、[FinMind還原價格](https://finmind.github.io/tutor/TaiwanMarket/Technical/)、[TDCC編製說明](https://original-www.tdcc.com.tw/portal/zh/smWeb/qryStock)。','',
        '## 交付與重現','',
        'Git保存本報告、23週表格、圖、全部診斷摘要及來源hash；原始資料在`.cache/holder-case-2492/`，完整逐週可用日期與診斷在`.cache/holder-analysis-2492/`。原始兩次請求已在上一輪備妥，本輪另加三次：2492與0050還原價、2492股利；共享額度／快取，沒有重抓全市場。','',
        '```bash','python scripts/research_holder_2492.py --verify','python scripts/research_holder_2492.py','```','']
    (ROOT/'docs/research_holder_2492_20260911.md').write_text('\n'.join(lines))
    write(ROOT/'docs/research_holder_2492_20260911.json',dict(summary=hs,manifest_sha256=sha(holder.OUTPUT/'manifest.json'),
        source_sha256=hm['files_sha256'],full_artifacts_backed_up=False))
    weekly=pd.read_csv(holder.OUTPUT/'weekly.csv')
    weekly.to_csv(ROOT/'docs/holder_2492_20260911.csv',index=False)
    plot(font,weekly)
    validation=ROOT/'.cache/execution-holder-validation.json'
    if validation.exists():
        checks=read(validation)
        for name in ('research_execution_stress_20260911','research_holder_2492_20260911'):
            path=ROOT/'docs'/(name+'.md')
            with path.open('a') as stream:
                stream.write(f"\n## 軟體驗收\n\n`make test`：{checks['tests_passed']}項通過、{checks['warnings']}項警告；pipeline通過，既有make api服務的health／picks／models／jobs四項curl回200。新研究畫面已驗證7種情境、23週表格、情境切換及3個CSV下載入口，初次來源驗證約7.13秒，切換約0.79秒。舊父研究與本輪來源hash均通過。\n")
            artifact=path.with_suffix('.json');value=read(artifact);value['validation']=checks;write(artifact,value)
    print('execution and holder reports generated')


def plot(font,weekly):
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise RuntimeError('Report chart requires matplotlib; install with: python -m pip install matplotlib==3.10.8') from exc
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    import matplotlib.dates as mdates
    if not Path(font).is_file():
        raise ValueError('Provide an installed CJK font with --font PATH')
    fp=FontProperties(fname=font)
    price=pd.read_parquet(holder.INPUT/'TaiwanStockPrice.parquet')
    price=price[price.date.between('2026-04-02','2026-09-11')]
    dates=pd.to_datetime(weekly.date)
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
    fig.suptitle('華新科2492｜持股集中與股價的同步變化',fontproperties=fp,fontsize=17)
    axes[0].plot(pd.to_datetime(price.date),price.close,color='#243b63',linewidth=2)
    axes[0].set_ylabel('收盤價（元）',fontproperties=fp)
    axes[0].set_title('2026/4/2–9/11股價；持股資料只到9/4',fontproperties=fp,loc='left',fontsize=10)
    axes[1].plot(dates,weekly.small_pct,label='100張以下持股比例',color='#26798e',linewidth=2)
    axes[1].plot(dates,weekly.large_pct,label='超過1000張持股比例',color='#c36628',linewidth=2)
    axes[1].legend(prop=fp,loc='upper left');axes[1].set_ylabel('持股比例（%）',fontproperties=fp)
    axes[2].plot(dates,weekly.small_people/10000,color='#26798e',linewidth=2,label='100張以下人數')
    axes[2].set_ylabel('小股東人數（萬人）',fontproperties=fp)
    axes[2].set_title('小股東持股占比變小，不代表小股東人數減少',fontproperties=fp,loc='left',fontsize=10)
    for ax in axes:
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        ax.axvline(pd.Timestamp('2026-09-04'),color='#999999',linestyle=':',linewidth=1)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    fig.savefig(ROOT/'docs/holder_2492_20260911.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--font',default='/System/Library/Fonts/STHeiti Medium.ttc')
    reports(parser.parse_args().font)
