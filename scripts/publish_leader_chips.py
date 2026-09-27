#!/usr/bin/env python3
"""Publish the complete fixed comparison only after both offline runs agree."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.research_exit_scenarios import read,write,sha
from scripts.research_leader_chips import OUTPUT,PREP,CHIPS,analyze
from skills.backtest_case_cache import file_identities
from skills.trial_registry import TRIAL_REGISTRY_PATH

REPORT=ROOT/'artifacts/forward_simulation/leader_chip_20260928.json'
DOC=ROOT/'docs/leader_chip_findings_20260928.md'
LABELS={'baseline':'舊版成交額排序','rank':'籌碼優先排序','filter':'籌碼硬篩選','coverage':'僅限制同資料覆蓋'}


def publish():
    proof_path=OUTPUT/'offline-verification.json';proof=read(proof_path)
    if (proof.get('passed') is not True or proof.get('compared_cases')!=16
            or file_identities([ROOT/p for p in proof['source_sha256']],ROOT)!=proof['source_sha256']):
        raise ValueError('Offline proof or source hashes invalid')
    folder=ROOT/proof['runs'][0];report=read(folder/'report.json')
    cases={name:read(ROOT/r['result']['path']) for name,r in report['cases'].items()}
    analysis=analyze(cases)
    if analysis!=read(folder/'analysis.json'):raise ValueError('Derived metrics changed')
    refs=dict(proof['source_sha256']);refs[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    trials=[json.loads(line) for line in TRIAL_REGISTRY_PATH.read_text().splitlines() if line.strip()]
    trials=[r for r in trials if r.get('source')=='leader_chip_increment' and r.get('run') in proof['runs']]
    for run in proof['runs']:
        for name in report['cases']:
            if name.startswith('benchmark_'):continue
            states=[r['status'] for r in trials if r['run']==run and r['case']==name]
            if states!=['started','completed']:raise ValueError('Trial history is incomplete or ambiguous')
    trial_path=ROOT/'artifacts/forward_simulation/leader_chip_trials_20260928.json'
    write(trial_path,dict(evaluations=28,distinct_strategy_configurations=14,reused_benchmarks=2,
        independent_validation=False,records=trials))
    refs[str(trial_path.relative_to(ROOT))]=sha(trial_path)
    descriptor=lambda p:dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
    publication=dict(report,analysis=analysis,source_sha256=refs,
        offline_verification=descriptor(proof_path),run_manifest=descriptor(folder/'manifest.json'),
        preregistered_screen=analysis['preregistered_screen'],publication_schema='leader_chip_publication_v1')
    if REPORT.exists() and read(REPORT)!=publication:raise ValueError('Preserve existing publication')
    write(REPORT,publication);REPORT.with_suffix('.sha256').write_text(sha(REPORT)+'\n')
    lines=['# 舊族群領先策略加籌碼：固定比較結果','',
        '2022-01-03～2026-09-09，100 萬元複利、五個個股名額、整張交易、閒置現金。保留相同 454 個訊號及出場規則；共 16 個帳戶，每組 1,136 交易日。',
        '', '**這是已研究歷史的增量比較，不是未見資料或實戰認證。** 過去約 700% 等舊版數字包含不同現金配置／資料／執行條件，本輪須比較可逐欄重現的封存基線，不能直接拼在同一排行榜。',
        '', '## 全部結果','',
        '正常滑價每邊 0.45%；合併壓力為每邊 0.90%，且進出各多延遲一日。兩種情境都包含手續費與交易稅。報酬包含期末持股及待收權益，未假設末日強制清倉。',
        '', '| 規則／資料時序 | 一般報酬 | 一般最大回撤 | 壓力報酬 | 壓力最大回撤 |',
        '| --- | ---: | ---: | ---: | ---: |']
    pairs=[('0050 獨立基準','benchmark_control','benchmark_combined'),('舊版成交額排序','baseline_0','baseline_7')]
    pairs += [(LABELS[a]+'／'+('主時序' if t=='main' else '延遲時序'),f'{a}_{t}_0',f'{a}_{t}_7')
              for a in ('rank','filter','coverage') for t in ('main','delayed')]
    for label,a,b in pairs:
        x,y=cases[a]['summary'],cases[b]['summary']
        lines.append(f'| {label} | {x["total_return"]:.2%} | {x["max_drawdown"]:.2%} | {y["total_return"]:.2%} | {y["max_drawdown"]:.2%} |')
    lines+=['','主時序：法人前 1 交易日、持股觀察後 8 日曆日；延遲時序：法人前 3 交易日、持股後 15 日曆日。所有條件固定在原訊號日；延後成交不更新籌碼。',
        '', '## 資料覆蓋與事前門檻','']
    for timing,stats in report['coverage'].items():
        lines.append(f'- {timing}：{stats["events"]} 個訊號，{stats["known"]} 個可判斷，{stats["passed"]} 個符合集中且外資投信五日皆淨買超；未知 {stats["events"]-stats["known"]} 個。')
    for arm,passed in analysis['preregistered_screen'].items():
        lines.append(f'- {LABELS[arm]}：事前研究保留門檻 **'+('通過' if passed else '未通過')+'**。門檻為兩時序×兩情境都勝基線、主時序兩情境都勝 0050，且各情境回撤不超過 50%。')
    lines+=['','未知與明確不符合分開記錄。coverage 組只排除未知；filter 與 coverage 的差額才是同資料覆蓋下的篩選比較。持股分布集中是代理指標，不能據此認定主力在吸籌；法人五日合計買超也不等於連續五天買超。',
        '', '## 成本、滾動區間與獲利集中','',
        '| 帳戶 | 買／賣筆數 | 成本合計（元） | 252 日超額勝率 | 最差 252 日超額 | 前兩大正獲利占比 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,r in cases.items():
        if name.startswith('benchmark_'):continue
        s=r['summary'];a=analysis['cases'][name]
        share='無正獲利' if a['top_two_positive_profit_share'] is None else f'{a["top_two_positive_profit_share"]:.2%}'
        lines.append(f'| {name} | {s["buy_count"]}／{s["sell_count"]} | {s["costs"]["total_cost"]:,.2f} | {a["rolling252_win_rate"]:.2%} | {a["rolling252_worst_excess"]:.2%} | {share} |')
    lines+=['','滾動視窗彼此重疊，只作穩定性描述，不能當成獨立試驗的成功機率。','', '## 逐年淨報酬','',
        '| 帳戶 | 2022 | 2023 | 2024 | 2025 | 2026 截至 9/9 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,r in cases.items():
        lines.append('| '+name+' | '+' | '.join(f'{a["total_return"]:.2%}' for a in r['summary']['annual'])+' |')
    cash=lambda name:sum(d['cash']/d['nav'] for d in cases[name]['account']['daily'])/len(cases[name]['account']['daily'])
    lines+=['','## 為何沒有改善','',
        '籌碼优先排序保留全部候選，但主時序實際改變了 15 個入場日的先後順序；部位名額與資金路徑因此改變。它在壓力情境改善，一般情境卻退步，兩個資料時序都是如此，沒有一致增益。',
        f'硬篩選把主時序候選縮到 62 個，一般情境只有 {cases["filter_main_0"]["summary"]["buy_count"]} 筆買進，舊版為 {cases["baseline_0"]["summary"]["buy_count"]} 筆；平均每日現金占資產 {cash("filter_main_0"):.2%}，舊版為 {cash("baseline_0"):.2%}。較低回撤部分伴隨較低投入，不能解讀成同等曝險下的選股改善。',
        '舊版主要獲利來源中，盟立、德宏在原訊號日未符合集中及外資投信同買；欣興、一詮、萬海雖集中，未符合外資投信同買。臻鼎兩個已買事件則因週距／股數分母檢查而屬未知，不能說它沒有集中。這些篩選確實排除了後續獲利機會，不代表事前可以知道誰必漲。',
        '穩懋仍符合且被選到，但其配置金額與帳戶資產路徑改變；因此不可把被排除股票的舊版損益直接加回新策略。完整 entry_paths 及 stock_pnl 已保留供逐筆核對。',
        '同覆蓋對照本身也顯著低於基線，表示「丟掉未知資料」就會改變績效；硬篩選還低於同覆蓋組，不能把全部退步都歸因資料缺失。此結果不支持把集中與法人同買直接設為飆股入場必要條件。',
        '', '## 準備、驗證與限制','',
        '- 原 277 檔候選均保留；新增 14 次 FinMind 法人請求，三檔週資料從原始全市場快取重建。所有 API 走共用額度，不重訓模型。',
        '- 104,525 筆跨版本重疊資料中，3491、3529 的 2026-07-31 投信各有差異；固定採逐股來源。兩檔該日期都未進入本輪事件使用窗口，沒有依績效挑來源。',
        '- 準備與績效分開：四個 filter 路徑原先在 2887 的配股阻擋，已保留失敗紀錄，補齊後全部重跑。成交來源新增 FinMind／官方 HTTP 次數見 preparation.json；公司文件另由瀏覽工具核對，沒有動用行情配額。',
        '- 2887 在 2024、2026 的配股比例與整股交付日依[公司股利表](https://www.tsholdings.com.tw/tsh/relations/shareholders/policy/distribute.html?__locale=en)核對。小數股款依[2024 股東手冊](https://www.taishinholdings.com.tw/tsh/relations/files/shareholders/meeting/Meeting-Manual-2024-Shareholders-Annual-General-Meeting_c.pdf)及[2026 股東手冊](https://www.tsholdings.com.tw/tsh/relations/files/shareholders/Meeting-Manual-2026-Shareholders-Annual-General-Meeting_c.pdf)按面額、元以下捨去；費用與實際付款日未確證，保留 0～毛額的不可動用應收款，沒有提前當成現金。文件為可回查的官方資料及事實摘錄，並非歷史首次發布快照。',
        '- 兩個 baseline 完整帳戶與舊版逐欄一致。兩次獨立禁止 HTTP 的執行中，16 份完整帳戶、排序、特徵、年度及滾動比較全部相同；交易、現金、股數、配股、費用逐帳核對。0050 重用封存核對帳戶，未作為個股帳戶的投資標的。',
        '- 三個歷史截點 × 兩種時序 × 未來截斷／改寫，共 12 項特徵因果檢查通過；原選股的因果性封存證據仍保留。這不能補出歷史首次公告或修訂版本，故不能宣稱嚴格 PIT 或零漏洞。',
        '- 每次正式回測皆記入 trial_registry，第二轮重現有相同 case 及獨立 run 標記，不將其當成新獨立策略。歷史反覆研究、多重嘗試及題材名單選擇偏誤仍存在。',
        '- 這輪為整張帳戶；日線成交、容量與滑價是模型假設，不代表完整逐筆零股撮合或未來可成交。沒有啟動排程、修改正式選股器或下單。',
        '- 驗收：make test 通過 3,951 項（36 個既有警告）；兩次 make pipeline 程序 exit 0，但保留 TWSE 來源 hold，不能解讀成最新資料已補齊。make api 因 8000 已有服務而 exit 2；保留服務後四個指定 curl 端點均 HTTP 200。詳見 leader_chip_acceptance_20260928.json。',
        '', '## 可重現紀錄','',
        f'- 報告：`{REPORT.relative_to(ROOT)}`。',
        f'- 第一輪：`{folder.relative_to(ROOT)}`；第二輪：`{proof["runs"][1]}`。',
        f'- 核對：`{proof_path.relative_to(ROOT)}`。',
        '- 各輪 `cases/*.json` 含所有買賣、每日資產、現金／股數／未成交原因；`features.json` 含每筆訊號的籌碼與可用時間；`analysis.json` 含每個滾動區間及基線買到／錯過事件對照。',
        '', '```sh',
        'python scripts/prepare_leader_chips.py --fetch',
        'python scripts/research_leader_chips.py --prepare',
        'python scripts/research_leader_chips.py --output .cache/leader-chip-accounts-20260928/new-run',
        '```','',
        '既有輸出不覆寫。來源與規則 SHA 改變會拒絕沿用準備證據；重新執行應採新的輸出名稱。', '']
    DOC.write_text('\n'.join(lines).replace('第二轮','第二輪').replace('优先','優先'))
    return publication


if __name__=='__main__':
    print('published',publish()['case_count'])
