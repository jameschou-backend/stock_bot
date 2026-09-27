# 次一交易日進場：研究方向與報酬總表

整理日期：2026-09-28。這份是既有結果的完整方向盤點及帳本核對；沒有重新調參、重跑選股、下單或恢復排程。

## 如何讀這份報告

T 日收盤形成訊號，T+1（下一交易日）嘗試成交；本表不加入「進場再晚一天」或「出場再晚一天」的壓力。沒有量、漲跌停、資金或名額不足仍可能買不到；不能把所有訊號強制視為成交。既有日資料引擎使用次日收盤／日成交參考價及量價限制，**不是次日開盤價回測，也不是撮合簿成交保證**。

A～C 主期間為 2022-01-03 至 2026-09-09，本金100萬元，依帳戶淨值複利，閒錢留現金，0050只作獨立基準，不買00631L。D是較短期間，另列。百分比均為全期累積淨報酬，不是年化；期末資產含持股及應收估值，不等於可立即提領現金。

一般成本每邊滑價0.45%、佣金0.1425%（每筆最低20元）、股票賣出稅0.3%；0050賣出稅0.1%。普通盤容量受當日量及事前20日均量較小者1%限制，另有資源、名額及價格檢查；混合零股另受零股日量5%等限制。現金股利為毛額，未扣個人所得稅等帳戶外稅費。整張策略的公司行動餘股仍保留估值與曝險。

各組股票池、訊號頻率、名額與資料版本不同，只能在相同底座中比較新增條件。資料可用延遲（法人／集保／文件）和成交延遲是兩件事；資料較晚可用的組別仍在它的訊號成立後T+1嘗試買進。

「完成日資料帳戶」不代表已證明未來獲利、沒有任何漏洞或真實券商一定成交。以下均為已研究歷史，`live_qualified=false`。這次只核對結果檔與逐日帳本，沒有重新驗證所有供應商歷史原始版本。

## A．相同454個族群領先訊號：現行五名額、只整張帳戶

原篩選是族群領先訊號，不是只按成交金額買全市場股票；成交金額只決定同日候選的先後。除各列明示改動，其餘沿用12%收盤停損、63市場日上限、前日資金／名額鎖定。現行對照釋放已退出且只剩不足一張餘股的名額，餘股本身沒有假造賣出。

候補兩日組的+513.41%含T+2重試成交，不能放進「只准T+1成交、沒買到就作廢」的排名。籌碼main使用T−1以前法人與集保觀察日後8日；delayed使用T−3以前法人與集保後15日，均不是把成交再延一天。

| 研究方向 | 淨報酬 | 100萬期末資產 | 最大回撤 | 買賣筆數 | 成本合計（元） |
| --- | ---: | ---: | ---: | ---: | ---: |
| [0050同條件獨立基準](/Users/james.chou/JamesProject/stock_bot/.cache/residual-slots-20260926/final-v1/cases/benchmark_control.json) | +233.87% | 333.87萬 | -32.60% | 4 | 6,595 |
| [餘股仍占名額（舊名額規則）](/Users/james.chou/JamesProject/stock_bot/.cache/residual-slots-20260926/final-v1/cases/keep_0.json) | +380.67% | 480.67萬 | -18.71% | 158 | 359,452 |
| [族群領先＋成交金額排序＋12%停損（現行對照）](/Users/james.chou/JamesProject/stock_bot/.cache/residual-slots-20260926/final-v1/cases/release_0.json) | +415.79% | 515.79萬 | -20.84% | 177 | 436,680 |
| [候補保留兩日：T+1未買到可於T+2重試](/Users/james.chou/JamesProject/stock_bot/.cache/candidate-queue-20260927/final-v1/cases/valid2_0.json) | +513.41% | 613.41萬 | -25.52% | 183 | 530,662 |
| [依30%年化波動目標縮倉](/Users/james.chou/JamesProject/stock_bot/.cache/completed-corporate-studies-20260927/volatility-final-v1/cases/vol30_0.json) | +136.92% | 236.92萬 | -15.57% | 170 | 196,526 |
| [依40%年化波動目標縮倉](/Users/james.chou/JamesProject/stock_bot/.cache/completed-corporate-studies-20260927/volatility-final-v1/cases/vol40_0.json) | +161.06% | 261.06萬 | -14.69% | 172 | 222,206 |
| [依50%年化波動目標縮倉](/Users/james.chou/JamesProject/stock_bot/.cache/completed-corporate-studies-20260927/volatility-final-v1/cases/vol50_0.json) | +180.10% | 280.10萬 | -19.45% | 176 | 284,964 |
| [只持有63日，不加12%停損](/Users/james.chou/JamesProject/stock_bot/.cache/completed-corporate-studies-20260927/exits-final-v1/cases/fixed63_0.json) | +157.53% | 257.53萬 | -29.63% | 140 | 260,288 |
| [漲20%後回落12%出場／63日](/Users/james.chou/JamesProject/stock_bot/.cache/completed-corporate-studies-20260927/exits-final-v1/cases/trail20_12_0.json) | +207.32% | 307.32萬 | -26.38% | 166 | 361,133 |
| [連續兩日跌破20日線且弱於0050／63日](/Users/james.chou/JamesProject/stock_bot/.cache/completed-corporate-studies-20260927/exits-final-v1/cases/weak20_0.json) | +191.41% | 291.41萬 | -22.99% | 287 | 454,520 |
| [大盤與個股同步轉弱出場／63日](/Users/james.chou/JamesProject/stock_bot/.cache/completed-corporate-studies-20260927/exits-final-v1/cases/market_weak_0.json) | +151.03% | 251.03萬 | -28.83% | 146 | 247,116 |
| [強勢延長至最多126日](/Users/james.chou/JamesProject/stock_bot/.cache/completed-corporate-studies-20260927/exits-final-v1/cases/trend126_0.json) | +159.15% | 259.15萬 | -34.10% | 133 | 234,854 |
| [停損＋移動停利＋轉弱＋強勢延長](/Users/james.chou/JamesProject/stock_bot/.cache/completed-corporate-studies-20260927/exits-final-v1/cases/adaptive_0.json) | +98.01% | 198.01萬 | -24.38% | 311 | 457,540 |
| [20日低點支撐失效出場](/Users/james.chou/JamesProject/stock_bot/.cache/support-risk-20260927/final-v1/cases/support20_0.json) | +195.06% | 295.06萬 | -25.42% | 223 | 345,440 |
| [每筆計畫風險2%配置](/Users/james.chou/JamesProject/stock_bot/.cache/support-risk-20260927/final-v1/cases/risk2_0.json) | +132.60% | 232.60萬 | -17.55% | 184 | 227,661 |
| [支撐失效＋每筆計畫風險2%](/Users/james.chou/JamesProject/stock_bot/.cache/support-risk-20260927/final-v1/cases/support_risk2_0.json) | +52.71% | 152.71萬 | -26.13% | 212 | 199,010 |
| [支撐＋2%配置，再加一次強勢加碼](/Users/james.chou/JamesProject/stock_bot/.cache/pyramid-cash-20260927/final-v1/cases/pyramid_0.json) | +53.12% | 153.12萬 | -26.13% | 213 | 199,247 |
| [原策略加收斂突破篩選](/Users/james.chou/JamesProject/stock_bot/.cache/pattern-cash-20260927/final-v1/cases/pattern_0.json) | +53.18% | 153.18萬 | -27.73% | 84 | 120,245 |
| [支撐＋2%配置，再加收斂突破篩選](/Users/james.chou/JamesProject/stock_bot/.cache/pattern-cash-20260927/final-v1/cases/support_risk2_pattern_0.json) | +20.31% | 120.31萬 | -12.98% | 65 | 64,637 |
| [籌碼集中＋外資投信同買：優先排序](/Users/james.chou/JamesProject/stock_bot/.cache/leader-chip-accounts-20260928/run-a/cases/rank_main_0.json) | +369.13% | 469.13萬 | -20.77% | 169 | 430,538 |
| [同上，較晚可用籌碼資料](/Users/james.chou/JamesProject/stock_bot/.cache/leader-chip-accounts-20260928/run-a/cases/rank_delayed_0.json) | +368.31% | 468.31萬 | -25.54% | 180 | 448,276 |
| [籌碼集中＋外資投信同買：必備門檻](/Users/james.chou/JamesProject/stock_bot/.cache/leader-chip-accounts-20260928/run-a/cases/filter_main_0.json) | +57.45% | 157.45萬 | -17.63% | 73 | 99,130 |
| [同上，較晚可用籌碼資料](/Users/james.chou/JamesProject/stock_bot/.cache/leader-chip-accounts-20260928/run-a/cases/filter_delayed_0.json) | -3.07% | 96.93萬 | -21.67% | 49 | 60,627 |
| [只保留籌碼資料齊全的對照](/Users/james.chou/JamesProject/stock_bot/.cache/leader-chip-accounts-20260928/run-a/cases/coverage_main_0.json) | +152.14% | 252.14萬 | -20.34% | 172 | 276,783 |
| [資料齊全對照：較晚可用籌碼](/Users/james.chou/JamesProject/stock_bot/.cache/leader-chip-accounts-20260928/run-a/cases/coverage_delayed_0.json) | +142.57% | 242.57萬 | -25.06% | 174 | 294,946 |

## B．固定觀察日的籌碼與法人方向：五名額、整張加零股

每21個市場日的相對強勢／籌碼比較，母體及訊號日期與A不同。這是籌碼條件另行選股，不是A策略加同一個條件。

| 研究方向 | 淨報酬 | 100萬期末資產 | 最大回撤 | 買賣筆數 | 成本合計（元） |
| --- | ---: | ---: | ---: | ---: | ---: |
| [0050同條件獨立基準](/Users/james.chou/JamesProject/stock_bot/.cache/holder-flow-completed-20260928-a/benchmark_control.json) | +242.34% | 342.34萬 | -33.89% | 16 | 7,031 |
| [相對強勢＋籌碼集中](/Users/james.chou/JamesProject/stock_bot/.cache/holder-flow-completed-20260928-a/concentrated_strength_control.json) | +27.35% | 127.35萬 | -50.24% | 294 | 226,629 |
| [強勢＋集中＋外資投信同買](/Users/james.chou/JamesProject/stock_bot/.cache/holder-flow-completed-20260928-a/cs_both_buy_control.json) | +234.88% | 334.88萬 | -33.53% | 292 | 324,777 |
| [強勢＋集中＋外資投信同賣](/Users/james.chou/JamesProject/stock_bot/.cache/holder-flow-completed-20260928-a/cs_both_sell_control.json) | +93.56% | 193.56萬 | -25.29% | 234 | 215,094 |
| [強勢＋集中＋外資買／投信賣](/Users/james.chou/JamesProject/stock_bot/.cache/holder-flow-completed-20260928-a/cs_foreign_buy_trust_sell_control.json) | +19.11% | 119.11萬 | -29.32% | 279 | 230,927 |
| [強勢＋集中＋外資賣／投信買](/Users/james.chou/JamesProject/stock_bot/.cache/holder-flow-completed-20260928-a/cs_foreign_sell_trust_buy_control.json) | +18.87% | 118.87萬 | -39.30% | 306 | 217,076 |
| [相對強勢](/Users/james.chou/JamesProject/stock_bot/.cache/holder-flow-completed-20260928-a/relative_strength_control.json) | -36.89% | 63.11萬 | -62.37% | 237 | 153,927 |

## C．飆股研究轉成帳戶：相對強勢與同行成交升溫

固定觀察日的RS候選與「排除個股自身的同行成交升溫」候選；與B的已知資料範圍不同。整張與混合模式各有自己的0050基準。

| 研究方向 | 淨報酬 | 100萬期末資產 | 最大回撤 | 買賣筆數 | 成本合計（元） |
| --- | ---: | ---: | ---: | ---: | ---: |
| [0050持有 · 整張＋零股 · 一般成本](/Users/james.chou/JamesProject/stock_bot/.cache/sector-accounts-20260925/cases/benchmark_control_mixed.json) | +242.34% | 342.34萬 | -33.89% | 16 | 7,031 |
| [相對強勢 · 整張＋零股 · 一般成本](/Users/james.chou/JamesProject/stock_bot/.cache/sector-accounts-20260925/cases/relative_strength_control_mixed.json) | +1.55% | 101.55萬 | -45.05% | 285 | 224,335 |
| [相對強勢＋同行成交升溫 · 整張＋零股 · 一般成本](/Users/james.chou/JamesProject/stock_bot/.cache/sector-accounts-20260925/cases/strength_with_turnover_control_mixed.json) | +48.16% | 148.16萬 | -45.38% | 308 | 232,977 |
| [0050持有 · 只整張 · 一般成本](/Users/james.chou/JamesProject/stock_bot/.cache/sector-accounts-20260925/cases/benchmark_control_board_only.json) | +233.87% | 333.87萬 | -32.60% | 4 | 6,595 |
| [相對強勢 · 只整張 · 一般成本](/Users/james.chou/JamesProject/stock_bot/.cache/sector-accounts-20260925/cases/relative_strength_control_board_only.json) | +17.38% | 117.38萬 | -31.98% | 131 | 147,588 |
| [相對強勢＋同行成交升溫 · 只整張 · 一般成本](/Users/james.chou/JamesProject/stock_bot/.cache/sector-accounts-20260925/cases/strength_with_turnover_control_board_only.json) | +17.92% | 117.92萬 | -25.61% | 139 | 146,535 |

## D．低軌衛星官方營運催化＋突破：短期間帳戶

期間2025-04-16至2026-09-09。主名單只有原始七家公司；增補昇達科是使用者指定、已看過漲幅的案例，不能稱全市場事前盲選。文件首次發布版本仍未完全證明。原七檔0%表示零訊號、全現金。

| 研究方向 | 淨報酬 | 100萬期末資產 | 最大回撤 | 買賣筆數 | 成本合計（元） |
| --- | ---: | ---: | ---: | ---: | ---: |
| [七檔另加昇達科＋營運證據＋突破](/Users/james.chou/JamesProject/stock_bot/.cache/theme-catalyst-20260927-v4/augmented_delay0_confirmed_catalyst_control.json) | +35.71% | 135.71萬 | -13.46% | 6 | 13,592 |
| [七檔另加昇達科＋營運證據＋突破（文件多延五日可用）](/Users/james.chou/JamesProject/stock_bot/.cache/theme-catalyst-20260927-v4/augmented_delay5_confirmed_catalyst_control.json) | +35.71% | 135.71萬 | -13.46% | 6 | 13,592 |
| [0050同期間獨立基準](/Users/james.chou/JamesProject/stock_bot/.cache/theme-catalyst-20260927-v4/benchmark_control.json) | +175.43% | 275.43萬 | -15.34% | 7 | 6,223 |
| [原始七檔＋營運證據＋突破](/Users/james.chou/JamesProject/stock_bot/.cache/theme-catalyst-20260927-v4/primary_delay0_confirmed_catalyst_control.json) | +0.00% | 100.00萬 | 0.00% | 0 | 0 |
| [原始七檔＋營運證據＋突破（文件多延五日可用）](/Users/james.chou/JamesProject/stock_bot/.cache/theme-catalyst-20260927-v4/primary_delay5_confirmed_catalyst_control.json) | +0.00% | 100.00萬 | 0.00% | 0 | 0 |

## E．尚無完整帳戶報酬的方向

第一根放量＋籌碼集中、等突破、剛轉強、三日確認、首次回測支撐等：已發布的股票帳戶仍有歷史執行資料缺口，不能用已走完的部分路徑或單筆平均價格變化代替全期帳戶收益。三日確認與回測支撐的訊號定義本身較晚；若改為最初第一根T+1買進，就變成第一根策略，不是單純取消成交壓力。

| 研究 | 現有證據／未完成項 | 原始報告 |
| --- | ---: | ---: |
| 第一根＋集中，與等待突破比較 | 16個股票帳戶均中止；無完整淨報酬 | [research_first_bar_20260927.md](/Users/james.chou/JamesProject/stock_bot/docs/research_first_bar_20260927.md) |
| 全部第一根／剛轉強，各分直接、三日確認、首次回測 | 24個股票帳戶均中止；無完整淨報酬 | [research_early_strength_20260927.md](/Users/james.chou/JamesProject/stock_bot/docs/research_early_strength_20260927.md) |
| 低軌衛星名單僅要求突破、不要求營運證據 | 8個股票帳戶中止；不能當成0% | [research_theme_catalyst_20260927.md](/Users/james.chou/JamesProject/stock_bot/docs/research_theme_catalyst_20260927.md) |
| 飆股啟動前價量、均線、法人連買 | 事件／案例比較；沒有独立完整資金帳戶報酬 | [research_launch_flows_20260927.md](/Users/james.chou/JamesProject/stock_bot/docs/research_launch_flows_20260927.md) |
| 第一根後量縮、放量轉弱、資金轉往其他族群 | 提早出場的事件比較；不是100萬帳戶收益 | [research_launch_warning_20260927.md](/Users/james.chou/JamesProject/stock_bot/docs/research_launch_warning_20260927.md) |
| 法人賣但價格守住／吸籌代理 | 事件比較；没有完整資金帳戶報酬 | [research_absorption_20260927.md](/Users/james.chou/JamesProject/stock_bot/docs/research_absorption_20260927.md) |
| 題材×籌碼集中、華新科百張以下／千張以上比例 | 歷史題材日期與負例不完整；籌碼關聯及個股案例不等於策略 | [research_theme_chips_20260927.md](/Users/james.chou/JamesProject/stock_bot/docs/research_theme_chips_20260927.md) |
| 台達電、禾伸堂、國巨、華新科、華邦電、南亞科、南亞、聯電、群創、昇達科 | 指定贏家案例與廣母體對照，沒有這十檔合成的盲選績效 | [research_stock_launch_20260927.md](/Users/james.chou/JamesProject/stock_bot/docs/research_stock_launch_20260927.md) |
| 藥華藥及當時龍頭／輪動／大盤 | 個股案例診斷；無独立完整帳戶收益 | [research_pharma_context_20260927.md](/Users/james.chou/JamesProject/stock_bot/docs/research_pharma_context_20260927.md) |
| W底、頭肩底、完整新聞自動選題材、訂單與現金流品質 | 未有相同完整帳戶下獨立已完成比較；收斂突破的已完成結果見A | 不可填入推測報酬 |
| 戰爭／升息機率驅動水位 | 來源時序盤點；價格急跌／趨勢減倉舊結果見F，不能冒稱戰爭或升息預測模型 | 未有完整可核對同條件收益 |

## F．較早研究方向與當時報酬：保留版本，不混入A的排名

下列是原報告中的不額外延後成交結果；只做文件數值轉錄及來源雜湊，**本次未用現行引擎重新計算這些舊帳戶**。不同起訖、閒錢0050、候選母體、零股成交、資金重用、名額釋放及公告時間假設都有影響。不得稱這些數字已在現行成交規則下重現，也不能把全部方向最高值挑出來當實盤保證。原文稱「壓力成本」但只提高滑價的早期表格，與本輪多延一天的壓力不同。

原報告中的合併壓力欄位在此移除；限價等待、回測支撐及分批加碼本身可能晚於最初訊號T+1，已在分組說明。

### 閒錢配置／先前700%版本

2022-01-03～2026-09-09；舊三股帳戶，表內部分配置閒錢買0050。 原報告：[research_cash_allocation_20260910.md](/Users/james.chou/JamesProject/stock_bot/docs/research_cash_allocation_20260910.md)

| 配置 | 扣成本總報酬 | 最大回撤 |
| --- | ---: | ---: |
| 剩餘資金買0050 | +710.81% | -35.89% |
| 剩餘資金留現金 | +541.90% | -25.52% |
| 趨勢向上才買0050 | +705.92% | -37.81% |
| 只持有0050 | +242.07% | -33.96% |

### 持股名額及大盤校正排序

2022-01-03～2026-06-23；舊合成價格試算，非現行現金帳本。 原報告：[research_capacity_20260910.md](/Users/james.chou/JamesProject/stock_bot/docs/research_capacity_20260910.md)

| 方法 | 累積試算淨報酬 | 最大回撤 |
| --- | ---: | ---: |
| 原排序・3個部位 | 706.00% | -46.42% |
| 原排序・6個部位 | 622.51% | -35.37% |
| 可評分子集・原排序3個部位 | 558.96% | -46.42% |
| 同子集・大盤校正排序3個部位 | 554.31% | -46.64% |
| 0050持有 | 240.61% | -33.96% |

### 族群領先、擴散與接力股

2022-01-03～2026-06-23；舊合成價格试算及閒錢0050。 原報告：[research_diffusion_20260910.md](/Users/james.chou/JamesProject/stock_bot/docs/research_diffusion_20260910.md)

| 方法 | 累積淨報酬 | 最大回撤 |
| --- | ---: | ---: |
| 領先出現就買 | 326.50% | -39.92% |
| 擴散後買原領先股 | 15.78% | -64.30% |
| 擴散後買接力股 | 99.16% | -52.35% |
| 擴散後同群整籃持有 | 70.68% | -47.90% |
| 0050持有 | 240.61% | -33.96% |

### 大盤情境切換／閒置ETF／共同退出

2022-01-03～2026-06-23；舊合成價格與ETF配置。 原報告：[research_regime_switch_20260910.md](/Users/james.chou/JamesProject/stock_bot/docs/research_regime_switch_20260910.md)

| 方法 | 累積試算淨報酬 | 最大回撤 |
| --- | ---: | ---: |
| 所有領先訊號 | 326.50% | -39.92% |
| 趨勢只管進場 | 706.00% | -46.42% |
| 轉弱只收回閒置0050 | 702.56% | -46.46% |
| 轉弱連個股一起退出 | 57.84% | -50.74% |
| 初始一半領先策略、一半0050 | 283.55% | -33.50% |
| 初始一半趨勢進場、一半0050 | 473.30% | -39.12% |
| 0050持有 | 240.61% | -33.96% |

### 七種出場（舊三股＋閒錢0050）

2022-01-03～2026-09-09；A已有現金五名額新版，不得沿用舊值。 原報告：[research_exit_scenarios_20260910.md](/Users/james.chou/JamesProject/stock_bot/docs/research_exit_scenarios_20260910.md)

| 方法 | 累積淨報酬 | 最大回撤 |
| --- | ---: | ---: |
| 固定持有63日 | +659.48% | -45.42% |
| 跌12%停損 | +710.81% | -35.89% |
| 漲20%後回落12%停利 | -22.68% | -61.13% |
| 個股趨勢轉弱出場 | +142.17% | -48.16% |
| 大盤與個股同步轉弱出場 | +117.16% | -54.83% |
| 強勢延長，最長126日 | +64.70% | -61.40% |
| 依虧損、趨勢與大盤調整出場 | +43.70% | -50.58% |
| 0050持有 | +242.07% | -33.96% |

### 支撐、2%風險、加碼、收斂突破（舊底座）

2022-01-03～2026-09-09；A已有新版；此表含閒錢0050。 原報告：[research_technical_20260910.md](/Users/james.chou/JamesProject/stock_bot/docs/research_technical_20260910.md)

| 策略 | 累積報酬 | 最大回撤 |
| --- | ---: | ---: |
| 原策略 | +710.81% | -35.89% |
| 支撐出場 | +62.04% | -63.73% |
| 2%風險配置 | +321.61% | -33.94% |
| 支撐＋2%配置 | +168.99% | -43.75% |
| 支撐＋2%＋加碼 | +149.51% | -43.48% |
| 支撐＋2%＋收斂突破 | +188.07% | -38.34% |
| 0050持有 | +242.07% | -33.96% |

### 投信／外資加速、不追高、大戶、融資、借券、分點、族群、法人轉賣

2022-01-03～2026-09-09；舊底座與部分資料延遲假設，含閒錢0050。 原報告：[research_chip_20260911.md](/Users/james.chou/JamesProject/stock_bot/docs/research_chip_20260911.md)

| 方法 | 總淨報酬 | 最大回撤 |
| --- | ---: | ---: |
| 0050持有 | +242.07% | -33.96% |
| 原策略 | +710.81% | -35.89% |
| 投信買超加速 | +109.00% | -43.04% |
| 外資買超加速 | +208.96% | -49.14% |
| 不追高（距20日線≤10%） | +96.19% | -42.15% |
| 投信加速＋不追高 | +66.15% | -42.16% |
| 外資加速＋不追高 | +13.66% | -45.64% |
| 大戶持股增加 | +246.19% | -35.50% |
| 融資減少 | +36.05% | -44.80% |
| 大戶增加＋融資減少 | +71.63% | -52.37% |
| 大戶延後14日＋融資減少 | +72.10% | -39.30% |
| 借券賣出餘額減少＊ | +415.75% | -35.99% |
| 分點買超集中 | +690.40% | -40.54% |
| 同族群成交占比增加 | +76.90% | -50.09% |
| 法人轉賣後全出 | -19.06% | -51.36% |
| 法人轉賣後減半 | +194.39% | -39.60% |

### 限價、回測突破位、排序、分散、族群重疊與分批加碼

2022-01-03～2026-09-09；舊資金／名額規則。限價3天與回測策略不是全在原訊號T+1買到。 原報告：[research_five_axis_20260913.md](/Users/james.chou/JamesProject/stock_bot/docs/research_five_axis_20260913.md)

| 規則 | 原條件費後總報酬 | 最大回撤 |
| --- | ---: | ---: |
| 0050同條件基準 | +242.07% | -33.96% |
| 原現金版 | +541.90% | -25.52% |
| 限價等待3天 | +202.39% | -38.35% |
| 回測突破位後進場 | +254.35% | -35.27% |
| 前20日成交金額排序 | +1111.00% | -26.10% |
| 成交金額除以波動排序 | +1142.93% | -23.36% |
| 訊號族群不重疊 | +852.42% | -37.04% |
| 分散至5檔 | +510.36% | -25.35% |
| 先買一半再確認加碼 | +591.02% | -23.97% |
| 限價＋成交金額排序＋族群限制 | +211.41% | -35.48% |

### 營收超預期與毛利／獲利改善

2022-01-03～2026-09-09；公告15／30日與季報120／150日僅可用時間假設，不是已核實歷史公告時序。 原報告：[research_five_axis_20260913.md](/Users/james.chou/JamesProject/stock_bot/docs/research_five_axis_20260913.md)

| 規則 | 原條件費後總報酬 | 最大回撤 |
| --- | ---: | ---: |
| 營收同覆蓋對照，延遲情境15 | +505.30% | -33.08% |
| 營收超預期10%，延遲情境15 | +229.35% | -34.66% |
| 營收及財報同覆蓋對照，延遲情境15 | +730.33% | -37.09% |
| 超預期＋毛利改善及獲利為正，延遲情境15 | +79.22% | -38.31% |
| 營收同覆蓋對照，延遲情境30 | +505.30% | -33.08% |
| 營收超預期10%，延遲情境30 | +85.83% | -45.97% |
| 營收及財報同覆蓋對照，延遲情境30 | +730.33% | -37.09% |
| 超預期＋毛利改善及獲利為正，延遲情境30 | +81.05% | -34.57% |

### 分點買超集中與5日／20日持續性

2022-01-03～2026-09-09；舊底座，資料已知範圍對照與完整母體不能混比。 原報告：[research_broker_persistence_20260924.md](/Users/james.chou/JamesProject/stock_bot/docs/research_broker_persistence_20260924.md)

| 方法 | 候選數 | 正常淨報酬 |
| --- | ---: | ---: |
| 完整成交金額對照 | 458 | 1111.00% |
| 5日已知範圍對照 | 251 | 1060.02% |
| 5日持續性排序 | 251 | 738.13% |
| 5日持續性篩選 | 208 | 482.95% |
| 20日已知範圍對照 | 125 | 219.24% |
| 20日持續性排序 | 125 | 252.43% |
| 20日持續性篩選 | 53 | 311.93% |
| 0050基準 | — | 242.07% |

### 同日候選：原順序／成交金額／族群領先幅度

2022-01-03～2026-09-09；舊資源重用與名額規則，1111%不是現行版本。 原報告：[research_priority_20260924.md](/Users/james.chou/JamesProject/stock_bot/docs/research_priority_20260924.md)

| 方法 | 正常累積淨報酬 |
| --- | ---: |
| 原順序 | 541.90% |
| 成交金額排序 | 1111.00% |
| 族群領先幅度排序 | 378.82% |
| 0050 | 242.07% |

### 前日現金、未用預算、賣出資金、名額鎖定

2022-01-03～2026-09-09；執行假設消融；包含已被後續更嚴格規則替代的結果。 原報告：[research_execution_resources_20260924.md](/Users/james.chou/JamesProject/stock_bot/docs/research_execution_resources_20260924.md)

| 開啟限制 | 正常淨報酬 |
| --- | ---: |
| 無（舊日模型） | 1111.00% |
| U：鎖未用預算 | 1193.66% |
| S：鎖名額 | 874.57% |
| S＋U | 637.16% |
| C：限前日現金 | 725.02% |
| C＋U | 837.76% |
| C＋S | 874.57% |
| C＋S＋U | 637.16% |

### 日內補位與未成交名額釋放

2022-01-03～2026-09-09；舊執行版本的2×2×2消融。 原報告：[research_slot_reuse_20260924.md](/Users/james.chou/JamesProject/stock_bot/docs/research_slot_reuse_20260924.md)

| 資金背景 | 名額政策 | 正常報酬 |
| --- | ---: | ---: |
| 允許資金重用 | 可補位；未成交可釋放 | 1111.00% |
| 允許資金重用 | 可補位；未成交占位（F） | 1100.16% |
| 允許資金重用 | 當天不可補位（R）；未成交可釋放 | 874.57% |
| 允許資金重用 | 當天不可補位；未成交占位（R＋F） | 874.57% |
| 前日現金＋鎖未用預算 | 可補位；未成交可釋放 | 837.76% |
| 前日現金＋鎖未用預算 | 可補位；未成交占位（F） | 801.90% |
| 前日現金＋鎖未用預算 | 當天不可補位（R）；未成交可釋放 | 637.16% |
| 前日現金＋鎖未用預算 | 當天不可補位；未成交占位（R＋F） | 637.16% |

### 更嚴格資源鎖定後：三檔／五檔

2022-01-03～2026-09-09；458候選、整張加零股；A已更新歷史身分與整張政策。 原報告：[research_conservative_diversification_20260924.md](/Users/james.chou/JamesProject/stock_bot/docs/research_conservative_diversification_20260924.md)

| 帳戶 | 正常淨報酬 | 正常最大回撤 |
| --- | ---: | ---: |
| 3檔 | 637.16% | -37.76% |
| 5檔 | 552.49% | -29.20% |
| 同條件0050 | 242.34% | -33.89% |

### 候選保留兩日（舊混合成交）

2022-01-03～2026-09-09；三檔混合帳戶，和A整張五檔兩日候補不是相同策略。 原報告：[research_deferred_entry_20260924.md](/Users/james.chou/JamesProject/stock_bot/docs/research_deferred_entry_20260924.md)

| 帳戶 | 正常淨報酬 | 正常最大回撤 |
| --- | ---: | ---: |
| 原1日候選 | 637.16% | -37.76% |
| 保留2日候選 | 380.83% | -31.58% |
| 同資金規則0050 | 242.34% | -33.89% |

### 零股容量／對手價／滑價條件與帳戶風險

2022-01-03～2026-09-09；此處排除額外進出場延遲和合併壓力列。 原報告：[research_cash_risk_20260913.md](/Users/james.chou/JamesProject/stock_bot/docs/research_cash_risk_20260913.md)

| 情境 | 費後總報酬 | 最大回撤 |
| --- | ---: | ---: |
| 原現金版 | +541.90% | -25.52% |
| 零股加日末對手量限制 | +337.77% | -37.36% |
| 零股按對手價 | +534.70% | -25.79% |
| 每邊滑價0.90% | +705.98% | -26.15% |

| 情境 | 費後總報酬 | 最大回撤 |
| --- | ---: | ---: |
| 原現金版 | +541.90% | -25.52% |
| 60日均線減碼 | +180.58% | -28.15% |
| 急跌／波動減碼 | +473.73% | -22.47% |

### 同日隨機排序與漏買獲利股的敏感性

除另標2023／2024／2025起跑外，為2022-01-03～2026-09-09；敏感性診斷，不是按種子挑策略。 原報告：[research_cash_risk_20260913.md](/Users/james.chou/JamesProject/stock_bot/docs/research_cash_risk_20260913.md)

| 情境 | 費後總報酬 | 最大回撤 |
| --- | ---: | ---: |
| 同日排序 seed 11 | +1032.64% | -30.16% |
| 同日排序 seed 29 | +341.72% | -45.21% |
| 同日排序 seed 47 | +379.23% | -38.17% |
| 移除事後最賺的4958 | +446.12% | -24.12% |
| 移除事後最賺的4958及6739 | +505.30% | -33.08% |
| 2023起跑 | +430.87% | -36.55% |
| 2024起跑 | +299.20% | -26.58% |
| 2025起跑 | +121.51% | -26.87% |

### 60日線、急跌／波動減碼及有效收盤修正

2022-01-03～2026-09-09；價格風控代理，不是用事後戰爭或升息標籤。 原報告：[research_observed_risk_20260913.md](/Users/james.chou/JamesProject/stock_bot/docs/research_observed_risk_20260913.md)

| 規則 | 原成交條件總報酬 | 最大回撤 |
| --- | ---: | ---: |
| 原現金版 | +541.90% | -25.52% |
| 原60市場日風控 | +180.58% | -28.15% |
| 新60有效收盤風控 | +231.00% | -26.90% |
| 原急跌／波動風控 | +473.73% | -22.47% |
| 新急跌／波動風控 | +589.98% | -25.46% |
| 獨立0050 | +242.07% | -33.96% |

### 買入預算及名額事前預留

2022-01-03～2026-09-09；只列一般成本；後续整張版另見A。 原報告：[research_reservation_bridge_20260914.md](/Users/james.chou/JamesProject/stock_bot/docs/research_reservation_bridge_20260914.md)

| 帳戶 | 條件 | 流程 | 費後總報酬 | 最大回撤 |
| --- | ---: | ---: | ---: | ---: |
| 成交金額排序 | 一般 | 原流程 | +1111.00% | -26.10% |
| 成交金額排序 | 一般 | 事前預留 | +637.16% | -37.76% |
| 0050 | 一般 | 原流程 | +242.07% | -33.96% |
| 0050 | 一般 | 事前預留 | +242.37% | -33.89% |

### 次日預設限價／盤中條件成交

2022-01-03～2026-09-09；這是另套盤中條件假設，不能當A次日開盤結果。 原報告：[research_intraday_limit_20260914.md](/Users/james.chou/JamesProject/stock_bot/docs/research_intraday_limit_20260914.md)

| 策略 | 條件 | 扣成本總報酬 | 最大回撤 |
| --- | ---: | ---: | ---: |
| 原排序 | 一般 | +60.25% | -34.26% |
| 成交金額排序 | 一般 | +100.12% | -30.19% |
| 0050 基準 | 一般 | +233.95% | -31.27% |

### 中期動能、波動調整動能、接近一年新高

2018-01-02～2026-06-23；十檔、月末訊號次日收盤；原文表採每邊0.45%滑價，並未多延成交日。 原報告：[research_rules_20260909.md](/Users/james.chou/JamesProject/stock_bot/docs/research_rules_20260909.md)

| 方式 | 全期累積報酬 | 全期最大回撤 |
| --- | ---: | ---: |
| 中期動能 | +12.57% | -65.50% |
| 波動調整動能 | -28.20% | -69.43% |
| 接近一年新高 | +245.39% | -60.23% |
| 0050 買入持有 | +591.76% | -33.96% |

### 价格強勢＋投信持續買超／上漲放量

2018-01-02～2026-06-23；舊十檔月頻合成組合，非2022起完整資金帳戶。 原報告：[research_flow_20260909.md](/Users/james.chou/JamesProject/stock_bot/docs/research_flow_20260909.md)

| 方法 | 累積淨報酬 | 最大回撤 |
| --- | ---: | ---: |
| 價格強勢 | +49.44% | -60.84% |
| 加投信持續買超 | +17.70% | -41.64% |
| 加上漲放量 | -11.84% | -62.33% |
| 投信與放量都加 | +0.68% | -28.69% |
| 0050 買入持有 | +591.76% | -33.96% |

### 價格＋營收加速、投信與營收優先

2018-01-02～2026-06-23；舊十檔月頻及公告時間假設。 原報告：[research_revenue_20260909.md](/Users/james.chou/JamesProject/stock_bot/docs/research_revenue_20260909.md)

| 規則 | 累積淨報酬 | 最大回撤 |
| --- | ---: | ---: |
| 價格強勢 | +49.44% | -60.84% |
| 價格＋營收齊全 | +46.27% | -60.45% |
| 價格＋營收加速 | +5.33% | -63.20% |
| 價格＋營收加速＋投信 | -25.41% | -40.76% |
| 營收優先 | +15.18% | -55.91% |
| 0050 買入持有 | **+591.76%** | **-33.96%** |

### 新聞營運事件、題材群轉強及合併

2022-01-03～2026-06-23；供應商新聞日期已知不可靠，以下是作廢資格的診斷數字，不能作有效T+1績效。 原報告：[research_event_groups_20260909.md](/Users/james.chou/JamesProject/stock_bot/docs/research_event_groups_20260909.md)

| 規則 | 最多持有日 | 官方參考價：模擬淨報酬 |
| --- | ---: | ---: |
| 營運事件 | 63 | -5.59% |
| 題材群轉強 | 63 | 48.39% |
| 營運事件＋題材群 | 63 | 56.87% |
| 營運事件 | 126 | 11.40% |
| 題材群轉強 | 126 | 152.24% |
| 營運事件＋題材群 | 126 | 18.27% |

### 台積電實績超指引、展望上修、價量確認

2023-01-03～2026-06-23；單公司合成組合，含0050；文件首次版本及時序限制。 原報告：[research_guidance_20260910.md](/Users/james.chou/JamesProject/stock_bot/docs/research_guidance_20260910.md)

| 規則 | 累積淨報酬 | 最大回撤 |
| --- | ---: | ---: |
| 實績超過指引 | 335.76% | -27.48% |
| 實績超標＋價量確認 | 330.82% | -27.64% |
| 全年展望上修（時間診斷） | 338.33% | -27.48% |
| 兩者同時成立（時間診斷） | 335.48% | -27.69% |
| 0050 持有 | 333.72% | -27.48% |
| 70% 0050＋30%2330 固定持有 | 376.08% | -28.51% |

### 歷史股票池修正的影響（同為T+1、餘股仍占名額）

2022-01-03～2026-09-09；同為整張五名額，這是資料修正對照，不是四個任選的交易策略。以下結果檔和A～D一樣重新核對SHA256、NAV及帳本。

| 版本 | 淨報酬 | 最大回撤 |
| --- | ---: | ---: |
| 原股票池458候選 | +249.94% | -19.87% |
| 只修正身分／停牌／轉板 | +303.75% | -18.70% |
| 只補遺漏股票 | +326.90% | -20.21% |
| 身分＋遺漏股票一起修正 | +380.67% | -18.71% |

## G．其他研究記錄：不要拿事件漲幅充當帳戶報酬

| 方向 | 結果定位 | 來源 |
| --- | ---: | ---: |
| 新聞抓取、缺貨／漲價／出貨、產業鏈資金流 | 來源研究及當時族群成交占比，不是完整策略收益 | [news_research_20260909.md](/Users/james.chou/JamesProject/stock_bot/docs/news_research_20260909.md)；[research_chain_flow_20260909.md](/Users/james.chou/JamesProject/stock_bot/docs/research_chain_flow_20260909.md) |
| 記憶體、被動元件、低軌衛星歷史題材 | 事件窗口回報與價格口徑核對；不是有限本金複利帳戶 | [research_themes_20260909.md](/Users/james.chou/JamesProject/stock_bot/docs/research_themes_20260909.md) |
| 2492集保比例與股價 | 單股時段／21日事件關聯，不是全市場盲選 | [research_holder_2492_20260911.md](/Users/james.chou/JamesProject/stock_bot/docs/research_holder_2492_20260911.md) |
| 飆股共同性、同行領先與擴散 | 事件研究已轉成C帳戶；事件命中率不等於帳戶報酬 | [research_surge_anatomy_20260925.md](/Users/james.chou/JamesProject/stock_bot/docs/research_surge_anatomy_20260925.md)；[research_surge_sector_20260925.md](/Users/james.chou/JamesProject/stock_bot/docs/research_surge_sector_20260925.md) |
| ML模型、技術／籌碼權重、TopN與walk-forward | 既有模型研究不直接等於2022～2026/9/9的T+1完整資金帳戶；未確認同口徑前不填入收益 | artifacts/evaluation_*、artifacts/ai_answers/（舊研究） |
| PEAD、排名、零股與個人基準的早期規格 | 預登記不等於完成回測；不從規格推測報酬 | [prereg_pead_arm_20260711.md](/Users/james.chou/JamesProject/stock_bot/docs/prereg_pead_arm_20260711.md)；[prereg_pead_rank_arm_20260711.md](/Users/james.chou/JamesProject/stock_bot/docs/prereg_pead_rank_arm_20260711.md) |
| 00631L曝險、波動目標、2016起連續測試 | 屬使用者已排除的ETF路線，沒有把其收益計入個股方向排名 | [index_exposure_20260927.json](/Users/james.chou/JamesProject/stock_bot/artifacts/forward_simulation/index_exposure_20260927.json) |
| 資料清潔、歷史身分、股利、成交簿、核帳、統計校正 | 驗證工具與限制檢查，不是另一套選股收益 | 各資料驗收與account_statistics報告 |

## 本次核對範圍

直接核對47列已完成的一般情境帳戶（含各組基準），涉及47份不同結果檔。全部結果檔SHA256相符、NAV總報酬及最大回撤重算相符，逐日現金／股份／成本帳本稽核通過。對第一根、剛轉強及題材純突破另核對24列一般情境中止紀錄；未將中止前淨值作完整收益。這是已發布結果的核對，不是這次又執行47次新回測，也不代表重新檢查所有輸入來源和當時可交易性。

完整來源SHA256、case、設定、逐年收益、成本、帳本參照與中止原因見 [research_next_session_catalog_20260928.json](/Users/james.chou/JamesProject/stock_bot/docs/research_next_session_catalog_20260928.json)。本次研究盤點使用本機檔案，0次新增FinMind請求。專案驗收pipeline的既有增量請求另計。

現行可比A組中，T+1單日有效對照為+415.79%，0050為+233.87%；允許T+2候補另為+513.41%。新增籌碼硬門檻、更多出場／支撐／型態條件並未在這個底座改善正常全期報酬。這支持保留簡單底座繼續驗證，不能推論已排除選擇偏誤或已取得實戰資格。

## 專案驗收

- `make test`：3,951 passed，36項既有警告，83.81秒。
- `make pipeline`：exit 0；保留既有TWSE origin hold，受暫停的抓取步驟未發送HTTP，這不是資料已更新完整的證據。
- `make api`：8000埠已有服務，新啟動exit 2；保留既有服務，curl核對health／picks／models／jobs全部HTTP 200且JSON有效。
- 只新增本總表與來源索引；研究盤點不更動選股、成交或實戰資格，排程維持原狀。

驗收日誌位於`.cache/next-session-catalog-*-20260928.*`。
