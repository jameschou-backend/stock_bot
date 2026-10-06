# 飆升前訊號普查與公開策略家族研究

研究日期：2026-10-06。範圍：2024-01-02～2026-10-05。這是固定窗口的事件分類研究，不是帳戶複利回測，也沒有把任何策略升級為實戰認證。

## 固定方法與完整分母

- 使用封存2,076檔個股母體，0050只作現有規則的市場比較。要求當日資料有效、歷史身分合格、20日平均「收盤價×股數」估計成交值至少5,000萬元；這是代理成交值，不是交易所實際成交金額。
- 先固定T收盤可知訊號，只有昨日已知不成立、今日成立才算首日。未明前日不冒充首日。
- 兩個事後標籤：T+1還原開盤到T+20還原收盤至少+30%；同起點到T+60至少+50%。不用訊號當日收盤當買價，也不用未來最高點。
- 持有路徑每一市場日都要有有效資料；未成熟、缺日分列，不提前出場、不補值。
- 成本用現有訊號研究假設：買賣各0.1425%手續費、各0.1%滑價，個股賣出稅0.3%；另列扣費獲利勝率。飆升命中率不等於獲利勝率，未達+30%/+50%可能仍賺錢。
- 29套既有進場規則全部保留；另固定兩項來源支持的缺口原型，只在本研究測試，不增加正式36個可掃描評估器。
- 逐股以最早合格窗口起點取案例，下一起點至少間隔20/60市場日，避免該股票案例持有路徑重疊。不同股票和不同窗口仍相關。起點是事後標籤，不是當时可知的底部。
- 案例列出起點前10日至當日的已知首日訊號。完整案例可查，72個首頁例子僅按各年各窗口期末漲幅取12個展示，不參與抽樣或統計。

## 實際完成的普查

| 固定結果標籤 | 可評估股票日 | 達標股票日 | 達標公司 | 不重疊案例 | 未成熟 | 缺未來路徑 | 母體達標率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20日+30% | 416,313 | 20,911 | 807 | 2,604 | 12,215 | 2,455 | 5.02% |
| 60日+50% | 384,288 | 25,571 | 602 | 1,175 | 39,924 | 6,771 | 6.65% |

60日+50%的1,175段案例中，1,106段在起點前10日到當日出過至少一個既有訊號。但在各規則前後資料均可觀測的股票日比較中，「任一非POC訊號」在飆升窗口出現98.58%、非飆升窗口也出現98.10%；相對基準倍率只有1.0046。這表示把所有指標取聯集幾乎每檔都會亮，不能把「曾出現訊號」當成飆股辨識能力。

## 全部規則的飆升命中率

以下每列只統計該規則首日事件。POC資料母體有限，獨立列出，不能用全市場5%/6.65%作它的基準。其餘各規則仍有暖機與可觀測範圍差異，完整JSON保留每列精確分母。此表按原目錄順序，不挑選勝者。

| 規則 | 20日事件數 | 20日+30%命中率 | 60日事件數 | 60日+50%命中率 |
|---|---:|---:|---:|---:|
| 原封存價量突破 | 8,202 | 9.73% | 7,808 | 10.04% |
| 原突破＋紅 K | 7,900 | 9.47% | 7,529 | 9.97% |
| POC 上移＋紅 K（獨立硬篩研究版） | 1,363 | 12.47% | 1,223 | 14.72% |
| 中期動能 | 11,209 | 5.56% | 10,253 | 7.38% |
| 接近一年新高 | 10,949 | 9.52% | 10,366 | 10.70% |
| 收斂後帶量突破 | 2,352 | 7.06% | 2,173 | 8.28% |
| 20 日價格通道突破 | 15,718 | 7.49% | 14,800 | 8.84% |
| 55 日價格通道突破 | 10,831 | 8.70% | 10,347 | 9.84% |
| 跌出布林下軌後收回 | 6,444 | 3.07% | 5,865 | 4.67% |
| 上升趨勢回測均線 | 19,648 | 5.70% | 18,491 | 7.45% |
| 成交升溫與早期相對強勢（日掃描版） | 6,960 | 5.95% | 6,614 | 7.51% |
| 舊工廠：動能趨勢（日掃描版） | 11,260 | 7.71% | 10,704 | 8.59% |
| 舊工廠：均值回歸（日掃描版） | 4,918 | 3.68% | 4,486 | 4.66% |
| 舊工廠：400日價量高點（日掃描版） | 2,092 | 12.19% | 1,992 | 12.50% |
| MA20／60 金叉 | 3,817 | 4.82% | 3,548 | 6.29% |
| MACD 訊號線金叉 | 15,456 | 5.33% | 14,288 | 7.00% |
| RSI14 收復 30 | 3,485 | 4.13% | 3,113 | 3.95% |
| 慢速 KD 低檔金叉 | 18,433 | 3.86% | 16,762 | 5.51% |
| 布林收斂後上破 | 461 | 4.12% | 410 | 7.80% |
| Keltner 通道上破 | 12,481 | 8.39% | 11,833 | 9.33% |
| ADX 趨勢確認＋DI 金叉 | 2,384 | 4.03% | 2,215 | 6.37% |
| 一目均衡表過雲 | 5,781 | 5.71% | 5,361 | 6.40% |
| OBV 突破 20 日高點 | 25,574 | 6.41% | 23,949 | 8.08% |
| CMF20 由負轉正 | 12,878 | 6.00% | 11,968 | 7.90% |
| MFI14 收復 20 | 4,115 | 3.11% | 3,697 | 4.03% |
| 超過一倍 ATR 的收盤漲幅 | 30,909 | 7.31% | 29,030 | 8.48% |
| 盤整後第一根放量紅K（純價量子版） | 3,601 | 4.75% | 3,455 | 6.08% |
| 60日收盤突破＋領先0050 | 10,212 | 10.40% | 9,669 | 10.57% |
| 近五日成交值升溫 | 15,064 | 6.15% | 14,147 | 7.56% |
| 研究原型：真向上缺口＋紅K | 11,613 | 7.45% | 10,589 | 9.72% |
| 研究原型：缺口後五日內回測守穩 | 3,854 | 8.15% | 3,420 | 9.94% |

400日價量高點的60日命中率12.50%（249/1,992），其可觀測母體6.72%；值得保留研究，但87.50%仍未達+50%。既有策略研究已顯示多數規則平均超額不佳，尾端捕捉力增加不等於平均帳戶報酬能贏0050。

## POC：用同一批紅K首日事件交叉比較

限制為有已知up/down逐筆POC資料的原突破紅K首日，再比較不加POC條件、POC上移與未上移。這能避免把2026候選和其他年度全市場相比，也避免POC首次成立與原紅K首次成立的事件時點不同。

| 固定窗口 | 條件 | 事件數 | 達標數 | 飆升命中率 | 扣費獲利勝率 | 平均單筆淨報酬 |
|---|---|---:|---:|---:|---:|---:|
| 20日 | 不加POC門檻（僅已知profile母體） | 1,563 | 198 | 12.67% | 46.71% | 3.61% |
| 20日 | POC上移 | 1,313 | 159 | 12.11% | 46.46% | 3.27% |
| 20日 | POC未上移 | 250 | 39 | 15.60% | 48.00% | 5.42% |
| 60日 | 不加POC門檻（僅已知profile母體） | 1,397 | 215 | 15.39% | 52.25% | 13.43% |
| 60日 | POC上移 | 1,178 | 170 | 14.43% | 51.10% | 12.00% |
| 60日 | POC未上移 | 219 | 45 | 20.55% | 58.45% | 21.11% |

目前樣本沒有支持把POC上移當成必須通過的硬門檻。這不證明POC永遠無效，也不是對原POC優先排序帳戶的重跑；樣本集中於部分2026候選，歷史已反覆研究，尚無獨立樣本外證據。

## 同一股票也可能有不同起漲前狀態

以下是使用者曾關注股票的事後說明例；普查不是以這些名字挑股票。表中的漲幅從起點下一交易日開盤到60日後收盤計算。

| 股票 | 事後窗口起點 | 60日漲幅 | 起點前10日至當日訊號節錄 |
|---|---|---:|---|
| 2308 台達電 | 2025-07-03 | 95.79% | 原封存價量突破、原突破＋紅 K、20 日價格通道突破、55 日價格通道突破、舊工廠：動能趨勢（日掃描版） |
| 2408 南亞科 | 2025-09-25 | 152.00% | 原封存價量突破、原突破＋紅 K、中期動能、接近一年新高、20 日價格通道突破 |
| 3491 昇達科 | 2024-04-15 | 75.60% | 沒有已知首日訊號 |
| 3491 昇達科 | 2025-12-24 | 115.11% | 原封存價量突破、原突破＋紅 K、接近一年新高、20 日價格通道突破、55 日價格通道突破 |
| 6446 藥華藥 | 2024-03-22 | 50.84% | OBV 突破 20 日高點 |
| 2492 華新科 | 2025-07-29 | 56.25% | 原封存價量突破、原突破＋紅 K、收斂後帶量突破、20 日價格通道突破、55 日價格通道突破 |
| 2221 大甲 | 2026-05-28 | 55.39% | 中期動能、20 日價格通道突破、55 日價格通道突破、舊工廠：動能趨勢（日掃描版）、超過一倍 ATR 的收盤漲幅 |

「沒有已知首日訊號」不等於整段從未符合策略，也可能先前已成立、資料未知或未達5,000萬元估計成交值門檻。此研究沒有逐案證明新聞/訂單/主力因果，因此不以股價大漲倒推某題材一定是原因。

## 目前是不是所有策略？

不是。現有80項是本專案目錄：36項可每日評估（29進場、5篩選、2排序），44項仍在目錄或等待資料/介面。公開文獻、指標、事件與參數組合沒有一份可窮盡的「所有策略」清單。本次整理15個來源支持的研究方向；其中只有2項新缺口原型完成本輪離線測試，其餘不能冒充已完成回測。

### 動能、通道突破與趨勢 — tested_proxy

作者定義有形成期與持有期；不同均線不是彼此獨立證據。現有日掃描是改寫條件，不冒稱複製美股因子。

研究規格：固定形成期，和同日起訖0050對照；本輪另評估其飆升辨識力。

資料／介面需求：adjusted_daily_prices, historical_universe。

既有關聯模組：momentum, near_high, donchian20, donchian55。

來源：[mba.tuck.dartmouth.edu](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor.html)。

### 均值回歸與超跌收復 — tested_proxy

布林工具作者反對把碰帶視為自動買賣；趨勢盤與橫盤的適用性須分開實測。

研究規格：只保留事先固定收復条件，檢查是否抓到飆升或只抓到反彈。

資料／介面需求：adjusted_daily_prices。

既有關聯模組：bollinger_reclaim, legacy_mean_reversion, ma_pullback。

來源：[www.bollingerbands.com](https://www.bollingerbands.com/bollinger-band-rules)。

### 向上缺口與缺口後回測 — tested_prototype

本輪新增兩個離線研究原型；缺口定義取自平台官方，紅K及五日守穩是我們提出的假設，未加入正式36項掃描。

研究規格：今低>昨高且紅K；或缺口後1至5日回測上緣並收回，期間不收破下緣。

資料／介面需求：adjusted_ohlcv, corporate_actions。

既有關聯模組：尚無。

來源：[www.tradingview.com](https://www.tradingview.com/support/solutions/43000675999-gaps/)。

### 事件錨定VWAP與成本區移動 — requires_adapter

POC是最高成交量價位，VWAP是成交量加權均價，兩者不同；錨點必須事前指定，不能事後挑波段最低點。

研究規格：以首次放量日或已公布事件日為固定錨點，測守住成本線；日線近似與逐筆版本分開。

資料／介面需求：declared_event_anchor, consistent_volume_and_amount, intraday_prices_if_exact_execution。

既有關聯模組：poc_up_red, poc_red_priority。

來源：[www.tradingview.com](https://www.tradingview.com/support/solutions/43000669764-anchored-vwap-drawing-tool/)、[www.tradingview.com](https://www.tradingview.com/support/solutions/43000502040-volume-profile-indicators-basic-concepts/)。

### 產業動能、龍頭與族群廣度 — requires_data

產業動能有原始論文支持研究；台灣歷史族群名冊不足，不能以今日AI/衛星分類回填過去。

研究規格：族群相對0050強勢、成交占比增加及領先股啟動分开記錄，再做交叉因子比較。

資料／介面需求：point_in_time_sector_membership, whole_market_turnover, market_breadth。

既有關聯模組：group_diffusion, sector_breadth_flow, chain_flow_context。

來源：[onlinelibrary.wiley.com](https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00146)。

### 營收加速與公告後漂移 — requires_data

FinMind明示create_time從2026-04-21起才記錄，代表入庫日期而非公司公告時刻；月營收日期不可直接作可交易時點。

研究規格：用當時版本推估季節性預期，實際公告超預期後的下一可交易時段才形成訊號。

資料／介面需求：revenue_first_publication, revision_versions, historical_seasonality。

既有關聯模組：revenue_growth, revenue_surprise, revenue_pead。

來源：[finmind.github.io](https://finmind.github.io/tutor/TaiwanMarket/Fundamental/)。

### 財報驚喜、分析師上修與PEAD — requires_data

公告後漂移與市場注意力有學術研究；月營收驚喜不能當成EPS驚喜或分析師共識。

研究規格：比較事前共識與當期結果，再測上修是否領先價量；缺歷史共識時保持未評估。

資料／介面需求：earnings_release_timestamp, pre_release_consensus, revision_history。

既有關聯模組：guidance_surprise, financial_quality。

來源：[www.nber.org](https://www.nber.org/papers/w11683)、[www.nber.org](https://www.nber.org/papers/w9246)。

### 漲價、訂單、量產、驗證與政策催化 — requires_data

官方重大訊息含公司財務業務資訊，但事件類型與受惠鏈須另做可追溯抽取，公告存在不等於股價必漲。

研究規格：逐篇記錄首次披露、產品/產能/價格影響與受惠公司，再配合量能/族群；禁止事後故事回灌。

資料／介面需求：original_documents, first_publication_and_revision_times, dated_company_benefit_evidence。

既有關聯模組：theme_catalyst, news_event_group。

來源：[www.twse.com.tw](https://www.twse.com.tw/zh/about/company/service.html)。

### 法人連買、賣壓承接及融資修復 — requires_data

FinMind有法人、融資、借券資料，但淨買賣不等於完整庫存；外資與投信方向必須分拆。

研究規格：固定連買天數、淨買占成交量與價格抗跌；缺交易日不能補零。

資料／介面需求：daily_institutional_flow, first_available_at, margin_and_lending_balance, share_capital_adjustment。

既有關聯模組：institutional_flow, institutional_absorption, margin_lending_repair。

來源：[finmind.github.io](https://finmind.github.io/tutor/TaiwanMarket/Chip/)。

### 分點集中與持續承接 — requires_data

分點買賣表反映交易流量，不等於同一主力的持股或未賣庫存；名字熱門不能當勝率證據。

研究規格：先固定集中度與連買定義，用全部分點與對照股票測，不事後挑神分點。

資料／介面需求：complete_broker_flows, branch_identity_history, buy_sell_balance_checks, availability_timestamps。

既有關聯模組：broker_concentration, broker_persistence。

來源：[finmind.github.io](https://finmind.github.io/tutor/TaiwanMarket/Chip/)。

### 集保持股分級與大戶集中 — requires_data

FinMind提供持股級距、人數、股數與比例；級距不是投資人身分，週觀測日也不能直接當發布日。

研究規格：大戶占比增加與小戶減少使用首刊可得日，控制股本變更，再配合股價相對強弱。

資料／介面需求：holding_distribution_publication_time, consistent_share_count, weekly_history。

既有關聯模組：holder_strength。

來源：[finmind.github.io](https://finmind.github.io/tutor/TaiwanMarket/Chip/)。

### 價值、盈利品質、投資與現金流 — requires_data

French五因子與AQR品質研究提出可量化家族；買便宜且品質好不是短期飆升的充分條件。

研究規格：品質和估值分開評估，禁止季末日提前知道財報；檢查高品質是否少虧而不是提高飆升命中率。

資料／介面需求：point_in_time_financial_statements, book_value, market_cap, profitability, cash_flow, asset_growth。

既有關聯模組：financial_quality。

來源：[mba.tuck.dartmouth.edu](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_5_factors_2x3.html)、[www.aqr.com](https://www.aqr.com/Insights/Research/Working-Paper/Quality-Minus-Junk)。

### 低波動、低Beta與風險預算 — requires_adapter

AQR低Beta研究為因子組合，不能直接等同本專案三檔多頭選股；可作風險對照而非飆股保證。

研究規格：明訂126日Beta估計與截面範圍，測回撤與成本，獨立於飆升分類。

資料／介面需求：market_aligned_daily_returns, point_in_time_cross_section, portfolio_weight_rules。

既有關聯模組：legacy_defensive_low_vol, risk_momentum。

來源：[www.aqr.com](https://www.aqr.com/insights/research/journal-article/betting-against-beta)。

### 庫藏股、內部人買進與資本配置 — requires_data

內部人與回購研究是可檢驗假設；計畫公告、實際執行和內部人公開申報須區分。

研究規格：以公開發布後才可見的淨買入/已執行回購為事件，對照只宣布未執行的公司。

資料／介面需求：first_publication_buyback_plan, executed_repurchases, insider_report_release_time。

既有關聯模組：尚無。

來源：[www.nber.org](https://www.nber.org/papers/w4965)、[www.nber.org](https://www.nber.org/papers/w6656)、[www.twse.com.tw](https://www.twse.com.tw/zh/about/company/service.html)。

### W底、頭肩底與型態確認 — requires_adapter

原始研究把圖形轉成可程式辨認特徵；不能以右側未來K棒提早確認谷底。

研究規格：固定轉折需幾根確認及頸線條件，訊號只出現在確認日，保留假突破。

資料／介面需求：causal_pivot_confirmation, adjusted_ohlcv, predeclared_pattern_geometry。

既有關聯模組：chart_patterns, first_bar_confirmation。

來源：[www.nber.org](https://www.nber.org/papers/w7613)。

## 程式、版本及驗收

- 程式：`scripts/study_rally_precursors.py`；定點測試：`tests/test_rally_precursors.py`，19項通過。
- 最終來源：`.cache/scanner-20261006/inputs-v1`、`.cache/scanner-20261006/poc-complete-v2/report.json`；直接輸入SHA驗證沿用既有載入器。
- 最終研究：`.cache/rally-precursors-20261006/run-v4/report.json`；全部案例：同目錄`all-episodes.json`，3,779筆。
- 公開摘要：`artifacts/forward_simulation/strategy_rally_attribution_20261006.json`及SHA sidecar，含完整248列年度比較、全部62配置、POC配對、家族對照、文獻來源與15方向。
- 本次運算31.381秒（不含前置載入与输出），FinMind新增請求0。
- 31規則×2窗口=62個最終配置，全部寫入trial_registry。前3次實際診斷58+58+62=178配置亦保留並追補登錄；共240次配置評估，不冒充62個獨立檢定。
- v1誤把POC非候選的可知不成立也放入基準，導致比較分母不當；v2改為真正有已知profile候選。v3加入同紅K首日配對、前10日非飆升對照、2個新缺口原型；v4修正成交值代理文字並自動記錄試驗。過程沒有改20/60日或30%/50%門檻來追求好結果。
- 補充診斷：16個家族×窗口對照、6個POC配對組、2個去重案例窗口，均非獨立檢定。
- 年度切片保留2024/2025/2026，但這些歷史已研究過，不稱未見測試集。未做多重比較校正，沒有勝率保證、交易委託或排程變更。

重跑命令（輸出目錄須全新）：

```bash
python scripts/study_rally_precursors.py \
  --bundle .cache/scanner-20261006/inputs-v1 \
  --poc-report .cache/scanner-20261006/poc-complete-v2/report.json \
  --directions .cache/rally-precursors-20261006/research-directions.json \
  --output .cache/rally-precursors-20261006/NEW_RUN
```
