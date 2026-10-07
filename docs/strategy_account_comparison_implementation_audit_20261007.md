# 八組帳戶比較：實作稽核紀錄

稽核日期：2026-10-07。範圍為新增候選準備器、帳戶比較 runner 與資料適配器；本次僅閱讀程式與已保存來源，未重跑回測、未抓取資料、未修改策略或金融引擎。

**結論：在下列雜湊版本中，未發現會使比較失去既定公平條件、或需要中止執行的新增阻斷缺陷。** 這是有限範圍的程式路徑與來源交集查核，不代表系統無漏洞、全部八組已完成，或策略已獲實戰認證。尚未在本次稽核中以不同順序重新執行八組；最終完成狀態與離線重播一致性須另依正式輸出確認。

## 受查版本

| 檔案 | SHA-256 |
|---|---|
| `scripts/research_strategy_account_comparison.py` | `6a8c4da948b88a1959e45941d89c812c5554534d8efbc8ba5be7d38a38a4a455` |
| `skills/strategy_comparison_data.py` | `7be9e33b1308c4e74cf51f847b023d6bd347718ba8ebc2612424442ff8b09c60` |
| `scripts/prepare_strategy_account_comparison.py` | `89b05535e7c830da26b0c0a39b7c96b93b1dec46e3bce3a935e39458d73b6208` |

查核沿著實際呼叫與繼承路徑進行，沒有以測試通過取代實作檢查。後續公司行動證據或程式補丁不在以上版本的結論內；應保留新來源雜湊與重播證據。

## 進出場時間與共同條件

| 檢查點 | 實作與結果 |
|---|---|
| 候選時間 | `prepare_strategy_account_comparison.prepare_entries` 使用完整封存歷史暖機及原 scanner RSI14 收復 30 規則；`load_candidates` 與 runner `validate_candidate_calendar` 核對訊號日與下一個已觀測交易日。最後一天新訊號為 pending，不捏造期間外成交。 |
| 候選與排序 | runner `select_arm_entries` 保留原紅 K 的全部合格事件，沒有額外重算首次訊號；RSI 使用另一份固定候選，不額外套紅 K。RSI 同日按當時近 20 日平均估計成交值遞減、代碼遞增，runner 核對 event ID 平手排序與代碼排序相同。 |
| 股票池 | runner `_run` 對原候選、RSI 候選及 0050 取聯集後讀取行情和還原收盤；資格矩陣、交易日曆、期間及金融來源政策共用。沒有把 RSI 限於舊 POC 曾持有的股票。 |
| 三黑 K | `skills/three_black_exit.py:ThreeBlackSignals.exits` 與 `ThreeBlackControl.corporate_day` 只讀執行日前已完成 K 棒；最早須從實際進場日起已完成三個持有交易日。進場日可算第一根，不把進場前兩根拼成持有一天即三黑。 |
| 共同出場 | `skills/scenario_exit_replay.py:ExitSignals.context` 讀取執行日的前一日；`ScenarioExitReplay.corporate_day` 保存首次退出指示，再沿原金融與委託流程執行。原 12%／63 日規則及三黑 K 優先次序保持。12% 的原規則使用還原價格與進場日還原收盤作參考，不能解讀成每筆實際成交成本最多只虧 12%。 |
| RSI20 | runner `NativeTime20Scenario.corporate_day` 以實際第一次正數成交的 entry index 起算；買入日為第 1 日，完成第 20 日收盤後於 `entry_index + 20` 賣出。沒有用第 20 日收盤直接成交，亦不啟用 12%／三黑 K 提早退出。 |
| RSI20 重試與權益 | native MRO 只替換原出場決策，保留其後金融處理；首次 signal／target latch 在部分成交、未成交及股份晚到時仍保留。runner `audit_time20` 另用交易／批次與日曆核對時序，不只重用決策函式。 |
| 帳戶與費用 | 各組相同本金 100 萬、三檔個股名額、獲利複利、閒錢現金與共同成本。0050 為獨立基準，沒有回填個股帳戶的閒錢。依法不同的股票／ETF 賣出稅率保留，不把兩者設成不同研究滑價。 |

共同出場與 RSI20 的規則本來不同；`rsi_shared_exit` 才是比較訊號家族時使用相同出場的對照。不能用 `rsi_time20` 與 POC 的差額，精確歸因為單一進場指標的效果。

## 日區間價格代理

檢查 runner `engine_types` 的 MRO，`skills/poc_range_execution.py:RangeOrders` 位於舊 midpoint 執行祖先之前。`match_range_board` 與 `match_range_odd` 分別使用普通盤、盤中零股自己的日高低／成交量，買價為 `L + 0.7 × (H − L)`，賣價為 `L + 0.3 × (H − L)`。

數量、現金預算與限價由事前帳戶及前一交易日資料決定；當日高低代理價沒有进入候選排序。保留嚴格穿越限價、普通盤日量與前 20 日均量的 1% 整張上限、盤中零股獨立日量 1% 整股上限，以及同日賣款不支應新買、失敗訂單資源鎖與賣出未成交重試。

這是共同的**事後日區間成交估算**。不代表盤前知道當日高低，也不證明某時點有足夠對手量、排隊順位或實際成交。`daily_range_proxy=true`、`actual_fill_verified=false`、`live_qualified=false` 必須保留。

## POC 已知組與舊回退

- runner `MemoizedProfiles` 對原 event ID 綁定完整事件、深複製輸入輸出；同一事件跨組只保留同一原始評估結果。
- `profile_for_arm('red_known', ...)` 只把回傳副本中的已知 POC flag 改為 true，以借用硬篩選擇器保留原排序；不改共享 memo，因此不會把其他組的 POC false 污染成 true。
- `select_known_profiles` 只允許既定四類永久品質原因成為 unknown 排除。每次排除後使用全新的私有 `ReservationPlanner`，並檢查 live opening state 未被改動；不把臨時預留資源帶入下一次選擇。
- `poc_filter`、`red_known`、`poc_priority_known` 沒有整日回原排序的 fallback。`audit_profile_admissions` 再核對所有實際 funded cohorts，硬篩組必須原始 POC 為 true。
- 舊 `poc_priority` 經 `skills/volume_profile_account_adapter.py:AccountCandidateHook.corporate_day` 保留已登錄的永久品質問題整日原排序回退；這是明示策略差異，不可和共同已知組混稱相同過濾。
- 未抓取、空回應、配額／採集預算或未完成 request，經 `ComparisonDataRequired`／`RuntimeError` 使該組 incomplete；不能算 POC false、一般零成交或輸掉的交易。

共同已知的意思是使用相同可用性定義與原始評估，不代表不同帳戶路徑一定查詢或買進相同股票。現金／名額耗盡後的未查詢候選不被標記為不合格。POC 排序與硬篩本身亦分開報告。

## 共用 provider 與執行順序

檢查 runner `_run.replay`、`skills/million_replay.py:Replay.__init__`、`skills/poc_executable_data.py:BoardTicks`／`ExecutableAccountData.finmind`、`skills/poc_range_data.py:RangeOddData.get`，以及新適配器下列函式：

- `StrategyComparisonData.profile` 的來源優先順序固定為舊已封存 account profile、固定 daily donor、必要 lazy builder。股票／訊號日身分必須一致，memo 回傳副本；選哪個策略或帳戶持倉不參與 POC 計算。
- 每組重新建立 Feeds、Odds、Corp 與 engine。`Replay` 複製行情、深複製事件，重新初始化現金、持倉、應收、cohorts 和交易帳本；成交容量 `used` 是 engine／交易日狀態，沒有在組間共用。
- Corp 的載入快取每組獨立；共用公司行動條款只被讀取，`CorporateActions.on_date` 回傳每筆副本，沒有把上一組權益處理寫回共同條款。
- BoardTicks 共用已驗證來源與查核結果，回傳 DataFrame 副本。金融來源 `execution_loaded` 回傳副本。累積來源查詢紀錄是證據清單，不是下一組的現金、容量或選股狀態。
- `ComparisonTickReceipts.get` 保留舊來源優先順序，新 board／profiles 同座標重用原始 receipt；跨用途孤兒 attempt 會阻擋，不會換用途重抓來繞過失敗。尚未完成的來源不缓存成 POC false。
- `RangeOddData.get` 每次檢查同市场日已存在來源的一致性；新增來源只有在必要市場日缺失時取得，沒有依上一組成交量把下一組可用量扣除。

因此，程式狀態檢查未見「先跑某一組會改變另一組已知訊號或已知成交來源語義」的路徑。**有限採集預算仍會讓執行順序影響哪一組先取得必要資料、哪一組可以完成。** 該情況必須維持 incomplete，不能據此比較被截斷的回報。本稽核沒有以八組反向執行實驗證明順序不變；後续零網路重播才是封存來源上的運行證據。

## 103／4,612／30 個 POC 來源交集查核

下表是實際讀取並核對的來源；profile 檔的 SHA 同時與各自 report 中綁定的 SHA 相等。

| 來源 | SHA-256 |
|---|---|
| `.cache/poc-latest-20261003/run-v3/report.json` | `f82dbcf728199ead57bc41eeea7463acc6c2c78f26cfa490deef4a9735a3623c` |
| `.cache/poc-latest-20261003/run-v3/profile-data/poc_red/profile-features.json` | `77a06498bcdb8f872e5e3da3aac3ccf4f62778ff505acdb2160ab79d41d79572` |
| `.cache/poc-daily-opportunities-20261004/snapshots/20261004T063227825688Z/0063/report.json` | `d5dcdaf1c2f42bda3ac0eadd8392e0fa86ca8c3703ee61f4b7b1f6da3421976d` |
| `.cache/poc-daily-opportunities-20261004/snapshots/20261004T063227825688Z/0063/profiles.json` | `cdc52886db0d689a48423db51aa21f8874243f9b78f147e9df0caadd6b896af4` |

舊帳戶 profile 共 **103** 個 `(stock_id, signal_date)` 座標，daily donor 共 **4,612** 個座標，交集 **30** 個。交集的 `available` 全部一致；其中可用座標的舊 `poc_up` 與 daily `status == 'up'` 全部一致：**可用性或方向不一致為 0**。

這 30 個不是全部候選，不代表另外 4,582 個 daily 座標已與獨立引擎逐一重算，也不代表所有 unknown 原因逐欄完全相同。查核沒有重新下載或重建底層逐筆；只是已保存兩份評估的座標／可用性／方向一致性。適配器對 original profile 的固定优先順序不依這個交集結果改變。

## 結果發布界線

runner `preserve_failure` 將未完成組的 `summary` 設為 null，保存 partial journal；完整比較需有完整交易日與會計／執行 audit，不能從未完成帳本擷取暫時獲利發布成完整報酬。`verify_runner_source`、來源快照及結束時雜湊覆核防止在執行中修改來源卻引用新程式快照。

本紀錄不重述或認證報酬。原 POC／0050 的 legacy exact parity、最終新組的完成與封存重播、已結清勝率分母及未結清持股 NAV，仍須引用對應正式結果和獨立摘要檢查。所有成果維持 `historical_period_already_researched=true`、`unseen_validation=false`、`actual_fill_verified=false`、`live_qualified=false`。

## 補充：無確定支付日現金應收的報表口徑

追加唯讀檢查 `scripts/summarize_strategy_account_comparison.py:pending_cash_claims`、`summarize`、`markdown` 與對應回歸 fixture。受查摘要程式 SHA-256 為 `f463052c0d0698ce5211ab400f7ab5b0dffc61589eed72892f858cf9fb25aea4`。本補充沒有重跑交易或修改帳本。

`pending_cash_claims` 只彙總期末 `kind='cash'` 且 `pay_date is None` 的應收；同時列出帳載毛額 NAV 與這些款項淨支付為零時的期末 NAV，使用同一初始本金換算兩個端點報酬。它明示 `available_cash_increment=0`，未把待確認款項當作可再投資現金。`describe_cohorts` 的未結清權益判定仍獨立保留，沒有為了顯示敏感度把該批次放入已結清勝率。

Markdown 將主表欄位標為報酬／資產「估值」，逐組揭露無支付日應收，並直接說明只是該款項的期末敏感度、不是完整淨資產認證。JSON 亦明示不是信賴區間、不是另一條成交重播路徑、不是淨 NAV 認證。因此未見這次新增揭露把 3086 畸零款毛額冒充已收到的淨現金的阻斷問題。

此函式不是完整的公司行動費用估計器：不推定集保費、稅费、額外負債或真正淨額，也不涵蓋所有已定支付日或非現金權益的估值不確定性。兩個端點不代表所有可能結果的上下界；年度／最大回撤仍是原帳載估值路徑，不能宣稱已依淨款重播。

## 補充：3086／8932 公司行動證據與接線

本段於 `online-v4` 執行期間追加，只閱讀 live code、原公告與既存來源；沒有修改正在執行的程式，也沒有重跑交易。以下是此次補件的受查版本，前段原版本雜湊不覆寫。

| 檔案 | SHA-256 |
|---|---|
| `skills/strategy_comparison_corporate.py` | `a5f801967d9a94be6164ce0155d3c74d82ed31497cef5eb3b4c05340f3248444` |
| `skills/strategy_comparison_data.py` | `0c28f2a319783d4ec031d57bfb9880d2ff221f6a982af73371bcaae9948dc73a` |
| `scripts/research_strategy_account_comparison.py` | `9981ed28a997fb027bc799cc5162c08243a7960b1b818d009fd232bc5865ed32` |
| `docs/strategy_comparison_corporate_terms_20261007.json` | `0beb606d43e1aff2c42b94e1bfdc40785b8f7e863116e68adf5e5a72d7e31efb` |
| `docs/strategy_comparison_corporate_3086_evidence_20261007.json` | `4a5412cf18f865745d4510bbefdf8e7f40a2e0e4e9872480a0f3de278cdcc88a` |
| `docs/strategy_comparison_corporate_8932_evidence_20261007.json` | `2ecd1e66a939f36b180cdf344dae10006ea7b23a1e2f3b66c231c8910135030a` |

**未發現此補件新增的阻斷缺陷。** 比較本次 runner 與 `.cache/strategy-account-comparison-20261007/online-v3/runner_source.py`，逐行 diff 只有來源快照篩選 tokens 加入 `'strategy_comparison_corporate'` 一處。其用途是保存新 helper、terms 與證據 manifests 副本；候選、進出場、價格代理、成本與帳戶金融演算法沒有變更。公司行動補件會讓原本因交付日期未解而中止的帳戶得以繼續，不表示它們的舊部分帳本曾經完整。

### 公告事實

實際閱讀兩個 manifest 指向的八份原始 MOPS HTML，核對暫定權益公告、除權資料、正式交付與上櫃確認，不只接受整理後 JSON 的文字。

| 事件 | 原公告核對的權益 | 正式整股交付與可交易日 | 未確定事項 |
|---|---|---|---|
| 3086，2024-09-24 除權 | 每仟股配 100.00000423 股，面額 10 元；2024-09-05 權益公告 | 2024-10-08 發放公告、2024-10-17 上櫃確認支持 2024-10-29 | 畸零款指定抵費；實際淨額與支付日未確認 |
| 8932，2025-09-08 除權 | 每仟股配 84.14122310 股，面額 5 元；2025-08-20 最終權益公告 | 2025-10-23 發放公告、2025-10-28 上櫃確認支持 2025-10-30 | 畸零款指定處理帳簿劃撥費；實際淨額與支付日未確認 |

另將 `.cache/strategy-account-comparison-20261007/corporate-evidence/8932/wiselink-114-agm-minutes.pdf` 第 4 頁渲染後直接閱讀，並檢查第 5 頁通過決議，確認股票股利畸零款計算至元、元以下捨去及抵費條文。該 PDF SHA-256 為 `08e0dd7add57f1725fd0b5b450b591c71047464d4d79c926bec1307997defd45`。議事錄原提案每仟股 83.56 股不是本輪最終採用比率；最終 84.14122310 股以後續正式權益公告為準。沒有用慣常 10 元面額去反推 8932 權益。

### Helper／adapter／金融引擎

- `load_comparison_corporate_terms` 限定這兩個 action ID、settlement-only 用途與空的 cash supplements；檢查 manifest／原始來源／receipt 雜湊、主機、成功狀態、原公告股票與日期、確定交付／上市公告及日期順序。以 `Decimal` 比對每仟股、每股比率與歷史面額。
- 本次以真實目錄執行這個純讀 helper，兩筆條款成功載入，共綁定 **27 個來源檔**，包含原中止 case 作為補件脈絡。兩筆 qualifications 都保持 `fractional_net_cash_amount=None`、`fractional_cash_available_for_trading=False`。此檢查沒有初始化網路 provider 或執行交易。
- `StrategyComparisonData._load_corporate_terms` 在帳戶建立前載入條款；若與既有同 key 條款衝突則停止，不靜默覆寫。新條款與 helper／全部證據來源綁入 refs，估值限制進入 `profile_snapshot.corporate_settlement_qualifications`。
- runner 仍經原 `merge_overrides` 將 provider 的條款送入每組新建的 Corp。選股候選準備器、POC 計算及排序不讀這些新增交付條款；較晚公布的正式交付公告僅用於历史權益交付重建，不回填成除權前買進訊號。
- `FractionalCashActions` 與原 Replay 把整股交付與畸零現金權益分開。確認日期前股份不可交易；交付後保留原 cohort／退出指示。無支付日的畸零毛額只留應收，沒有自動在整股交付日支付或加進可用現金。

條款的 7 元與 2 元是原中止持倉範例（3086 的 1,147 舊股、8932 的 3,215 舊股）所算毛額，不是寫死於所有組的固定應收。最終揭露必須從各組實際 receivable journal 讀取。`pending_cash_claims` 本次仍為前段記錄的 `f463052c…` 版本，沒有為這兩筆補件重寫交易、填入假支付日、推定淨現金或將未結清批次算入已結清勝率。

正式整股交付證據已補強，**畸零費用扣抵後淨額與支付仍未認證**。毛額與零資產值端點只揭露該款項估值敏感度；不代表完整淨 NAV 已驗證，亦不改變非實戰認證的界線。最終新完成組與原已完成組的離線重播／完整帳戶 parity 仍需另行保存。

## 驗收限制：既有 daily pipeline 的研究特徵契約錯誤

本輪 `make pipeline` 仍失敗。已直接閱讀 `.cache/strategy-account-comparison-20261007/pipeline-post-corporate.log`：`daily_pick.run` 第 821 行把 `feature_df` 送入 `_research_score_candidates`，第 229 行對缺欄的 `pd.to_numeric(data.get(col))` 呼叫 `.fillna`，因回傳 scalar `numpy.float64` 而拋出 `AttributeError`。這個既有 daily pipeline 使用的模型／特徵來源，不是本輪封存帳戶 runner 的候選輸入；不能用帳戶測試成功宣稱 pipeline 已通過。

已沿來源與裁切路徑核對到兩個不同問題：

1. `skills/build_features.py:415` 有計算 `breakout_20 = close / rolling_max20 - 1`，但該欄不在第 184 行組成的 `FEATURE_COLUMNS`。第 2023 行保存清單只取 `FEATURE_COLUMNS`，同一清單於第 2073 行寫入 Parquet，因此 breakout 計算結果未持久化。不是換成未裁切的來源 DataFrame 就能取得。
2. `amt_ratio_20` 有計算，也有持久化，但在 `skills/build_features.py:231` 的 `_PRUNE_SET` 中。`daily_pick.run` 第 534～536 行讀取模型 artifact 的 `feature_names`，第 629～635 行先補齊／裁切為模型矩陣；研究 fallback 到第 821 行卻重用這個已裁切的矩陣。研究排序器自己的三欄契約因此與模型契約混用。

### 實際欄位檢查

純讀 `artifacts/features/features_2026.parquet` 的 schema 與最近日期資料：最新日為 **2026-10-06**、該日 **2,131 列**、共 **87 個特徵欄**。另讀取 `.cache/scanner-20261007/api-validation/models.json` 中最新的已保存模型紀錄，並打開其指向的本地 artifact `ranker_lgbm_20210805_20260804_8f02bc28e47710bf.pkl`：模型使用 **58 欄**。此處模型身分依該 API 快照，不冒稱重新查過當下 DB 最新模型；本次未查 DB、未寫入 DB。

| 研究排序必要欄位 | `FEATURE_COLUMNS` | 實際 Parquet schema | 最新日 null 數 | 上述模型 58 欄 |
|---|---|---|---:|---|
| `ret_20` | 有 | 有 | 0 | 有 |
| `breakout_20` | 無 | 無 | 不適用：整欄不存在 | 無 |
| `amt_ratio_20` | 有 | 有 | 0 | 無 |

研究 fallback 只在 research 模式且法人資料降級時啟用；已保存同日 data-quality job 確實記錄 `degraded_mode=true`、`degraded_datasets=['raw_institutional']`。既有單元測試 `test_research_mode_degraded_execution` 人工提供齊全三欄，只驗證法人值不影響分數，沒有穿過真實模型裁切／持久化路徑，因此未攔下這個契約錯誤。

### 修復界線

正確修復須將研究所需的原始特徵／有效性與模型矩陣分開，並補齊有來源證明的 breakout 計算或持久化契約；不能用 default Series、零分、其他相似指標，或全欄 NaN 後再 impute，讓驗收表面通過。研究欄位必須在填補前驗證，必要行情／還原因子缺失要明示停止或依另行固定的有效性政策排除並記錄分母。

先前未套用的 `.cache/scanner-20261007/legacy-pipeline-fix.patch` 採 21 個市場交易日價格重建、要求對齊還原因子且禁止暗補 1。其歷史嘗試 `.cache/scanner-20261007/pipeline-fixed.log` 已進一步停止於「沒有完整 price／adjustment inputs 候選」。這只能證明當次嚴格重建尚未具備完整輸入，不能推論目前所有股票都缺還原因子，更不能以最大資料日期已更新便證明每股每一天覆蓋完整。

本輪僅保存診斷，**沒有修改 `daily_pick.py`／`build_features.py`、沒有套回舊 patch、沒有默補 factor、沒有回填資料或新訓練**。策略比較與這個既有 pipeline 修復分開。整體測試與 API 即使通過，仍須如實標示 pipeline 未通過；不因這個例外聲稱完整實戰驗收完成。

| 診斷來源 | SHA-256 |
|---|---|
| `skills/daily_pick.py` | `2fccda318827ae2d551313f21ed9911887467ecf3525e4cb48796b604d6b79a8` |
| `skills/build_features.py` | `5f6675560b044d8c396455a833c58387f884114bb3f198594fe0b018b0ddd0f1` |
| `.cache/strategy-account-comparison-20261007/pipeline-post-corporate.log` | `b7f4b47c0757979ef29073358ee810db7bcf3b861acd8864d3ebbfa42b73bbf2` |
| `.cache/scanner-20261007/api-validation/models.json` | `1fba0dd3c198e64a7846c83dba40f8cd03a6b1fc3baff81b225de003f284566c` |
| `.cache/scanner-20261007/legacy-pipeline-fix.patch` | `28b11f54e2656db003bbb60449818625a68ac5dccc727c58114032a2fa53d6f1` |
| `.cache/scanner-20261007/pipeline-fixed.log` | `3ee9d3b164b58c4059d03ebdd0b702d0b264961f6d1341dc0ca5352bce4ace39` |
