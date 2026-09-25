# 回測資料補全與逐案證據清單（2026-09-25）

本輪新增可離線重播的股日依賴清單，並實際取得一項免費官方歷史產業異動。**完整普通盤／零股時序、完整歷史母體與公告修訂版本仍未補齊，`strict_data_ready=false`、`live_qualified=false`。** 日資料帳戶完成、逐筆檔可解析及來源真正完整是三個不同層級。

## 已完成且可重播

- `skills/backtest_data_evidence.py` 逐案抽取所有正數委託需求，包含未成交、買賣雙向與重試；按市場日／股票／通道去重。只買賣整張案例被政策拒絕的零股尾數另計，不要求禁止交易的通道。資源／名額檢查在送單前拒絕的 event 不冒充市場委託；部分帳戶只計已走到的路徑，不推斷中止後所需股日。
- 重用 `scripts/replay_contingent_day.py:load_tape` 及 `skills/intraday_limit_replay.py:normalize_ticks`，核對來源雜湊、股票／日期／通道、股數單位與時間格式；不新造市場資料匯入器。既有 `FrozenProviders` 仍是跨日逐筆重播的供應入口。
- 核對 `.cache/intraday-limit-20260914-final` 的 233 份普通盤檔與 `.cache/contingent-ticks-20260914` 的 43 份已完成檔，按股票／日期去重。舊 v1–v5 的拷貝不重算成新資料。實際去重、每案缺件與全部股日列表以 JSON 為準。
- 實際補抓公司事件新版 8 個完整帳戶的 **286 個缺漏普通盤股日**，全數成功，共 1,541,948 列、20,962,594 bytes（Parquet）。既有資料去重 269 股日，加補抓後共 555；這 8 案實際需要的 416 股日全部有本機檔，普通盤 `missing_sessions=0`。擷取於 2026-09-25 02:32:17–02:34:21 UTC，約 124 秒。新資料仍是 provider query／格式證據，不擅自認證完整場次或佇列成交。
- 新族群 12 案完成後，20 案聯集再補 **501 個缺漏普通盤股日**，全數成功，共 5,013,340 列、65,492,155 bytes，擷取於 03:20:41–03:25:54 UTC，約 313 秒。兩批合計 787 calls、6,555,288 列、86,454,749 bytes，無失敗／重試。本機 catalog 共 1,056 股日；目前 20 案已知路徑需要的普通盤 917 股日全部具備本機檔，零股仍缺 1,303 股日（204 家股票）。不是宣稱全市場、全日曆或所有逐筆重播可能路徑都已齊全。
- 新擷取並解析 TWSE 1121802250 號公告與附件：2023-05-22 公告、2023-07-03 生效，共 47 家舊／新產業類別。包含 PDF 換行的「電腦及週邊設備業」，要求七群總數與 47 個代碼唯一，缺列即失敗。[官方公告](https://www.twse.com.tw/rwd/zh/announcement/announcement_detail?id=346FAB95F87B11EDB2DA005056BE380E&response=html)、[附件](https://www.twse.com.tw/staticFiles/announcement/announcement/1121802250-1.pdf)。
- 這是 **TWSE 官方單一產業分類** 的一次異動；與 FinMind 多重供應鏈分類不同。事件保留 `prior_interval_start=null`、`next_change_date=null`、`official_publication_time=null`，不能延伸為 2022–2026 連續成分史，也不回寫原策略成員。
- 每案重新核對實際 orders／trades／holdings 的日期身分，重用最新上市櫃 episodes。新案例不能直接借用舊六帳戶通過的結果。
- `historical_value_available()` 要求正式發布時間、該版本可得時間、時區、version_id、原始內容及公告來源 SHA。入庫日或日後修訂的數值不能回填到過去決策。這是資料契約驗證，不自行認證公告真實性。

## 實查可取得範圍

| 資料 | 官方／提供商實查 | 對本案 2022-01-03 至 2026-09-09 的結果 |
|---|---|---|
| TWSE 普通盤成交 H2 | 2006-01-01 起、最近一年不提供，客製產製至少 7 工作天，內部使用每月全股 NT$10,000。[H2](https://eshop.twse.com.tw/zh/product/detail/0000000063ce6ab00163d860b694000a) | 已確認購買途徑，未下單；此商品也不能即刻補最近一年。免費格式／樣本不是全期資料。 |
| TWSE 委託 H1 | 同樣排除最近一年、內部使用每月全股 NT$10,000。[H1](https://eshop.twse.com.tw/zh/product/detail/00000000639057100163905e1d7c0001) | 真正佇列重建另需委託、新增／取消／變更及序號證據；只有成交列不能證明自己的委託成交。 |
| TWSE 盤中零股揭示 H4 | 2020-11-01 起、每月全股內部使用 NT$1,500，僅售前兩個月底以前資料；2026-04-01 起格式版本更換。[H4](https://eshop.twse.com.tw/zh/product/detail/0000000080da7fa70182334eb932009d) | 本機舊範例只有 10 筆試算、0 筆實際撮合；不能當作完整歷史盤中零股。未購買。 |
| TPEx MTH | 產品標示 2022/11/01 起，僅提供購買一年前資料；公開格式／樣本可讀，完整檔需客製訂購。[MTH](https://eshop.tpex.org.tw/zh/product/detail/2c92e0139984eab70199892c78bf0004) | 公開起點缺 2022-01-03 至 2022-10-31，最近一年亦有限制；訂購介面一般日期提示與商品文字不完全一致，需供應方確認範圍與用途授權，未自行推斷。 |
| FinMind 普通盤 Tick | TaiwanStockPriceTick 為 Backer／Sponsor 資料；2019-01-01 起，逐股逐日；上市櫃量單位為張。[技術文件](https://finmind.github.io/tutor/TaiwanMarket/Technical/) | 本輪使用已授權的既有 Sponsor 共用 adapter 補抓，沒有新 client。公開 schema 沒有獨立零股通道保證。需補最小請求數由 JSON 普通盤 missing 股日聯集算出，仍只涵蓋已知日資料路徑。 |
| FinMind 月營收 create_time | 2026-04-21 起記錄入庫日期，較舊資料空值，初始批次同日；不是正式公告時鐘。[基本面文件](https://finmind.github.io/tutor/TaiwanMarket/Fundamental/) | 不能把此欄當作 2022 年時點證據，也不能由目前值復原已覆寫的版本。 |
| MOPS 歷史更正查詢 | 本輪讀取[官方查詢入口](https://mops.twse.com.tw/mops/web/t120sb02_q10)得到 HTTP 200 的安全阻擋頁。 | 明確記錄取得 0 筆歷史內容；這不是證明所有免費歷史公告不存在。可透過官方歷史公告／更正與原始財報逐件補證，但本輪未取得完整版本鏈。 |

上述價格與範圍是 2026-09-25 擷取頁面的證據，不是購買指示。新 cache 保存原始 HTML／PDF、URL、HTTP 狀態、擷取時間、SHA256 與 PDF 抽取工具版本。本輪 9 次成功 HTTP 文件擷取（包含 1 份安全阻擋頁），前有 1 次 sandbox DNS 失敗；另有搜尋／網頁閱讀。普通盤補抓另計 shared-adapter calls，主稽核重播為 0 網路。沒有付費、資料庫改寫或交易。

## 歷史母體與公告版本的實際缺口

重用 `.cache/listing-continuation-20260924/report.json` 及離線重播收據；已改善原先 41 個上市起點未知問題，現在僅 1507、2358 的精確開始日尚未證實。另有 20 個未確認證券類別、19 個目前日期差異。選中記錄身分通過不等於完整歷史可投資母體；下市股票、停止／恢復交易、轉板與分類每次生效區間仍需完整覆蓋。明確只買 0050 的 benchmark 不使用全股產業／選股母體，因此這兩項標 `not_required_by_case`，全域缺口仍保留 false；ETF 自身的日期身分與公司事件仍要驗證。

現有 FinMind 分類只有當前 snapshot，缺每次成員加入／移出、分類改名、生效日、公告可得時間及歷史版本。新增 47 列官方事件可以提供部分反證與精確變更，無法補完另一套供應鏈分類。

本機月營收 append-only ledger 有 7,931 列、1,982 家、首次觀察日 2026-07-10 至 2026-09-22。第一批為左截尾，欄名 `announcement_date` 實為系統首次觀察日期；不能冒充正式發布時間。季報僅有 1101、2330 的 2026-09-24 觀察 snapshot，來源報表涵蓋較早季度也不表示當時拿得到此版本。前瞻 `source_guard` 的來源新鮮度收據只能保護當次紙上交易，不認證 2022 年歷史公告。

另將本輪 `docs/backtest_corporate_completion_20260925.json`（34 個來源 hash、2881／2880 共 3 筆事件）及 `docs/sector_account_corporate_20260925.json`（20 個來源 hash、3706 共 2 筆事件）納入驗證。已確認的公告日、普通股交付／可交易日與不確定的零碎股款日期逐欄顯示。2880 2026 年交付公告在研究結束日之後，僅用於保留期末未到期權利，不當作當時已知的選股資訊。3706 2026 年普通股交付日仍未知，保留 `pay_date=null` 與已核實最早界限 2026-09-15；研究末日 2026-09-09 嚴格早於該界限，只能保留不可交易應收股權。這些具體補證仍不是全市場所有版本與發布時鐘的完整歷史。

需補的資料欄位已列在 JSON `required_tape_fields`／`required_pit_fields`：逐筆必須保留日期、市場、通道、時區、時戳、股數、價格、試算／實際成交識別、序號／重複政策、格式版本、全場範圍、單位、逐日總量核對與原始 SHA；母體與產業需 valid_from／valid_to，公告需正式發布時鐘、version_id、版本可得時間、被取代版本與原始文件 hash。母體與特徵回看資料須從首次候選切點前起算，本案價量輸入起點為 2021-01-04。

## 使用與匯入邊界

```bash
python scripts/audit_backtest_data_completion.py \
  --extra-tick-summary .cache/backtest-board-ticks-20260925/corporate-v1/summary.json \
  --extra-tick-summary .cache/backtest-board-ticks-20260925/sector-v1/summary.json
python scripts/audit_backtest_data_completion.py --verify
python -m pytest tests/test_backtest_data_evidence.py -q
```

固定輸出：`artifacts/forward_simulation/backtest_data_completion_20260925.json` 與 `.sha256`；audit 不存績效數字、不啟動 pipeline/API，不碰既有封存文件。預設再次產生同名報告會失敗；重播使用 `--verify`。`verify_report(path, ROOT)` 適合 UI 消費前驗全部輸入／程式指紋；只驗 .sha256 文字而未查輸入，不能發現來源漂移。

JSON `cases[name]` 提供 `ordinary`／`odd_lot` 的 required、missing、local_format_valid、unverified、accepted 股日計數及明細，另有 `pit.components`、`missing_codes`、`case_completed_daily`、`complete_path`、`last_account_date`。`complete_path` 只指日資料案例跑完，`all_possible_paths_covered=false` 固定保留。`missing_sessions=0` 仍可能全部為 `local_format_valid_unverified`，不能啟動 strict 回測。

預設只檢查本輪 20 個完整新帳戶：corporate 為 `.cache/backtest-corporate-completion-20260925/probe-v2` 的 8 案，sector 為 `.cache/sector-accounts-20260925` 的 12 案；不合併舊封存中止案例。其他版本使用 `--case-set corporate=<目錄> --case-set sector=<目錄>`；每個目錄需有 `manifest.json` 與 `cases/*.json`，案例 SHA 必須符合 manifest。報告 keys 為 `corporate:<name>`／`sector:<name>`；`case_sources` 綁定實際路徑，`coverage_totals` 按股日去重，不把 20 個互斥研究帳戶的需求相加當作 API 次數。`--extra-tick-summary <新cache/summary.json>` 追加補抓來源，連同 plan、identity、attempts、hard budget、原始檔及 metadata 全部核對。報告保留明示 `reproduction` 輸入，未額外指定案例的 `--verify` 會依此重播，不重新猜測其他 cache。

補抓使用 `scripts/prepare_backtest_board_ticks.py --plan <盤點JSON> --output <新cache> --maximum <上限>` 先 dry-run，顯式加 `--fetch` 才讀 FinMind。它重用 `research_intraday_limit.TickCache` → `app.finmind.fetch_dataset` 與共用限額／cache，單批程式上限 600 calls。第一批 corporate 固定上限 286；20 案最終聯集另需 501，超出原保留 314，因此補抓前明確將本輪總額增為 787，第二批獨立上限 501。第一批 budget 及既有 preparer 程式完全未改，不為用完額度多抓。budget 與股日 attempt 在送出前保存；任何失敗即停止，重跑不會自動重試失敗股日；已成功但遺失的檔案也不會偷偷重抓。準備過程不產出策略報酬。

如取得真正完整原始檔，既有 `replay_contingent_day.py:load_tape` 接受 `finmind_board` 或 `normalized_auction_v1`。外部 manifest 的 `tapes` 列應包含 `path`、`sha256`、`format`、`stock_id`、`date`、`market`、`channel`；FinMind 格式另需 metadata_sha256。normalized 內容需 `timezone=Asia/Taipei`、`quantity_unit=shares`、`price_unit=TWD_cents`、`session_complete=true`、`synthetic=false`，每列含 time_us／price_cents／shares／record_type；空場需 `no_trades_confirmed=true`。但 `session_complete` 自我聲明不是獨立審核，此版驗證器拒絕用 Boolean 把現有檔案晉級 accepted；補齊後須新增有來源綁定的審核 adapter 並驗證原始檔、場次與單位，不能只是翻旗標。

明確重新抓公開文件才使用 `scripts/prepare_backtest_data_evidence.py --fetch --output <新空目錄>`（7 次 GET），或另加 `--execution-only`（H1／H2 共 2 次）。此入口不查市場 API、不讀 token，也不接受任意 URL。
