# 公告刊登日與原文版本補證

本輪封存華新科官網目前新聞索引列出的8篇營收公告，另外保存索引、HTTP回應、實際取得時間和原文SHA。使用9次官方HTTP、0次FinMind。它是可重播的局部證據，**不是2022年至今的完整月營收公告及修訂史**。沒有重新計算或更改任何已封存策略收益。

## 取得什麼

來源為[華新科官方新聞索引](https://www.passivecomponent.com/about/news/)，只沿索引實際出現的營收公告連結擷取，最多30篇；不猜測隱藏URL，也不逐月盲查。本次索引只有2024至2026的8篇內容，沒有據此假設其他月份不存在公告。

例如[2026年3月營收公告](https://www.passivecomponent.com/2026/04/09/walsin-technology-global-consolidated-net-sales-for-march-2026/)的營收所屬月份是2026-03，頁面可見刊登日期是2026-04-09。**這篇不能放到2026-04-02的策略判斷中。** 隱藏的 `.updated` 欄位是CMS更新時間，不能拿來當正式發言時間。程式會核對可見日期與官方URL日期，不一致即拒絕解析。

原文檔目前沒有可獨立核實的完整修訂鏈。因此即使頁面寫的是歷史日期，也不把今天取得的版本冒充當年已取得的內容。這一點同樣適用於其他七篇。

## 版本與時點規則

- `revenue_period`：營收所屬月份。
- `claimed_publication_date`：官網頁面明示的日期，只有日期精度。
- `official_published_at=null`：未憑空補入精確發言時分秒。
- `observed_at`：此次HTTP取得完成的真實UTC時間，CLI不接受自訂回填日期。
- `content_sha256`：公告標題、正文、刊登日及月份的語意內容指紋；CMS版型改變但公告內容相同，不另算內容修訂。
- `version_first_observed_at`：這一版內容第一次被本系統看見的時間。內容更正產生新版本，保留前一筆與其原始檔；同內容後續觀察不覆寫首次時間。

`as_of()` 僅回傳查詢時間前已實際觀察到的版本，並保守要求超過頁面刊登日的台北翌日零時；這是保守研究界線，不宣稱官網時區或精確發布時間已核實。較晚更正不會覆蓋較早查詢結果。目前這批原文首次封存在2026-09-25，對2026-04-02的嚴格版本查詢回傳0篇。它改善往後可追溯性，無法憑程式逆推出遺失的歷史版本。

## 使用與驗證

首次輸出：`.cache/publication-versions-20260925/walsin-v1/archive.json`。資料夾保留每份HTML及來源收據；`verify_archive()` 同時重查報告、原始檔、HTTP收據、版本鏈、實際查詢清單及解析內容，防止只修改摘要時間就把版本提前。

```bash
python scripts/capture_publication_versions.py \
  --verify .cache/publication-versions-20260925/walsin-v1/archive.json

# 有需要時手動保存下一次觀察；禁止覆寫既有資料夾。
python scripts/capture_publication_versions.py --fetch \
  --parent .cache/publication-versions-20260925/walsin-v1/archive.json \
  --output .cache/publication-versions-20260925/walsin-v2
```

沒有啟動排程。另行讀取MOPS新站公開入口時得到安全阻擋頁，已保存原文及取得時間；沒有繞過阻擋，也沒有把HTTP 200當作成功取得公告內容。完整MOPS公告與更正版本鏈仍需正式可取得的歷史來源。
