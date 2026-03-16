# 工作偏好與操作規範

> 最後更新：2026-03-16

---

## Git 工作流程

- **所有修改直接 push main**，不開分支、不發 PR
- Commit 使用 Conventional Commits 格式：`feat` / `fix` / `chore` / `docs` / `test` / `experiment`
- Push 前確保 `make test` 通過
- 不可 force push；若需回退，用 `git revert`

## 語言

- **輸出語言：繁體中文**
- 程式碼、技術術語、變數名保持英文原文

## 輸出路徑慣例

| 內容類型 | 路徑 |
|----------|------|
| 回測結果 JSON | `artifacts/backtest/` |
| 復盤報告 Markdown | `artifacts/review_pack.md`（最新）或 `artifacts/review_pack_{名稱}.md` |
| AI 提示稿 | `artifacts/ai_prompts/` |
| AI 回覆 | `artifacts/ai_answers/` |
| 模型檔案 | `artifacts/models/` |
| 報表 HTML | `artifacts/backtest_report.html` |

## 程式碼規範

- `stock_id` 只允許四碼台股（`^\d{4}$`），例外需加註解
- features/labels 嚴禁資料洩漏（只能使用當日可得資訊）
- 不可把任何 secret 寫入 repo（`FINMIND_TOKEN`、DB 密碼、API key）
- 不可修改 `.env`（只可改 `.env.example`）
- 禁止使用會造成環境行為不一致的 silent fallback

## 回測實驗完成後必做

1. 更新 `CLAUDE.md`「預設回測參數」區塊（若採用新配置）
2. 更新 `CLAUDE.md`「歷史基準對照」表格（加入新實驗結果）
3. 更新 `docs/strategy.md`（若生產配置變更）
4. 更新 `memory/decisions.md` 與 `memory/project_status.md`

## 驗收流程

最小驗收（每次改動）：
```bash
make test
```

完整驗收（涉及 pipeline/DB/ingest）：
```bash
make test
make pipeline
curl -s http://127.0.0.1:8000/health
curl -s "http://127.0.0.1:8000/picks"
```
