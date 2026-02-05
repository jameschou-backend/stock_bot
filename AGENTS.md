# Agent 工作規範

## 1) 目標
- 這是台股波段 ML 選股系統（Python）。
- 資料源：FinMind。
- DB：MySQL（docker），本機連線 `127.0.0.1:3307`，DB 名稱 `stock_bot`。

## 2) 永遠遵守的限制
- 不可把 `FINMIND_TOKEN`、DB 密碼或任何 secret 寫入 repo。
- 不可修改 `.env`（只可修改 `.env.example`）。
- **允許直接在 `main` 作業**（依使用者要求），但必須遵守：
  - 每個明確問題修完就 commit（小步提交，禁止 WIP commit）
  - push 前必跑 `make test`；涉及 pipeline/DB/資料 ingest 變更時，必跑第 3 點完整驗收
  - 若造成回歸，必須使用 `git revert` 回滾（禁止 reset/force push）
- 不可引入「silent fallback」造成不同環境不同行為；缺依賴必須用明確錯誤訊息指引安裝。

## 3) 必跑驗收命令（每次改動後）
- `make test`
- `make pipeline`（必須可重跑，idempotent）
- `make api` 後用 curl 驗證：
- `curl -s http://127.0.0.1:8000/health`
- `curl -s "http://127.0.0.1:8000/picks"`
- `curl -s "http://127.0.0.1:8000/models"`
- `curl -s "http://127.0.0.1:8000/jobs?limit=10"`

## 4) Commit 規範
- 每修完一個明確問題就要 commit，不要累積一大坨。
- 使用 Conventional Commits：`feat` / `fix` / `chore` / `docs` / `test`。
- Commit 前必做：
- `git status`
- `git diff`
- `git add -A`
- `git commit -m "..."`

## 5) 資料處理規範
- ingest 需要做欄位 mapping（集中管理），並避免把非股票代碼寫入 DB。
- `stock_id` 校驗規則：預設只允許台股四碼（regex `^\d{4}$`），例外要有清楚註解。
- features/labels 嚴禁資料洩漏（只用當日可得資料）。

## 6) AI Assist 規範
- 僅在 `make pipeline` / `make test` / `make api` 失敗時啟用回問。
- 產出位置：`artifacts/ai_prompts/`、`artifacts/ai_answers/`。
- 嚴禁洩漏 secrets（`OPENAI_API_KEY`、`DB_PASSWORD`、`FINMIND_TOKEN`）；prompt 與回覆需遮罩。
- AI 回覆僅作為建議，需由人或 agent 轉成實際 patch / TODO 後提交。

## 7) Troubleshooting
- pip SSL 若失敗，可在「個人環境」手動用 trusted-host，但不要把 trusted-host 寫死到 Makefile 的正常路徑；只放 README troubleshooting。
- 若 GitHub repo 顯示 empty，通常是遠端缺少 default branch（例如 main）。可用：
- `git push -u origin <current_branch>:main`
- 或到 GitHub 設定 default branch。

## 8) Push / PR 規範
- 每次完成一組可驗收的 commits，且 `make test` 通過後，必須 push 到遠端：
  - `git push`
- 不得 force push。
- **允許直接推 main**（依使用者要求），但 push 前需滿足：
  - `make test` 通過
  - 涉及 pipeline/DB/ingest 變更時，需完整跑第 3 點驗收（pipeline + api + curls）
  - 發現回歸優先用 `git revert` 回滾提交
- 若 push 失敗（權限/認證），需輸出完整錯誤訊息與建議解法（PAT/SSH）。
