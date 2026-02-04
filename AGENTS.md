# Agent 工作規範

## 1) 目標
- 這是台股波段 ML 選股系統（Python）。
- 資料源：FinMind。
- DB：MySQL（docker），本機連線 `127.0.0.1:3307`，DB 名稱 `stock_bot`。

## 2) 永遠遵守的限制
- 不可把 `FINMIND_TOKEN`、DB 密碼或任何 secret 寫入 repo。
- 不可修改 `.env`（只可修改 `.env.example`）。
- 不可直接改 `main`（除非明確要求）；預設在 feature branch 工作。
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

## 6) Troubleshooting
- pip SSL 若失敗，可在「個人環境」手動用 trusted-host，但不要把 trusted-host 寫死到 Makefile 的正常路徑；只放 README troubleshooting。
