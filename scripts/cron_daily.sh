#!/usr/bin/env bash
#
# 每日排程入口：跑 migrate + pipeline
#
# 建議排程時間：17:30 ~ 18:00 之間
# 理由：台股 13:30 收盤，TWSE 各 endpoint 大約 17:00 後才更新完
#   - STOCK_DAY_ALL（盤後成交）約 15:30 更新
#   - T86（三大法人）約 17:00 更新
#   - MI_MARGN（融資融券）約 21:00 才更新 ← 注意這個
#   - BWIBBU_d（PER）約 17:00 更新
# 若需要當日 margin，可改 21:00 或當晚跑兩次（17:30 + 22:00）。
#
# 安裝範例（macOS / Linux）：
#   make schedule-install        # 一鍵把這支腳本加進 crontab 17:30 工作日跑
# 或手動加 crontab：
#   30 17 * * 1-5 cd /Users/james.chou/JamesProject/stock_bot && bash scripts/cron_daily.sh
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

mkdir -p logs
LOG_FILE="logs/cron_daily_$(date +%F).log"

{
  echo "=== cron run $(date -Iseconds) ==="
  make migrate
  make pipeline
} >> "$LOG_FILE" 2>&1
