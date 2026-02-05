#!/usr/bin/env bash
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
