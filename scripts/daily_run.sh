#!/bin/bash
# 每日自動執行：pipeline → strategy_d_pick → TG push
# launchd: ~/Library/LaunchAgents/com.jameschou.stockbot.daily.plist (週一~週五 18:00)

set -e
cd /Users/james.chou/JamesProject/stock_bot

# 載入 .env (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID / FINMIND_TOKEN ...)
[ -f .env ] && set -a && source .env && set +a

PYTHON=/Users/james.chou/.pyenv/versions/3.10.16/bin/python3
LOG_DIR=logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y-%m-%d_%H%M)
LOG="$LOG_DIR/daily_run_$TS.log"

# ── TG 推播 helper（含 retry，避免 launchd 醒來時 DNS 還沒就緒）──
tg_send() {
    local msg="$1"
    [ -z "$TELEGRAM_BOT_TOKEN" ] || [ -z "$TELEGRAM_CHAT_ID" ] && return 0
    for i in 1 2 3; do
        if curl -s --max-time 15 -X POST \
            "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d "chat_id=${TELEGRAM_CHAT_ID}" \
            --data-urlencode "text=${msg}" > /dev/null 2>&1; then
            return 0
        fi
        sleep 10
    done
    return 1
}

# ── 等網路就緒（launchd 從休眠喚醒時 DNS 可能還沒好）──
wait_for_network() {
    for i in $(seq 1 30); do
        if curl -s --max-time 5 https://api.finmindtrade.com > /dev/null 2>&1; then
            echo ">>> network OK (after ${i} attempts)" | tee -a "$LOG"
            return 0
        fi
        echo "    [wait_net] attempt $i/30, sleeping 10s..." | tee -a "$LOG"
        sleep 10
    done
    echo "!!! network NOT ready after 5min" | tee -a "$LOG"
    return 1
}

echo "==== $(date) START ====" | tee -a "$LOG"

# 0) 等網路（避免 DNS 解析失敗）
if ! wait_for_network; then
    tg_send "⚠️ Stock_bot $(date '+%F %H:%M') 網路 5min 仍未就緒，跳過今日 pipeline"
    exit 1
fi

# 1) Pipeline（資料更新 + features + labels + model + picks）─ 最多 3 次 retry
echo ">>> [1/3] make pipeline" | tee -a "$LOG"
PIPELINE_OK=false
for try in 1 2 3; do
    echo "    [pipeline] attempt $try/3" | tee -a "$LOG"
    if $PYTHON scripts/run_daily.py >> "$LOG" 2>&1; then
        PIPELINE_OK=true
        break
    fi
    echo "    [pipeline] attempt $try failed, sleep 60s before retry" | tee -a "$LOG"
    sleep 60
done

if [ "$PIPELINE_OK" = "true" ]; then
    echo ">>> pipeline OK" | tee -a "$LOG"
else
    echo "!!! pipeline FAILED after 3 retries" | tee -a "$LOG"
    tg_send "⚠️ Stock_bot pipeline FAILED 3x $(date '+%F %H:%M')，請檢查 $LOG"
    exit 1
fi

# 2) Strategy D pick（基於今日收盤 + max_price 250 過濾）
echo ">>> [2/3] strategy_d_pick" | tee -a "$LOG"
if $PYTHON scripts/strategy_d_pick.py --max-price 250 >> "$LOG" 2>&1; then
    echo ">>> D pick OK" | tee -a "$LOG"
else
    echo "!!! D pick FAILED" | tee -a "$LOG"
    tg_send "⚠️ Stock_bot D pick FAILED $(date '+%F %H:%M')，已跳過推播以免推到昨日舊訊號，請檢查 $LOG"
    exit 1
fi

# 3) Push D 訊號到 Telegram
echo ">>> [3/3] telegram push D" | tee -a "$LOG"
if $PYTHON scripts/telegram_bot.py --push --strategy d >> "$LOG" 2>&1; then
    echo ">>> TG push OK" | tee -a "$LOG"
else
    echo "!!! TG push FAILED" | tee -a "$LOG"
fi

echo "==== $(date) END ====" | tee -a "$LOG"

# 清掉 30 天前 logs
find "$LOG_DIR" -name "daily_run_*.log" -mtime +30 -delete 2>/dev/null || true
