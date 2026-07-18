#!/bin/bash
# 每日自動執行：pipeline → D pick(紙上) → ipo scan → paper nav → TG 誠實日報 → 哨兵 → 營收爬蟲
# launchd: ~/Library/LaunchAgents/com.jameschou.stockbot.daily.plist (週一~週五 18:00)
#
# 推播口徑（2026-07-18 起）：每日推「誠實日報」（telegram_bot.py --push --strategy daily-brief）。
# D 訊號推播已降級為手動（--strategy d / listen 模式 /signal d），不再每日自動推。

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
echo ">>> [1/7] make pipeline" | tee -a "$LOG"
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

# 2) Strategy D pick（紙上訊號，供手動 /signal d 用；失敗不中斷——
#    每日推播已改誠實日報，不依賴 D 訊號檔）
echo ">>> [2/7] strategy_d_pick (paper, non-fatal)" | tee -a "$LOG"
if $PYTHON scripts/strategy_d_pick.py --max-price 250 >> "$LOG" 2>&1; then
    echo ">>> D pick OK" | tee -a "$LOG"
else
    echo "!!! D pick FAILED (non-fatal, /signal d 會拿到昨日舊訊號)" | tee -a "$LOG"
    tg_send "⚠️ Stock_bot D pick FAILED $(date '+%F %H:%M')（非致命，誠實日報照常推），請檢查 $LOG"
fi

# 3) 公開申購抽籤掃描（誠實日報的申購機會節資料源；失敗不中斷——
#    日報會顯示最後一次成功掃描的日期，過期會標注）
echo ">>> [3/7] ipo lottery scan" | tee -a "$LOG"
if $PYTHON scripts/ipo_lottery_scan.py >> "$LOG" 2>&1; then
    echo ">>> ipo scan OK" | tee -a "$LOG"
else
    echo "!!! ipo scan FAILED (non-fatal)" | tee -a "$LOG"
fi

# 4) 訊號價版 paper NAV（forward track record；在推播前跑，日報才拿得到今日淨值；
#    失敗不中斷、只 log——NAV 是紀錄性質，明日重跑會自動補齊缺日）
echo ">>> [4/7] paper nav" | tee -a "$LOG"
if $PYTHON scripts/paper_nav.py >> "$LOG" 2>&1; then
    echo ">>> paper nav OK" | tee -a "$LOG"
else
    echo "!!! paper nav FAILED (non-fatal, 明日重跑自動補齊)" | tee -a "$LOG"
fi

# 5) Push 誠實日報到 Telegram（picks 紙上追蹤 + paper NAV + 申購機會 + 處置股 + 哨兵）
echo ">>> [5/7] telegram push daily-brief" | tee -a "$LOG"
if $PYTHON scripts/telegram_bot.py --push --strategy daily-brief >> "$LOG" 2>&1; then
    echo ">>> TG daily-brief OK" | tee -a "$LOG"
else
    echo "!!! TG daily-brief FAILED" | tee -a "$LOG"
fi

# 6) Live 特徵一致性哨兵（P0-1 型錯位偵測：picks 特徵值必須 == features 表）
#    失敗不中斷（日報已推播），但立即 TG 告警——這類 bug 曾潛伏五個月
#    （日報內的哨兵節與此為雙保險：日報 inline 抽驗 + 此處獨立告警）
echo ">>> [6/7] pick sanity sentinel" | tee -a "$LOG"
if $PYTHON scripts/reconcile_live_vs_backtest.py --sanity >> "$LOG" 2>&1; then
    echo ">>> sanity OK" | tee -a "$LOG"
else
    echo "!!! sanity MISMATCH" | tee -a "$LOG"
    tg_send "🚨 Stock_bot pick 特徵錯位（P0-1 型）$(date '+%F %H:%M')！今日訊號分數不可信，請檢查 $LOG"
fi

# 7) 月營收公告爬蟲（另人開發中；失敗不中斷、不告警、只 log——
#    爬蟲壞了只丟資料，不影響生產訊號）
echo ">>> [7/7] revenue announcements crawler" | tee -a "$LOG"
if $PYTHON scripts/crawl_revenue_announcements.py >> "$LOG" 2>&1; then
    echo ">>> revenue crawler OK" | tee -a "$LOG"
else
    echo "!!! revenue crawler FAILED (non-fatal)" | tee -a "$LOG"
fi

echo "==== $(date) END ====" | tee -a "$LOG"

# 清掉 30 天前 logs
find "$LOG_DIR" -name "daily_run_*.log" -mtime +30 -delete 2>/dev/null || true
