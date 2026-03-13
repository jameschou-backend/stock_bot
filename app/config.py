from __future__ import annotations

import os
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional
    load_dotenv = None


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


@dataclass(frozen=True)
class AppConfig:
    finmind_token: str
    db_dialect: str
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    topn: int
    label_horizon_days: int
    train_lookback_years: int
    schedule_time: str
    tz: str
    api_host: str
    api_port: int
    force_train: bool = False
    force_recompute_days: int = 0  # 強制刪除並重算最近 N 天特徵（0=不強制）
    min_avg_volume: int = 500
    round_trip_cost: float = 0.00585
    hold_buffer: int = 10
    backtest_folds: int = 6
    backtest_val_months: int = 3
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    ai_assist_enabled: bool = False
    ai_assist_max_code_lines: int = 200
    ai_assist_max_log_lines: int = 200
    bootstrap_days: int = 365
    min_avg_turnover: float = 0.5  # 20日平均成交值門檻（億元）
    min_amt_20: float = 5e7  # 20日平均成交值門檻（元）
    use_adjusted_price: bool = True
    enable_tradability_filter: bool = True
    fallback_days: int = 10
    # 大盤過濾器
    market_filter_enabled: bool = True
    market_filter_ma_days: int = 60
    market_filter_bear_topn: int = 5
    market_filter_weekly_drop_threshold: float = -0.03  # 週跌幅觸發危機模式
    market_filter_crisis_topn: int = 3  # 危機模式 topN
    regime_detector: str = "ma"
    # 季節性過濾器
    seasonal_weak_months: Tuple[int, ...] = (3, 10)
    seasonal_topn_multiplier: float = 0.5
    # 停損冷卻機制
    stoploss_cooldown_weeks: int = 3
    # 回測基礎設定
    stoploss_pct: float = -0.07
    transaction_cost_pct: float = 0.001425
    strategy_weights_bull: dict = None
    strategy_weights_bear: dict = None
    # 回測進階設定
    backtest_entry_delay_days: int = 1
    backtest_risk_free_rate: float = 0.015
    backtest_benchmark_with_cost: bool = True
    backtest_position_sizing: str = "vol_inverse"
    backtest_trailing_stop_pct: float = -0.12
    backtest_atr_stoploss_multiplier: float = None
    backtest_atr_period: int = 14
    # 過熱過濾器
    enable_overheat_filter: bool = False
    overheat_rsi_threshold: float = 75.0
    # 突破確認進場（F+ 策略，生產選股標記用）
    enable_breakthrough_entry: bool = True  # 是否標記今日突破狀態並輸出三清單
    # API Rate Limiting
    finmind_requests_per_hour: int = 6000  # FinMind API 每小時請求限制
    chunk_days: int = 180  # 資料抓取每個 chunk 的天數
    institutional_bulk_chunk_days: int = 90  # 法人全市場抓取建議 chunk 天數
    margin_bulk_chunk_days: int = 90  # 融資融券全市場抓取建議 chunk 天數
    finmind_retry_max: int = 3  # FinMind API 最大重試次數
    finmind_retry_backoff: float = 1.0  # FinMind API 重試退避秒數
    # Backfill 設定
    backfill_years: int = 10  # 完整 backfill 年數
    # Data Quality 門檻（比例 + 最小值雙門檻）
    dq_min_stocks_prices: int = 1200  # raw_prices 每日最小股票數
    dq_min_stocks_institutional: int = 800  # raw_institutional 每日最小股票數
    dq_min_stocks_margin: int = 800  # raw_margin_short 每日最小股票數
    dq_coverage_ratio_prices: float = 0.7  # raw_prices 覆蓋率門檻
    dq_coverage_ratio_institutional: float = 0.5  # raw_institutional 覆蓋率門檻
    dq_coverage_ratio_margin: float = 0.5  # raw_margin_short 覆蓋率門檻
    dq_max_lag_trading_days: int = 1  # 允許最大落後交易日數
    dq_max_stale_calendar_days: int = 5  # 允許最大落後日曆天數
    eval_topk_list: Tuple[int, ...] = (10, 20)
    dashboard_show_ml: bool = False
    data_quality_mode: str = "strict"
    selection_mode: str = "model"
    multi_agent_topn: int = 20
    multi_agent_weights: dict = None

    @property
    def db_url(self) -> str:
        dialect = self.db_dialect
        if dialect == "mysql":
            dialect = "mysql+pymysql"
        return (
            f"{dialect}://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml must be a mapping")
    return data


def _get_env(key: str) -> str | None:
    value = os.getenv(key)
    if value is None:
        return None
    if value.strip() == "":
        return None
    return value


def _parse_eval_topk_list(raw_value: Any) -> Tuple[int, ...]:
    if raw_value is None:
        return (10, 20)
    if isinstance(raw_value, (list, tuple)):
        values = [int(v) for v in raw_value if int(v) > 0]
        return tuple(values) if values else (10, 20)

    text = str(raw_value).strip()
    if not text:
        return (10, 20)
    parts = [p.strip() for p in text.split(",")]
    values = [int(p) for p in parts if p]
    values = [v for v in values if v > 0]
    return tuple(values) if values else (10, 20)


def load_config() -> AppConfig:
    if load_dotenv is not None:
        load_dotenv(ENV_PATH, override=False)

    base = _read_yaml(CONFIG_PATH)

    def pick(key: str, default: Any) -> Any:
        env_val = _get_env(key)
        if env_val is not None:
            return env_val
        return base.get(key, default)

    def pick_json(key: str, default: dict) -> dict:
        value = pick(key, default)
        if isinstance(value, dict):
            return value
        if value is None or value == "":
            return default
        return json.loads(value)

    return AppConfig(
        finmind_token=str(pick("FINMIND_TOKEN", "")),
        db_dialect=str(pick("DB_DIALECT", "mysql")),
        db_host=str(pick("DB_HOST", "127.0.0.1")),
        db_port=int(pick("DB_PORT", 3307)),
        db_name=str(pick("DB_NAME", "stock_bot")),
        db_user=str(pick("DB_USER", "")),
        db_password=str(pick("DB_PASSWORD", "")),
        topn=int(pick("TOPN", 20)),
        label_horizon_days=int(pick("LABEL_HORIZON_DAYS", 20)),
        train_lookback_years=int(pick("TRAIN_LOOKBACK_YEARS", 5)),
        schedule_time=str(pick("SCHEDULE_TIME", "16:50")),
        tz=str(pick("TZ", "Asia/Taipei")),
        api_host=str(pick("API_HOST", "0.0.0.0")),
        api_port=int(pick("API_PORT", 8000)),
        force_train=str(pick("FORCE_TRAIN", "0")) in {"1", "true", "True"},
        force_recompute_days=int(pick("FORCE_RECOMPUTE_DAYS", 0)),
        min_avg_volume=int(pick("MIN_AVG_VOLUME", 500)),
        round_trip_cost=float(pick("ROUND_TRIP_COST", 0.00585)),
        hold_buffer=int(pick("HOLD_BUFFER", 10)),
        backtest_folds=int(pick("BACKTEST_FOLDS", 6)),
        backtest_val_months=int(pick("BACKTEST_VAL_MONTHS", 3)),
        openai_api_key=str(pick("OPENAI_API_KEY", "")),
        openai_model=str(pick("OPENAI_MODEL", "gpt-4.1-mini")),
        ai_assist_enabled=str(pick("AI_ASSIST_ENABLED", "0")) in {"1", "true", "True"},
        ai_assist_max_code_lines=int(pick("AI_ASSIST_MAX_CODE_LINES", 200)),
        ai_assist_max_log_lines=int(pick("AI_ASSIST_MAX_LOG_LINES", 200)),
        bootstrap_days=int(pick("BOOTSTRAP_DAYS", 365)),
        min_avg_turnover=float(pick("MIN_AVG_TURNOVER", 0.5)),
        min_amt_20=float(pick("MIN_AMT_20", 5e7)),
        use_adjusted_price=str(pick("USE_ADJUSTED_PRICE", "true")).lower() in {"1", "true"},
        enable_tradability_filter=str(pick("ENABLE_TRADABILITY_FILTER", "true")).lower() in {"1", "true"},
        fallback_days=int(pick("FALLBACK_DAYS", 10)),
        market_filter_enabled=str(pick("MARKET_FILTER_ENABLED", "true")).lower() in {"1", "true"},
        market_filter_ma_days=int(pick("MARKET_FILTER_MA_DAYS", 60)),
        market_filter_bear_topn=int(pick("MARKET_FILTER_BEAR_TOPN", 5)),
        market_filter_weekly_drop_threshold=float(pick("MARKET_FILTER_WEEKLY_DROP_THRESHOLD", -0.03)),
        market_filter_crisis_topn=int(pick("MARKET_FILTER_CRISIS_TOPN", 3)),
        regime_detector=str(pick("REGIME_DETECTOR", "ma")),
        seasonal_weak_months=_parse_eval_topk_list(pick("SEASONAL_WEAK_MONTHS", "3,10")),
        seasonal_topn_multiplier=float(pick("SEASONAL_TOPN_MULTIPLIER", 0.5)),
        stoploss_cooldown_weeks=int(pick("STOPLOSS_COOLDOWN_WEEKS", 3)),
        stoploss_pct=float(pick("STOPLOSS_PCT", -0.07)),
        transaction_cost_pct=float(pick("TRANSACTION_COST_PCT", 0.001425)),
        strategy_weights_bull=pick_json("STRATEGY_WEIGHTS_BULL", {}),
        strategy_weights_bear=pick_json("STRATEGY_WEIGHTS_BEAR", {}),
        finmind_requests_per_hour=int(pick("FINMIND_REQUESTS_PER_HOUR", 6000)),
        chunk_days=int(pick("CHUNK_DAYS", 180)),
        institutional_bulk_chunk_days=int(pick("INSTITUTIONAL_BULK_CHUNK_DAYS", 90)),
        margin_bulk_chunk_days=int(pick("MARGIN_BULK_CHUNK_DAYS", 90)),
        finmind_retry_max=int(pick("FINMIND_RETRY_MAX", 3)),
        finmind_retry_backoff=float(pick("FINMIND_RETRY_BACKOFF", 1.0)),
        backfill_years=int(pick("BACKFILL_YEARS", 10)),
        dq_min_stocks_prices=int(pick("DQ_MIN_STOCKS_PRICES", 1200)),
        dq_min_stocks_institutional=int(pick("DQ_MIN_STOCKS_INSTITUTIONAL", 800)),
        dq_min_stocks_margin=int(pick("DQ_MIN_STOCKS_MARGIN", 800)),
        dq_coverage_ratio_prices=float(pick("DQ_COVERAGE_RATIO_PRICES", 0.7)),
        dq_coverage_ratio_institutional=float(pick("DQ_COVERAGE_RATIO_INSTITUTIONAL", 0.5)),
        dq_coverage_ratio_margin=float(pick("DQ_COVERAGE_RATIO_MARGIN", 0.5)),
        dq_max_lag_trading_days=int(pick("DQ_MAX_LAG_TRADING_DAYS", 1)),
        dq_max_stale_calendar_days=int(pick("DQ_MAX_STALE_CALENDAR_DAYS", 5)),
        eval_topk_list=_parse_eval_topk_list(pick("EVAL_TOPK_LIST", "10,20")),
        dashboard_show_ml=str(pick("DASHBOARD_SHOW_ML", "0")).lower() in {"1", "true"},
        data_quality_mode=str(pick("DATA_QUALITY_MODE", "strict")).lower(),
        selection_mode=str(pick("SELECTION_MODE", "model")).lower(),
        multi_agent_topn=int(pick("MULTI_AGENT_TOPN", 20)),
        multi_agent_weights=pick_json(
            "MULTI_AGENT_WEIGHTS",
            {"tech": 0.35, "flow": 0.30, "margin": 0.10, "fund": 0.15, "theme": 0.10},
        ),
        backtest_entry_delay_days=int(pick("BACKTEST_ENTRY_DELAY_DAYS", 1)),
        backtest_risk_free_rate=float(pick("BACKTEST_RISK_FREE_RATE", 0.015)),
        backtest_benchmark_with_cost=str(pick("BACKTEST_BENCHMARK_WITH_COST", "true")).lower() in {"1", "true"},
        backtest_position_sizing=str(pick("BACKTEST_POSITION_SIZING", "equal")).lower(),
        backtest_trailing_stop_pct=float(pick("BACKTEST_TRAILING_STOP_PCT", "nan")) if pick("BACKTEST_TRAILING_STOP_PCT", None) not in {None, "null", "None", ""} else None,
        backtest_atr_stoploss_multiplier=float(pick("BACKTEST_ATR_STOPLOSS_MULTIPLIER", "nan")) if pick("BACKTEST_ATR_STOPLOSS_MULTIPLIER", None) not in {None, "null", "None", ""} else None,
        backtest_atr_period=int(pick("BACKTEST_ATR_PERIOD", 14)),
        enable_overheat_filter=str(pick("ENABLE_OVERHEAT_FILTER", "false")).lower() in {"1", "true"},
        overheat_rsi_threshold=float(pick("OVERHEAT_RSI_THRESHOLD", 75.0)),
        enable_breakthrough_entry=str(pick("ENABLE_BREAKTHROUGH_ENTRY", "true")).lower() in {"1", "true"},
    )
