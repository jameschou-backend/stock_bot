from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

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
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    ai_assist_enabled: bool = False
    ai_assist_max_code_lines: int = 200
    ai_assist_max_log_lines: int = 200
    bootstrap_days: int = 365
    min_avg_volume: int = 0
    fallback_days: int = 10

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


def load_config() -> AppConfig:
    if load_dotenv is not None:
        load_dotenv(ENV_PATH, override=False)

    base = _read_yaml(CONFIG_PATH)

    def pick(key: str, default: Any) -> Any:
        env_val = _get_env(key)
        if env_val is not None:
            return env_val
        return base.get(key, default)

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
        openai_api_key=str(pick("OPENAI_API_KEY", "")),
        openai_model=str(pick("OPENAI_MODEL", "gpt-4.1-mini")),
        ai_assist_enabled=str(pick("AI_ASSIST_ENABLED", "0")) in {"1", "true", "True"},
        ai_assist_max_code_lines=int(pick("AI_ASSIST_MAX_CODE_LINES", 200)),
        ai_assist_max_log_lines=int(pick("AI_ASSIST_MAX_LOG_LINES", 200)),
        bootstrap_days=int(pick("BOOTSTRAP_DAYS", 365)),
        min_avg_volume=int(pick("MIN_AVG_VOLUME", 0)),
        fallback_days=int(pick("FALLBACK_DAYS", 10)),
    )
