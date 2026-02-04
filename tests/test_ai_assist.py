from pathlib import Path

import pytest

from app.config import AppConfig
from skills import ai_assist


def _config() -> AppConfig:
    return AppConfig(
        finmind_token="FINMIND_SECRET",
        db_dialect="mysql",
        db_host="127.0.0.1",
        db_port=3307,
        db_name="stock_bot",
        db_user="user",
        db_password="DB_SECRET",
        topn=20,
        label_horizon_days=20,
        train_lookback_years=5,
        schedule_time="16:50",
        tz="Asia/Taipei",
        api_host="0.0.0.0",
        api_port=8000,
        force_train=False,
        openai_api_key="OPENAI_SECRET",
        openai_model="gpt-4.1-mini",
        ai_assist_enabled=False,
        ai_assist_max_code_lines=20,
        ai_assist_max_log_lines=20,
    )


def test_mask_secrets():
    config = _config()
    text = "OPENAI_API_KEY=OPENAI_SECRET DB_PASSWORD=DB_SECRET FINMIND_TOKEN=FINMIND_SECRET"
    masked = ai_assist._mask_secrets(text, config)
    assert "OPENAI_SECRET" not in masked
    assert "DB_SECRET" not in masked
    assert "FINMIND_SECRET" not in masked
    assert "OPENAI_API_KEY=[REDACTED]" in masked


def test_prompt_file_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    prompts_dir = tmp_path / "ai_prompts"
    answers_dir = tmp_path / "ai_answers"
    monkeypatch.setattr(ai_assist, "PROMPTS_DIR", prompts_dir)
    monkeypatch.setattr(ai_assist, "ANSWERS_DIR", answers_dir)

    sample = tmp_path / "sample.py"
    sample.write_text("raise ValueError('boom')\n", encoding="utf-8")

    logs = ai_assist.run(
        _config(),
        None,
        job_name="test_job",
        error="boom",
        context_files=[str(sample)],
        extra_context={"logs": ["line1", "line2"]},
    )
    prompt_path = Path(logs["prompt_path"])
    assert prompt_path.exists()
