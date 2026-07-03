"""ai_assist._mask_secrets 遮罩回歸測試。

歷史 bug（2026-07-03 修正）：第二層 regex 寫成 r"...\\S+"（raw string 中雙反斜線
= 字面「反斜線+S」），導致 env-var 樣式的 secret 完全不會被遮罩。
本檔鎖定：即使 secret 值不在 config 已知清單中，只要以
KEY=value 形式出現就必須被遮罩。
"""
from app.config import AppConfig
from skills import ai_assist


def _config(**overrides) -> AppConfig:
    params = dict(
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
        api_host="127.0.0.1",
        api_port=8000,
        force_train=False,
        openai_api_key="OPENAI_SECRET",
        openai_model="gpt-4.1-mini",
        ai_assist_enabled=False,
        ai_assist_max_code_lines=20,
        ai_assist_max_log_lines=20,
    )
    params.update(overrides)
    return AppConfig(**params)


def test_mask_env_style_secrets_not_in_config():
    """secret 值不在 config 清單中（config 為空字串）時，KEY=value 樣式仍須被遮罩。"""
    config = _config(finmind_token="", openai_api_key="", db_password="x-not-present")
    text = (
        "error connecting with FINMIND_TOKEN=abc123 and "
        "TELEGRAM_BOT_TOKEN=123:AAxx something failed"
    )
    masked = ai_assist._mask_secrets(text, config)
    assert "abc123" not in masked
    assert "123:AAxx" not in masked
    assert "FINMIND_TOKEN=[REDACTED]" in masked
    assert "TELEGRAM_BOT_TOKEN=[REDACTED]" in masked


def test_mask_telegram_chat_id():
    config = _config()
    text = "env dump: TELEGRAM_CHAT_ID=987654321 end"
    masked = ai_assist._mask_secrets(text, config)
    assert "987654321" not in masked
    assert "TELEGRAM_CHAT_ID=[REDACTED]" in masked


def test_mask_all_known_env_keys():
    config = _config(finmind_token="", openai_api_key="", db_password="")
    text = "\n".join(
        [
            "OPENAI_API_KEY=sk-live-aaa",
            "DB_PASSWORD=hunter2",
            "FINMIND_TOKEN=tok111",
            "TELEGRAM_BOT_TOKEN=222:BBB",
            "TELEGRAM_CHAT_ID=333444",
        ]
    )
    masked = ai_assist._mask_secrets(text, config)
    for leaked in ("sk-live-aaa", "hunter2", "tok111", "222:BBB", "333444"):
        assert leaked not in masked
    assert masked.count("[REDACTED]") == 5


def test_mask_with_spaces_around_equals():
    config = _config(finmind_token="")
    masked = ai_assist._mask_secrets("FINMIND_TOKEN = tok-with-space", config)
    assert "tok-with-space" not in masked


def test_extract_trace_files_regex_matches():
    """歷史 bug：(\\d+) 雙反斜線使 traceback 檔案抽取永遠 match 不到。"""
    error_text = 'Traceback ...\n  File "/tmp/foo.py", line 12, in <module>\n    boom\n'
    files = ai_assist._extract_trace_files(error_text)
    assert len(files) == 1
    path, line_no = files[0]
    assert str(path) == "/tmp/foo.py"
    assert line_no == 12


def test_build_prompt_uses_real_newlines():
    """歷史 bug：多處 "\\n".join 產生字面 \\n 而非換行。"""
    ctx = ai_assist.PromptContext(
        error_text="boom",
        logs=["line1", "line2"],
        file_snippets=[],
        environment={"os": "test"},
        attempted_fixes=[],
    )
    prompt = ai_assist._build_prompt(ctx)
    assert "\\n" not in prompt
    assert "line1\nline2" in prompt
