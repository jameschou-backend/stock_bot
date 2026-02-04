from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import textwrap
import traceback
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

from sqlalchemy.orm import Session

from app.ai_client import AIResponse, call_openai_chat
from app.config import AppConfig
from app.job_utils import finish_job, start_job


ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
PROMPTS_DIR = ARTIFACTS_DIR / "ai_prompts"
ANSWERS_DIR = ARTIFACTS_DIR / "ai_answers"


@dataclass(frozen=True)
class PromptContext:
    error_text: str
    logs: List[str]
    file_snippets: List[str]
    environment: Dict[str, str]
    attempted_fixes: List[str]


def _mask_secrets(text: str, config: AppConfig) -> str:
    secrets = [
        config.openai_api_key,
        config.db_password,
        config.finmind_token,
    ]
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(OPENAI_API_KEY|DB_PASSWORD|FINMIND_TOKEN)\s*=\s*\\S+", r"\\1=[REDACTED]", text)
    return text


def _extract_trace_files(error_text: str) -> List[Tuple[Path, Optional[int]]]:
    matches = re.findall(r'File "([^"]+)", line (\\d+)', error_text)
    files: List[Tuple[Path, Optional[int]]] = []
    for path_str, line_str in matches:
        path = Path(path_str)
        line_no = int(line_str)
        files.append((path, line_no))
    return files


def _read_snippet(path: Path, line_no: Optional[int], max_lines: int) -> List[str]:
    if not path.exists() or not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    if line_no is None or line_no <= 0:
        return lines[:max_lines]

    start = max(line_no - 6, 1)
    end = min(line_no + 6, len(lines))
    snippet = lines[start - 1 : end]
    return snippet[:max_lines]


def _collect_file_snippets(
    context_files: Sequence[str],
    error_text: str,
    max_total_lines: int,
) -> List[str]:
    snippets: List[str] = []
    seen: set[Path] = set()
    remaining = max_total_lines

    for path_str in context_files:
        path = Path(path_str)
        if path in seen:
            continue
        seen.add(path)
        if remaining <= 0:
            break
        content = _read_snippet(path, None, min(remaining, 20))
        if not content:
            continue
        snippets.append(f"### {path}\\n```\\n" + "\\n".join(content) + "\\n```")
        remaining -= len(content)

    for path, line_no in _extract_trace_files(error_text):
        if path in seen:
            continue
        if remaining <= 0:
            break
        content = _read_snippet(path, line_no, min(remaining, 20))
        if not content:
            continue
        header = f"### {path} (line {line_no})"
        snippets.append(header + "\\n```\\n" + "\\n".join(content) + "\\n```")
        remaining -= len(content)

    return snippets


def _build_prompt(context: PromptContext) -> str:
    sections = [
        "# 目標",
        "我想讓 make pipeline 成功。",
        "",
        "# 實際錯誤",
        context.error_text.strip(),
        "",
        "# 最後 N 行 logs",
        "\\n".join(context.logs) if context.logs else "（無）",
        "",
        "# 環境",
        "\\n".join(f"- {k}: {v}" for k, v in context.environment.items()),
        "",
        "# 已嘗試的修復",
        "\\n".join(f"- {item}" for item in context.attempted_fixes) if context.attempted_fixes else "- 無",
        "",
        "# 相關檔案片段",
        "\\n\\n".join(context.file_snippets) if context.file_snippets else "（無）",
        "",
        "# 問題",
        "請給最小修改的修復建議，並附上具體 patch（或檔案/函式級指引），",
        "同時列出可執行的 TODO 清單。",
    ]
    return "\\n".join(sections).strip() + "\\n"


def _truncate_lines(lines: Iterable[str], max_lines: int) -> List[str]:
    result: List[str] = []
    for line in lines:
        if len(result) >= max_lines:
            break
        result.append(line)
    return result


def _extract_todos(answer: str, max_items: int = 10) -> List[str]:
    todos: List[str] = []
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.startswith("- [ ]") or stripped.startswith("TODO"):
            todos.append(stripped)
        if len(todos) >= max_items:
            break
    return todos


def run(
    config: AppConfig,
    db_session: Optional[Session],
    job_name: str,
    error: Exception | str,
    context_files: Sequence[str] | None = None,
    extra_context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    job_id = uuid4().hex
    if db_session is not None:
        job_id = start_job(db_session, f"ai_assist:{job_name}")

    error_text = str(error)
    if isinstance(error, Exception):
        error_text = "".join(traceback.format_exception(type(error), error, error.__traceback__))

    extra_context = extra_context or {}
    logs_lines = extra_context.get("logs", [])
    if isinstance(logs_lines, str):
        logs_lines = logs_lines.splitlines()
    logs_lines = _truncate_lines(list(logs_lines), config.ai_assist_max_log_lines)

    env_info = {
        "os": extra_context.get("os", "unknown"),
        "python": extra_context.get("python", "unknown"),
        "db": f"{config.db_dialect}://{config.db_host}:{config.db_port}/{config.db_name}",
        "ai_enabled": str(config.ai_assist_enabled),
        "openai_model": config.openai_model or "",
    }

    file_snippets = _collect_file_snippets(
        context_files or [],
        error_text,
        config.ai_assist_max_code_lines,
    )

    prompt_context = PromptContext(
        error_text=error_text,
        logs=logs_lines,
        file_snippets=file_snippets,
        environment=env_info,
        attempted_fixes=list(extra_context.get("attempted_fixes", [])),
    )

    prompt = _build_prompt(prompt_context)
    prompt = _mask_secrets(prompt, config)

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    prompt_path = PROMPTS_DIR / f"{job_id}.md"
    prompt_path.write_text(prompt, encoding="utf-8")

    answer_path: Path | None = None
    answer_summary = "prompt_generated"
    todos: List[str] = []
    response: AIResponse | None = None
    if config.ai_assist_enabled and config.openai_api_key:
        response = call_openai_chat(config.openai_api_key, config.openai_model, prompt)
        answer_text = response.content if response.ok else response.error or "OpenAI request failed"
        answer_text = _mask_secrets(answer_text, config)
        ANSWERS_DIR.mkdir(parents=True, exist_ok=True)
        answer_path = ANSWERS_DIR / f"{job_id}.md"
        answer_path.write_text(answer_text, encoding="utf-8")
        answer_summary = answer_text[:200]
        todos = _extract_todos(answer_text)

    logs = {
        "prompt_path": str(prompt_path),
        "answer_path": str(answer_path) if answer_path else None,
        "summary": answer_summary,
        "todos": todos,
        "job_name": job_name,
    }

    if db_session is not None:
        finish_job(db_session, job_id, "success", logs=logs)

    return logs
