from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict

import requests


@dataclass(frozen=True)
class AIResponse:
    ok: bool
    content: str
    error: str | None = None


def call_openai_chat(
    api_key: str,
    model: str,
    prompt: str,
    timeout: int = 30,
    max_retries: int = 2,
) -> AIResponse:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }

    retry_statuses = {429, 500, 502, 503, 504}
    last_error: str | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_error = f"OpenAI request error: {exc}"
            time.sleep(1 + attempt)
            continue

        if resp.status_code in retry_statuses:
            last_error = f"OpenAI HTTP {resp.status_code}: {resp.text[:200]}"
            time.sleep(1 + attempt)
            continue

        if resp.status_code != 200:
            return AIResponse(ok=False, content="", error=f"OpenAI HTTP {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            return AIResponse(ok=False, content="", error="OpenAI response missing choices")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        return AIResponse(ok=True, content=content)

    return AIResponse(ok=False, content="", error=last_error or "OpenAI request failed")
