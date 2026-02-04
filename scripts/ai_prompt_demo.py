from __future__ import annotations

import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.db import get_session
from skills import ai_assist


def main() -> None:
    config = load_config()
    fake_error = RuntimeError("Demo error: make pipeline failed (simulated)")
    with get_session() as session:
        ai_assist.run(
            config,
            session,
            job_name="ai_prompt_demo",
            error=fake_error,
            context_files=["pipelines/daily_pipeline.py"],
            extra_context={"os": platform.platform(), "python": sys.version.split()[0]},
        )


if __name__ == "__main__":
    main()
