from __future__ import annotations

import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

from app.config import load_config
from app.db import get_session


def main() -> None:
    config = load_config()
    try:
        uvicorn.run(
            "app.api:app",
            host=config.api_host,
            port=config.api_port,
            log_level="info",
        )
    except Exception as exc:
        from skills import ai_assist

        with get_session() as session:
            ai_assist.run(
                config,
                session,
                job_name="make_api",
                error=exc,
                context_files=["app/api.py"],
                extra_context={"os": platform.platform(), "python": sys.version.split()[0]},
            )
        raise


if __name__ == "__main__":
    main()
