from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.db import get_session


def main() -> int:
    result = subprocess.run(["pytest", "-q"], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout, end="")
        return 0

    error_text = "\n".join([result.stdout, result.stderr]).strip()
    config = load_config()
    with get_session() as session:
        from skills import ai_assist

        ai_assist.run(
            config,
            session,
            job_name="make_test",
            error=error_text or "pytest failed",
            context_files=["tests"],
            extra_context={
                "os": platform.platform(),
                "python": sys.version.split()[0],
                "logs": error_text.splitlines(),
            },
        )
    print(error_text, end="")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
