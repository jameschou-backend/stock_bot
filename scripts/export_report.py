from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.db import get_session
from skills import export_report


def main() -> None:
    config = load_config()
    with get_session() as session:
        export_report.run(config, session)


if __name__ == "__main__":
    main()
