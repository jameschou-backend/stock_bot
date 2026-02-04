from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.db import run_migrations


def main() -> None:
    sql_path = Path(__file__).resolve().parent.parent / "storage" / "migrations" / "001_init.sql"
    run_migrations(sql_path)


if __name__ == "__main__":
    main()
