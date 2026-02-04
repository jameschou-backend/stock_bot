from pathlib import Path

from app.db import run_migrations


def main() -> None:
    sql_path = Path(__file__).resolve().parent.parent / "storage" / "migrations" / "001_init.sql"
    run_migrations(sql_path)


if __name__ == "__main__":
    main()
