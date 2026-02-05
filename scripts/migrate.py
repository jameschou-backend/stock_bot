from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.db import run_migrations


def main() -> None:
    migrations_dir = Path(__file__).resolve().parent.parent / "storage" / "migrations"
    
    # 按照檔名順序執行所有 migration
    migration_files = sorted(migrations_dir.glob("*.sql"))
    
    for sql_path in migration_files:
        print(f"Running migration: {sql_path.name}")
        run_migrations(sql_path)


if __name__ == "__main__":
    main()
