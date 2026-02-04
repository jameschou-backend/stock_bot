from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.daily_pipeline import run_daily_pipeline


if __name__ == "__main__":
    run_daily_pipeline()
