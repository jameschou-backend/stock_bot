import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.db import get_session
from pipelines.daily_pipeline import run_daily_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="執行每日 pipeline")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="跳過資料抓取（ingest），只跑 data_quality + features/labels/train/pick")
    args = parser.parse_args()

    try:
        run_daily_pipeline(skip_ingest=args.skip_ingest)
    except Exception as exc:
        from skills import ai_assist
        import platform

        config = load_config()
        with get_session() as session:
            ai_assist.run(
                config,
                session,
                job_name="pipeline",
                error=exc,
                context_files=[__file__],
                extra_context={"os": platform.platform(), "python": sys.version.split()[0]},
            )
        raise
