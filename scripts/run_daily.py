import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.db import get_session
from pipelines.daily_pipeline import run_daily_pipeline

MARKET_SOURCE_KEYS = ('INGEST_PRICES_SOURCE', 'INGEST_INSTITUTIONAL_SOURCE',
                      'INGEST_MARGIN_SHORT_SOURCE', 'INGEST_PER_SOURCE')


def configure_market_source(source):
    """Explicit CLI policy; never edit .env or switch providers after failure."""
    if source not in ('finmind', 'twse'):
        raise ValueError('Market data source must be finmind or twse')
    for key in MARKET_SOURCE_KEYS:
        os.environ[key] = source
    print(f'[daily] explicit market data source: {source}', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="執行每日 pipeline")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="跳過資料抓取（ingest），只跑 data_quality + features/labels/train/pick")
    parser.add_argument('--market-data-source', choices=('finmind', 'twse'),
                        help='Explicit provider for prices, institutions, margin and PER; overrides process settings')
    args = parser.parse_args()
    if args.market_data_source:
        configure_market_source(args.market_data_source)

    try:
        from app.file_lock import file_lock
        with file_lock(Path(__file__).resolve().parents[1] / '.cache/research-or-update.lock', timeout=0):
            from app.market_calendar import get_latest_trading_day
            load_config()
            with get_session() as session:
                end_date = get_latest_trading_day(session)
            if end_date is None:
                raise RuntimeError('No trading calendar: run the history bootstrap before the daily pipeline')
            os.environ['DAILY_DATA_END_DATE'] = end_date.isoformat()
            print(f'[daily] published data cutoff: {end_date}', flush=True)
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
