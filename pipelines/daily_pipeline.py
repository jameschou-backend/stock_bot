from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from app.config import load_config
from app.db import get_session
from app.models import ModelVersion


def _should_train(config) -> bool:
    if config.force_train:
        return True

    with get_session() as session:
        latest = session.query(ModelVersion).order_by(ModelVersion.created_at.desc()).first()
        if latest is None:
            return True

    today = datetime.now(ZoneInfo(config.tz)).date()
    return today.weekday() == 0


def run_daily_pipeline() -> None:
    config = load_config()

    def run_skill(skill_name, runner):
        try:
            with get_session() as session:
                return runner(config, session)
        except Exception as exc:
            import inspect
            import platform
            import sys
            from skills import ai_assist

            with get_session() as session:
                runner_path = inspect.getsourcefile(runner)
                context_files = [__file__]
                if runner_path:
                    context_files.append(runner_path)
                ai_assist.run(
                    config,
                    session,
                    job_name=skill_name,
                    error=exc,
                    context_files=context_files,
                    extra_context={
                        "os": platform.platform(),
                        "python": sys.version.split()[0],
                    },
                )
            raise

    # 延遲匯入技能模組，避免啟動時載入不必要依賴，並降低 import-time 失敗風險。
    from skills import ingest_prices
    run_skill("ingest_prices", ingest_prices.run)

    from skills import ingest_institutional
    run_skill("ingest_institutional", ingest_institutional.run)

    from skills import build_features
    run_skill("build_features", build_features.run)

    from skills import build_labels
    run_skill("build_labels", build_labels.run)

    if _should_train(config):
        from skills import train_ranker
        run_skill("train_ranker", train_ranker.run)

    from skills import daily_pick
    run_skill("daily_pick", daily_pick.run)

    from skills import export_report
    run_skill("export_report", export_report.run)


if __name__ == "__main__":
    run_daily_pipeline()
