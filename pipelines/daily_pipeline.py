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


def run_daily_pipeline(skip_ingest: bool = False) -> None:
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

    if not skip_ingest:
        # 1. 檢查/bootstrap 歷史資料（若 DB 為空會提示執行 make backfill-10y）
        from skills import bootstrap_history
        run_skill("bootstrap_history", bootstrap_history.run)
        
        # 2. 更新股票基本資料（market/industry/security_type/is_listed）
        from skills import ingest_stock_master
        run_skill("ingest_stock_master", ingest_stock_master.run)

        # 3. 交易日曆（目前先以工作日 seed，後續可替換為官方日曆來源）
        from skills import ingest_trading_calendar
        run_skill("ingest_trading_calendar", ingest_trading_calendar.run)

        # 4. 每日價格資料
        from skills import ingest_prices
        run_skill("ingest_prices", ingest_prices.run)

        # 5. 三大法人買賣超
        from skills import ingest_institutional
        run_skill("ingest_institutional", ingest_institutional.run)

        # 6. 公司行為/還原因子（來源未接妥時會寫入 adj_factor=1 的保底資料）
        from skills import ingest_corporate_actions
        run_skill("ingest_corporate_actions", ingest_corporate_actions.run)
        
        # 7. 融資融券資料（選用，不影響核心流程）
        try:
            from skills import ingest_margin_short
            run_skill("ingest_margin_short", ingest_margin_short.run)
        except Exception as exc:
            print(f"[WARN] ingest_margin_short failed: {exc}")

        # 8. 基本面（月營收，研究用；失敗不中斷）
        try:
            from skills import ingest_fundamental
            run_skill("ingest_fundamental", ingest_fundamental.run)
        except Exception as exc:
            print(f"[WARN] ingest_fundamental failed: {exc}")

        # 9. 題材/金流（由產業聚合，研究用；失敗不中斷）
        try:
            from skills import ingest_theme_flow
            run_skill("ingest_theme_flow", ingest_theme_flow.run)
        except Exception as exc:
            print(f"[WARN] ingest_theme_flow failed: {exc}")
    else:
        print("[skip-ingest] 跳過資料抓取，直接進入 data quality check + 建置流程")

    # 8. Data Quality Check: 確保資料完整性，若不達標則 fail-fast
    from skills import data_quality
    run_skill("data_quality_check", data_quality.run)

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
