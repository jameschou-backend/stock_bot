from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app.config import load_config
from app.db import get_session
from app.models import ModelVersion


def _check_prices_exist(min_rows: int = 100) -> bool:
    """驗證 raw_prices 表至少有 min_rows 筆近期資料（7 天內），確保 ingest_prices 成功執行。"""
    try:
        from sqlalchemy import func, select
        from app.models import RawPrice
        cutoff = (datetime.now() - timedelta(days=7)).date()
        with get_session() as session:
            cnt = session.execute(
                select(func.count()).select_from(RawPrice).where(RawPrice.trading_date >= cutoff)
            ).scalar() or 0
        return int(cnt) >= min_rows
    except Exception:
        return False


def _check_features_exist(min_rows: int = 50) -> bool:
    """驗證特徵資料至少有 min_rows 筆近期資料（7 天內），確保 build_features 成功執行。

    優先檢查 Parquet FeatureStore（精確且快速），fallback 至 MySQL。
    """
    cutoff = (datetime.now() - timedelta(days=7)).date()

    # ── 優先：Parquet FeatureStore ──
    try:
        from skills.feature_store import FeatureStore
        _fs = FeatureStore()
        _max_date = _fs.get_max_date()
        if _max_date is not None and _max_date >= cutoff:
            _sample = _fs.read(cutoff, _max_date)
            return len(_sample) >= min_rows
    except Exception:
        pass

    # ── Fallback：MySQL ──
    try:
        from sqlalchemy import func, select
        from app.models import Feature
        with get_session() as session:
            cnt = session.execute(
                select(func.count()).select_from(Feature).where(Feature.trading_date >= cutoff)
            ).scalar() or 0
        return int(cnt) >= min_rows
    except Exception:
        return False


def _check_labels_exist(min_rows: int = 50) -> bool:
    """驗證 labels 表至少有 min_rows 筆近期資料（30 天內），確保 build_labels 成功執行。"""
    try:
        from sqlalchemy import func, select
        from app.models import Label
        cutoff = (datetime.now() - timedelta(days=30)).date()
        with get_session() as session:
            cnt = session.execute(
                select(func.count()).select_from(Label).where(Label.trading_date >= cutoff)
            ).scalar() or 0
        return int(cnt) >= min_rows
    except Exception:
        return False


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
        
        def _run_optional_skill(skill_name: str, runner) -> None:
            """執行選用 skill，失敗時寫入 job 記錄但不中斷主流程。"""
            try:
                run_skill(skill_name, runner)
            except Exception as exc:
                print(f"[WARN] {skill_name} failed: {exc}")
                try:
                    from app.job_utils import finish_job, start_job
                    with get_session() as session:
                        _jid = start_job(session, skill_name)
                        finish_job(session, _jid, "failed", error_text=str(exc), logs={"error": str(exc), "optional": True})
                except Exception:
                    pass  # DB logging 失敗不阻斷主流程

        # 7. 融資融券資料（選用，不影響核心流程）
        from skills import ingest_margin_short
        _run_optional_skill("ingest_margin_short", ingest_margin_short.run)

        # 8. 基本面（月營收，研究用；失敗不中斷）
        from skills import ingest_fundamental
        _run_optional_skill("ingest_fundamental", ingest_fundamental.run)

        # 9. 題材/金流（由產業聚合，研究用；失敗不中斷）
        from skills import ingest_theme_flow
        _run_optional_skill("ingest_theme_flow", ingest_theme_flow.run)
    else:
        print("[skip-ingest] 跳過資料抓取，直接進入 data quality check + 建置流程")

    # 8. Data Quality Check: 確保資料完整性，若不達標則 fail-fast
    from skills import data_quality
    run_skill("data_quality_check", data_quality.run)

    # ── Checkpoint 1：price 資料驗證 ──
    # 確保 ingest_prices 確實寫入資料，避免因 API 靜默失敗導致後續特徵計算在空資料上運行
    if not skip_ingest and not _check_prices_exist():
        raise RuntimeError(
            "[pipeline checkpoint] ingest_prices 後 raw_prices 近 7 天資料不足 100 筆，"
            "請確認 FinMind API 是否正常。可跑 make backfill-10y 補齊。"
        )

    from skills import build_features
    run_skill("build_features", build_features.run)

    # ── Checkpoint 2：feature 資料驗證 ──
    if not _check_features_exist():
        raise RuntimeError(
            "[pipeline checkpoint] build_features 後 features 近 7 天資料不足 50 筆，"
            "特徵計算可能失敗，請檢查 build_features 日誌。"
        )

    from skills import build_labels
    run_skill("build_labels", build_labels.run)

    # ── Checkpoint 3：label 資料驗證 ──
    if not _check_labels_exist():
        raise RuntimeError(
            "[pipeline checkpoint] build_labels 後 labels 近 30 天資料不足 50 筆，"
            "標籤計算可能失敗，請檢查 build_labels 日誌。"
        )

    if _should_train(config):
        from skills import train_ranker
        run_skill("train_ranker", train_ranker.run)

    from skills import daily_pick
    run_skill("daily_pick", daily_pick.run)

    from skills import export_report
    run_skill("export_report", export_report.run)


if __name__ == "__main__":
    run_daily_pipeline()
