from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from app.config import load_config
from app.db import get_session
from app.models import ModelVersion
from skills import (
    build_features,
    build_labels,
    daily_pick,
    ingest_institutional,
    ingest_prices,
    train_ranker,
)


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

    with get_session() as session:
        ingest_prices.run(config, session)

    with get_session() as session:
        ingest_institutional.run(config, session)

    with get_session() as session:
        build_features.run(config, session)

    with get_session() as session:
        build_labels.run(config, session)

    if _should_train(config):
        with get_session() as session:
            train_ranker.run(config, session)

    with get_session() as session:
        daily_pick.run(config, session)


if __name__ == "__main__":
    run_daily_pipeline()
