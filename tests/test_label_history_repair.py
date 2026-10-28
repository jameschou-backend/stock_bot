from datetime import date
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.models import Base, RawPrice, Label
from skills import build_labels


def test_historical_recompute_updates_existing_labels_without_deleting_them(monkeypatch):
    # Execute the real normalization/horizon/write path against a disposable DB;
    # only translate MySQL's upsert syntax to SQLite's equivalent.
    class Upsert:
        def values(self, rows):
            self.statement = sqlite_insert(Label).values(rows)
            self.inserted = self.statement.excluded
            return self
        def on_duplicate_key_update(self, **values):
            return self.statement.on_conflict_do_update(index_elements=['stock_id','trading_date'],set_=values)
    monkeypatch.setattr(build_labels,'insert',lambda model:Upsert())
    engine = create_engine('sqlite:///:memory:'); Base.metadata.create_all(engine)
    days = [date(2026,10,d) for d in (20,21,22)]
    with Session(engine) as s:
        s.add_all(RawPrice(stock_id='2330',trading_date=d,close=p) for d,p in zip(days,[100,110,121]))
        s.add_all(Label(stock_id='2330',trading_date=d,future_ret_h=-.5) for d in days[:2]);s.commit()
        cfg = SimpleNamespace(label_horizon_days=1,use_adjusted_price=False)
        build_labels.run(cfg,s,recompute_from=days[0]);s.commit();s.expire_all()
        assert float(s.get(Label,('2330',days[0])).future_ret_h) == pytest.approx(.1)
        assert float(s.get(Label,('2330',days[1])).future_ret_h) == pytest.approx(.1)
        assert s.query(Label).count() == 2
        build_labels.run(cfg,s,recompute_from=days[0]);s.commit()
        assert s.query(Label).count() == 2
    engine.dispose()
