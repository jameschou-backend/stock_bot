"""Optional metadata mirror in a dedicated US MySQL instance; no Taiwan imports."""
import json
import os
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from .config import REPO


def engine_from_file(path):
    path = Path(path).expanduser().absolute()
    if path.is_symlink() or path.resolve().is_relative_to(REPO) or path.stat().st_mode & 0o077:
        raise ValueError("US DB configuration must be a mode-600 file outside the repository")
    cfg = json.loads(path.read_text())
    if (cfg.get("host") != "127.0.0.1" or cfg.get("port") != 3308
            or cfg.get("database") != "stock_bot_us" or cfg.get("user") != "us_market"):
        raise ValueError("US catalog requires its dedicated 127.0.0.1:3308/stock_bot_us account")
    return create_engine(URL.create("mysql+pymysql",username=cfg["user"],password=cfg["password"],
                                    host=cfg["host"],port=cfg["port"],database=cfg["database"]),
                         pool_size=1,max_overflow=0,connect_args={"connect_timeout":5})


def record(engine, snapshot, ref):
    """No auto migrations. A missing US schema is an explicit setup error."""
    with engine.begin() as conn:
        if conn.execute(text("SELECT DATABASE()")).scalar() != "stock_bot_us":
            raise ValueError("US catalog database mismatch")
        old = conn.execute(text("SELECT manifest_sha256 FROM ingest_snapshots WHERE snapshot_id=:id"),
                           {"id":snapshot["snapshot_id"]}).scalar()
        if old is not None:
            if old != ref["sha256"]:
                raise ValueError("US catalog manifest conflict")
            return
        conn.execute(text("INSERT INTO ingest_snapshots (snapshot_id,dataset,ticker,first_date,last_date,row_count,manifest_path,manifest_sha256) VALUES (:id,:dataset,:ticker,:start,:end,:rows,:path,:sha)"),
                     dict(id=snapshot["snapshot_id"],dataset=snapshot["query"]["dataset"],
                          ticker=snapshot["query"]["ticker"],start=snapshot["query"]["start"],
                          end=snapshot["query"]["end"],rows=snapshot["rows"],path=ref["path"],sha=ref["sha256"]))
