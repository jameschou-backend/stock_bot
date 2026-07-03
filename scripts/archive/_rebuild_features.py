#!/usr/bin/env python3
"""Full 10-year feature rebuild script."""
import os
import sys

# Must set env var BEFORE importing config
os.environ["FORCE_RECOMPUTE_DAYS"] = "3650"

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.db import get_session
from skills.build_features import run as run_build_features

if __name__ == "__main__":
    config = load_config()
    print(f"force_recompute_days from config: {config.force_recompute_days}")
    with get_session() as session:
        result = run_build_features(config, session)
    print(f"Done: {result}")
