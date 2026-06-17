import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# 這些 env var 會切換 ingest 後端（finmind ↔ twse）。app/config.py 的
# load_config() 會 load_dotenv() 把開發者本機 .env 灌進 os.environ，使測試行為
# 取決於本機設定——例如 INGEST_PRICES_SOURCE=twse 會讓 test_ingest_probe_skip
# 走 TWSE 分支連真實 DB，造成「紅綠取決於本機 .env + DB 狀態 + 當天日期」的
# 順序依賴失敗。autouse 清除確保測試密閉；需要特定來源的測試自行
# monkeypatch.setenv 覆蓋（fixture 先 setup、測試內 setenv 後生效，不衝突）。
_INGEST_SOURCE_ENV_VARS = (
    "INGEST_PRICES_SOURCE",
    "INGEST_INSTITUTIONAL_SOURCE",
    "INGEST_MARGIN_SHORT_SOURCE",
    "INGEST_PER_SOURCE",
    "INGEST_CORPORATE_ACTIONS_SOURCE",
)


@pytest.fixture(autouse=True)
def _isolate_ingest_source_env(monkeypatch):
    for _key in _INGEST_SOURCE_ENV_VARS:
        monkeypatch.delenv(_key, raising=False)
    yield
