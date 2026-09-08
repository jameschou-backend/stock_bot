from types import SimpleNamespace
import pytest
from fastapi import HTTPException
from app.workbench_api import require_local


def test_workbench_only_accepts_loopback_clients():
    for host in ('127.0.0.1','::1'):
        require_local(SimpleNamespace(client=SimpleNamespace(host=host)))
    for host in ('192.168.1.12','203.0.113.1','unresolved'):
        with pytest.raises(HTTPException) as exc:
            require_local(SimpleNamespace(client=SimpleNamespace(host=host)))
        assert exc.value.status_code==403
