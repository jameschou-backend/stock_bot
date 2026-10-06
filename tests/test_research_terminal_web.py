from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import research_terminal_web as web
from app.workbench_api import require_local


def client(local=True):
    app = FastAPI()
    app.include_router(web.router)
    if local:
        app.dependency_overrides[require_local] = lambda: None
    return TestClient(app)


def test_application_assets_are_local_only_and_do_not_expose_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(web, 'ASSETS', tmp_path)
    (tmp_path / 'index.html').write_text('<h1>股票工作台</h1>')
    (tmp_path / 'app.js').write_text('const ready=true;')
    (tmp_path / '.env').write_text('not-public')
    with client() as c:
        response = c.get('/terminal/')
        assert response.status_code == 200
        assert '股票工作台' in response.text
        assert "script-src 'self'" in response.headers['Content-Security-Policy']
        assert response.headers['Cache-Control'] == 'no-store'
        assert c.get('/terminal/app.js').headers['content-type'].startswith('text/javascript')
        assert c.get('/terminal/.env').status_code == 404
        assert c.get('/terminal/private/report.json').status_code == 404
        assert c.get('/terminal/app.css').status_code == 503
        assert c.get('/terminal', follow_redirects=False).headers['location'] == '/terminal/'
    with client(local=False) as c:
        assert c.get('/terminal/').status_code == 403
        assert c.get('/terminal/app.js').status_code == 403
