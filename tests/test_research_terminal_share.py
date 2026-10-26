import http.client
from http.server import ThreadingHTTPServer
from pathlib import Path
import re
import threading

from scripts.serve_research_terminal_share import allowed, Share


def test_share_allows_every_script_and_stylesheet_required_by_terminal():
    index = Path(__file__).resolve().parents[1] / 'ui/research_terminal/index.html'
    assets = re.findall(r'(?:src|href)="(/terminal/[^"?#]+\.(?:js|css))"', index.read_text())
    assert assets
    blocked = [path for path in assets if not allowed(path)]
    assert not blocked, f'Terminal assets blocked by read-only share: {blocked}'


def test_share_allows_only_public_research_and_saved_comparison_reads():
    for path in ('/terminal/account-comparison.js','/terminal/top3-validation.js','/terminal/legacy-comparison.js','/research-terminal/api/archive',
                 '/research-terminal/api/archive/legacy_return_comparison_20261012/cases/liquid_universe_baseline',
                 '/research-terminal/api/archive/top3_validation_20261012/cases/obv20_breakout_combined',
                 '/research-terminal/api/archive/all_entry_accounts_20261011/cases/momentum',
                 '/research-terminal/api/stocks/2330'):
        assert allowed(path)
    for path in ('/workbench/portfolio','/health','/research-terminal/api/backtests',
                 '/terminal/.env','/terminal/../app.py','/terminal/poc-private.js',
                 '/terminal/poc-drawdown.js.map','/research-terminal/api/archive/../secrets',
                 '/research-terminal/api/archive/%2e%2e','/research-terminal/api/archive/study/other'):
        assert not allowed(path)


def test_share_denies_job_submission_and_private_reads_before_upstream():
    server=ThreadingHTTPServer(('127.0.0.1',0),Share)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    connection=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=3)
    try:
        for method,path in [('POST','/research-terminal/api/backtests'),('GET','/workbench/portfolio'),
                            ('PUT','/research-terminal/api/archive'),('GET','/.env')]:
            connection.request(method,path)
            response=connection.getresponse();assert response.status==403;response.read()
    finally:
        connection.close();server.shutdown();server.server_close();thread.join(timeout=2)
