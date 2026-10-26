import gzip
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading

import pytest

from scripts import serve_research_terminal_share as share
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


@pytest.fixture
def public_share(monkeypatch):
    assets = Path(__file__).resolve().parents[1] / 'ui/research_terminal'
    responses = {
        '/terminal/app.js': (assets.joinpath('app.js').read_bytes(), 'text/javascript'),
        '/terminal/': (b'<body>' + b'<p>research</p>' * 200 + b'</body>', 'text/html'),
        '/research-terminal/api/archive': (
            json.dumps({'rows': [{'stock_id': '2330', 'date': '2026-10-14'}] * 200}).encode(),
            'application/json'),
        '/research-terminal/api/overview': (b'{"ready":true}', 'application/json'),
        '/terminal/app.css': (b'not text' * 500, 'application/octet-stream'),
    }
    requests = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append(self.path)
            body, content_type = responses[self.path]
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    upstream = ThreadingHTTPServer(('127.0.0.1', 0), Upstream)
    monkeypatch.setattr(share, 'UPSTREAM', f'http://127.0.0.1:{upstream.server_port}')
    proxy = ThreadingHTTPServer(('127.0.0.1', 0), Share)
    servers = [upstream, proxy]
    threads = [threading.Thread(target=s.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
               for s in servers]
    for thread in threads:
        thread.start()
    connection = http.client.HTTPConnection('127.0.0.1', proxy.server_port, timeout=3)
    try:
        yield connection, responses, requests
    finally:
        connection.close()
        for server, thread in zip(servers, threads):
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def test_share_compresses_browser_assets_and_json_without_changing_data(public_share):
    connection, responses, requests = public_share
    for path in ('/terminal/app.js', '/research-terminal/api/archive'):
        connection.request('GET', path, headers={'Accept-Encoding': 'gzip, deflate, br'})
        response = connection.getresponse()
        body = response.read()
        assert response.status == 200
        assert response.getheader('Content-Encoding') == 'gzip'
        assert response.getheader('Vary') == 'Accept-Encoding'
        assert response.getheader('Cache-Control') == 'no-store'
        assert int(response.getheader('Content-Length')) == len(body)
        assert gzip.decompress(body) == responses[path][0]
        assert len(body) < len(responses[path][0]) / 2
    # The next request must still read fresh upstream data, not a cached snapshot.
    responses['/research-terminal/api/archive'] = (
        json.dumps({'updated': True, 'rows': ['new data'] * 200}).encode(), 'application/json')
    connection.request('GET', '/research-terminal/api/archive', headers={'Accept-Encoding': 'gzip'})
    response = connection.getresponse()
    assert gzip.decompress(response.read()) == responses['/research-terminal/api/archive'][0]
    assert requests.count('/research-terminal/api/archive') == 2


@pytest.mark.parametrize('encoding,compressed', [
    ('gzip', True), ('br, gzip;q=0.5', True), ('*;q=0.5', True),
    ('', False), ('identity', False), ('br', False), ('gzip;q=0', False),
    ('gzip;q=0, *;q=1', False), ('gzip;q=invalid', False), ('gzip;q=2', False),
])
def test_share_respects_browser_encoding_preferences(public_share, encoding, compressed):
    connection, responses, _ = public_share
    path = '/terminal/app.js'
    connection.request('GET', path, headers={'Accept-Encoding': encoding})
    response = connection.getresponse()
    body = response.read()
    assert response.getheader('Content-Encoding') == ('gzip' if compressed else None)
    assert (gzip.decompress(body) if compressed else body) == responses[path][0]


def test_share_compression_preserves_html_notice_head_and_small_or_binary_responses(public_share):
    connection, responses, _ = public_share
    connection.request('GET', '/terminal/', headers={'Accept-Encoding': 'gzip'})
    response = connection.getresponse()
    body = response.read()
    assert '唯讀分享'.encode() in gzip.decompress(body)
    assert b'#account-run,#backtest-submit{display:none!important}' in gzip.decompress(body)
    connection.request('HEAD', '/terminal/', headers={'Accept-Encoding': 'gzip'})
    response = connection.getresponse()
    assert response.getheader('Content-Encoding') == 'gzip'
    assert int(response.getheader('Content-Length')) == len(body)
    assert response.read() == b''
    for path in ('/research-terminal/api/overview', '/terminal/app.css'):
        connection.request('GET', path, headers={'Accept-Encoding': 'gzip'})
        response = connection.getresponse()
        assert response.getheader('Content-Encoding') is None
        assert response.read() == responses[path][0]
