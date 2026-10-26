#!/usr/bin/env python3
"""Loopback-only read-only research proxy for the existing public tunnel."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import argparse
import gzip
import re

UPSTREAM = 'http://127.0.0.1:8000'
EXACT = {'/terminal','/terminal/','/terminal/index.html','/terminal/app.css','/terminal/app.js',
    '/terminal/poc-paper-preparation.js', '/terminal/poc-market-sizing.js',
    '/terminal/poc-drawdown.js', '/terminal/poc-weakness.js', '/terminal/poc-cap40.js',
    '/terminal/poc-trailing.js', '/terminal/poc-split-accounts.js', '/terminal/poc-risk-controls.js',
    '/terminal/poc-followup.js', '/terminal/poc-optimization.js',
    '/terminal/comparison-overview.js', '/terminal/account-comparison.js', '/terminal/top3-validation.js', '/terminal/legacy-comparison.js', '/research-terminal/api/overview',
    '/research-terminal/api/strategies','/research-terminal/api/signals',
    '/research-terminal/api/studies','/research-terminal/api/research',
    '/research-terminal/api/research/episodes','/research-terminal/api/research/examples',
    '/research-terminal/api/archive'}


def allowed(path):
    return path in EXACT or bool(re.fullmatch(
        r'/research-terminal/api/(?:stocks/[0-9]{4}|archive/[a-z0-9_]{1,80}(?:/cases/[a-z0-9_]{1,80})?)',path))


def accepts_gzip(value):
    qualities = {}
    for entry in value.lower().split(','):
        coding, *parameters = (part.strip() for part in entry.split(';'))
        if coding not in {'gzip', '*'}:
            continue
        quality = 1.0
        for parameter in parameters:
            name, _, raw = parameter.partition('=')
            if name.strip() == 'q':
                try:
                    quality = float(raw)
                except ValueError:
                    quality = 0.0
        qualities[coding] = quality if 0 <= quality <= 1 else 0.0
    # An explicit gzip;q=0 takes precedence over the wildcard.
    return qualities.get('gzip', qualities.get('*', 0.0)) > 0


class Share(BaseHTTPRequestHandler):
    def log_message(self,*args):
        pass

    def reply(self,status,body=b'',content_type='text/plain; charset=utf-8'):
        encoding = None
        media_type = content_type.split(';', 1)[0].strip().lower()
        compressible = media_type.startswith('text/') or media_type in {
            'application/json', 'application/javascript'}
        if status == 200 and len(body) >= 1024 and compressible:
            accepted = ','.join(self.headers.get_all('Accept-Encoding', []))
            if accepts_gzip(accepted):
                packed = gzip.compress(body, compresslevel=5, mtime=0)
                if len(packed) < len(body):
                    body, encoding = packed, 'gzip'
        self.send_response(status)
        if compressible:
            self.send_header('Vary', 'Accept-Encoding')
        if encoding:
            self.send_header('Content-Encoding', encoding)
        for key,value in {'Content-Type':content_type,'Content-Length':str(len(body)),
                'Cache-Control':'no-store','X-Content-Type-Options':'nosniff',
                'Referrer-Policy':'no-referrer', 'Content-Security-Policy':
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; "
                "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"}.items():
            self.send_header(key,value)
        self.end_headers()
        if self.command!='HEAD':
            self.wfile.write(body)

    def do_GET(self):
        parsed=urlsplit(self.path)
        if parsed.scheme or parsed.netloc or len(self.path)>4096:
            return self.reply(400,b'Invalid target')
        if parsed.path=='/':
            self.send_response(302);self.send_header('Location','/terminal/#backtest')
            self.send_header('Content-Length','0');self.end_headers();return
        if not allowed(parsed.path):
            return self.reply(403,b'Read-only research routes only.')
        if self.headers.get('Transfer-Encoding') or self.headers.get('Content-Length','0')!='0':
            return self.reply(400,b'Request body not allowed')
        target=UPSTREAM+parsed.path+('?' + parsed.query if parsed.query else '')
        try:
            with urlopen(Request(target,headers={'User-Agent':'StockTerminalReadOnlyShare/2.0',
                    'Accept-Encoding':'identity'}),timeout=60) as response:
                body=response.read();content_type=response.headers.get('Content-Type','application/octet-stream');status=response.status
            if content_type.startswith('text/html'):
                notice='<div style="padding:10px 20px;background:#e4f4ef;color:#154d45;font:14px sans-serif;text-align:center">唯讀分享：可查看訊號、研究比較與成交明細；執行回測請使用本機工作台。</div><style>#account-run,#backtest-submit{display:none!important}</style>'
                body=body.replace(b'<body>',('<body>'+notice).encode(),1)
            self.reply(status,body,content_type)
        except HTTPError as exc:
            self.reply(exc.code,exc.read(),exc.headers.get('Content-Type','application/json'))
        except (URLError,TimeoutError):
            self.reply(502,b'Research terminal is unavailable')

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        self.reply(403,b'Read-only share: writes and job execution are disabled.')

    do_PUT=do_POST;do_PATCH=do_POST;do_DELETE=do_POST;do_OPTIONS=do_POST;do_TRACE=do_POST;do_CONNECT=do_POST


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=18125)
    args=parser.parse_args()
    ThreadingHTTPServer(('127.0.0.1',args.port),Share).serve_forever()
