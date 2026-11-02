"""Explicit synthetic transport for offline integration acceptance, never a fallback."""
from datetime import datetime
from urllib.parse import urlsplit

from .query import NY


class FixtureClient:
    def __init__(self):
        self.requests = 0

    def page(self, url):
        self.requests += 1
        path = urlsplit(url).path
        stamp = int(datetime(2026,10,29,10,tzinfo=NY).timestamp()) * 10**9
        if path.startswith("/v2/aggs/"):
            result = [dict(t=int(datetime(2026,10,29,tzinfo=NY).timestamp())*1000,o=100,h=103,l=99,c=102,v=20000)]
        elif path.startswith("/v3/trades/"):
            result = [dict(sip_timestamp=stamp,price=100,size=10,id="trade-1",exchange=12,conditions=[0],sequence_number=1),
                      dict(sip_timestamp=stamp,price=100,decimal_size="10.25",id="trade-2",exchange=12,conditions=[0],sequence_number=2)]
        elif path.startswith("/v3/reference/tickers"):
            result = [dict(ticker="AAPL",locale="us",market="stocks",active=True,share_class_figi="FIXTURE_ONLY")]
        elif path.endswith("/splits"):
            result = [dict(ticker="AAPL",execution_date="2026-10-29",split_from=1,split_to=2)]
        elif path.endswith("/dividends"):
            result = [dict(ticker="AAPL",ex_dividend_date="2026-10-29",cash_amount="0.25",currency="USD")]
        else:
            raise ValueError("Unexpected synthetic endpoint")
        return dict(status="OK",results=result,adjusted=False,synthetic_fixture=True)
