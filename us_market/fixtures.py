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


class AlpacaFixtureClient:
    def __init__(self):
        self.requests = 0

    def page(self, url):
        from .alpaca import _parts
        path, params = _parts(url)
        self.requests += 1
        if params["symbols"] != "AAPL" or not params["start"].startswith("2026-10-29") or "page_token" in params:
            raise ValueError("Unexpected Alpaca synthetic query")
        if path.endswith("/bars"):
            field = "bars"
            rows = [dict(t="2026-10-29T04:00:00Z",o=100,h=100,l=100,c=100,v="10.25")]
        else:
            field = "trades"
            rows = [dict(t="2026-10-29T14:00:00.123456789Z",p=100,s=10,i=1,x="V",c=["@"],z="C"),
                    dict(t="2026-10-29T14:00:00.123456789Z",p=100,s="0.25",i=2,x="D",c=["I"],z="C")]
        return {field:{"AAPL":rows},"next_page_token":None,"synthetic_fixture":True}
