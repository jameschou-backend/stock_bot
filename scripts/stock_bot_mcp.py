"""Read-only stdio MCP for the local workbench. FinMind requests go through its quota gateway."""
from datetime import date
import json
from typing import Literal
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError,URLError

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

mcp=FastMCP('stock-bot',instructions='台股研究工具。候選名單不是上漲機率；所有策略先看資料與驗證狀態。工具不會下單或修改成交帳本。')
READ=ToolAnnotations(readOnlyHint=True,destructiveHint=False,openWorldHint=False)


def get(path,params=None):
    url='http://127.0.0.1:8000/workbench/'+path
    if params: url+='?'+urlencode(params)
    try:
        with urlopen(url,timeout=65) as response:
            return json.load(response)
    except HTTPError as exc:
        try: detail=json.loads(exc.read()).get('detail','API request failed')
        except ValueError: detail='API request failed'
        raise ValueError(f'HTTP {exc.code}: {detail}') from None
    except URLError:
        raise ValueError('Stock Bot API 尚未啟動。請在專案執行 make api。') from None


@mcp.tool(annotations=READ)
def get_data_status() -> dict:
    """Read per-market freshness, feature coverage and shared FinMind hourly usage."""
    return get('status')


@mcp.tool(annotations=READ)
def get_candidates(limit:int=20) -> list:
    """Read research candidates, signal dates and price dates; scores are not probabilities."""
    if not 1<=limit<=50: raise ValueError('limit must be 1..50')
    return get('candidates',{'limit':limit})


@mcp.tool(annotations=READ)
def get_portfolio(account_id:Literal['paper','real']='paper') -> dict:
    """Read separate paper/actual fill ledgers, costs and P&L. Missing prices remain unknown."""
    return get('portfolio',{'account_id':account_id})


@mcp.tool(annotations=READ)
def get_trade_plans(account_id:Literal['paper','real']='paper') -> list:
    """Read plans and reserved cash; no broker orders are placed."""
    return get('portfolio',{'account_id':account_id}).get('plans',[])


@mcp.tool(annotations=READ)
def get_strategy_evidence() -> dict:
    """Read qualification status and research artifacts before interpreting any return figure."""
    return {'evidence':get('evidence'),'recent_tasks':get('tasks')}


@mcp.tool(annotations=READ)
def get_news_research(mode:Literal['scan','review']='scan',limit:int=50) -> dict:
    """Read analyzed headline themes or historical review. No fetch, LLM, orders, or implied verified beneficiaries."""
    if not 1<=limit<=200: raise ValueError('limit must be 1..200')
    return get('news',{'mode':mode,'limit':limit})


@mcp.tool(annotations=READ)
def get_chain_flow_research(limit:int=60) -> dict:
    """Read completed sector turnover, breadth and estimated investor activity. No fetch or trades."""
    if not 1<=limit<=80: raise ValueError('limit must be 1..80')
    return get('chain-flow',{'limit':limit})


@mcp.tool(annotations=READ)
def get_trade_review(account_id:Literal['paper','real']='paper') -> dict:
    """Read user-reported executions for fee, sizing and exit reviews."""
    book=get('portfolio',{'account_id':account_id})
    return {key:book.get(key) for key in ('initialized','account_id','fills','realized_pnl','fees_and_tax','missing_quotes')}


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True,destructiveHint=False,openWorldHint=True))
def query_finmind(dataset:Literal['TaiwanStockPrice','TaiwanStockPriceAdj','TaiwanStockPER','TaiwanStockMonthRevenue','TaiwanStockKBar'],
                  stock_id:str,start_date:date,end_date:date) -> dict:
    """Query one stock through shared quota/cache. Max 31 days; KBar one day. Charges API usage on cache miss."""
    return get('finmind',dict(dataset=dataset,stock_id=stock_id,start_date=start_date,end_date=end_date))


if __name__=='__main__':
    mcp.run(transport='stdio')
