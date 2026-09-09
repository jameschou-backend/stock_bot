"""Local workbench endpoints. Fill writes only record user-reported executions."""
from datetime import date, datetime
from typing import Literal
from fastapi import APIRouter, HTTPException, Query, Request, Depends
from ipaddress import ip_address
import json
from pydantic import BaseModel, Field
from app.db import get_session
from app import workbench_service as service, workbench_ledger as ledger, workbench_jobs as jobs

def require_local(request: Request):
    try:
        local=request.client is not None and ip_address(request.client.host).is_loopback
    except ValueError:
        local=False
    if not local:
        raise HTTPException(403,'投資帳本與研究操作限本機使用')


router=APIRouter(prefix='/workbench',tags=['workbench'],dependencies=[Depends(require_local)])


def checked(fn,*args,**kwargs):
    try:
        return fn(*args,**kwargs)
    except (ValueError,TimeoutError) as exc:
        raise HTTPException(status_code=400,detail=str(exc)) from None


@router.get('/status')
def status(): return service.data_status()


@router.get('/candidates')
def candidates(limit:int=Query(20,ge=1,le=50)): return service.candidates(limit)


@router.get('/evidence')
def evidence(): return service.strategy_evidence()


@router.get('/news')
def news_research(mode:Literal['scan','review']='scan',limit:int=Query(50,ge=1,le=200)):
    from app.news_research import overview
    report=overview(mode)
    if report['available']:
        stories=report['stories']
        report={**report,'stories':stories[-limit:],'stories_truncated':len(stories)>limit,
                'story_count':len(stories)}
    return report


@router.get('/portfolio')
def portfolio(account_id:Literal['paper','real']='paper'): return service.portfolio(account_id)


@router.get('/chain-flow')
def chain_flow(limit:int=Query(60,ge=1,le=80)):
    from app.chain_flow_research import overview
    report=overview()
    if report['available']:
        report={**report,'groups':[{k:v for k,v in g.items() if k!='share_history'} for g in report['groups'][:limit]],
                'groups_truncated':len(report['groups'])>limit}
    return report


class AccountIn(BaseModel):
    account_id:Literal['paper','real']='paper'
    initial_cash:float=Field(gt=0,le=1e10,allow_inf_nan=False)


@router.post('/accounts')
def account(payload:AccountIn):
    with get_session() as s:
        checked(ledger.initialize_account,s,**payload.model_dump())
    return {'created':True}


class PlanIn(BaseModel):
    account_id:Literal['paper','real']='paper'
    stock_id:str=Field(pattern=r'^\d{4}$')
    entry_price:float=Field(gt=0,allow_inf_nan=False)
    stop_price:float=Field(gt=0,allow_inf_nan=False)
    qty:int=Field(gt=0,le=10000000)
    reason:str=Field(default='',max_length=500)


@router.post('/plans')
def plan(payload:PlanIn):
    with get_session() as s:
        plan_id=checked(ledger.create_plan,s,**payload.model_dump())
    return {'plan_id':plan_id}


@router.post('/plans/{plan_id}/cancel')
def cancel(plan_id:str,account_id:Literal['paper','real']='paper'):
    with get_session() as s:
        checked(ledger.cancel_plan,s,account_id,plan_id)
    return {'cancelled':True}


class FillIn(BaseModel):
    account_id:Literal['paper','real']='paper'
    fill_id:str=Field(pattern=r'^[0-9a-f]{32}$')
    stock_id:str=Field(pattern=r'^\d{4}$')
    side:Literal['buy','sell']
    qty:int=Field(gt=0,le=10000000)
    price:float=Field(gt=0,allow_inf_nan=False)
    fee:float=Field(ge=0,allow_inf_nan=False)
    tax:float=Field(ge=0,allow_inf_nan=False)
    executed_at:datetime
    plan_id:str|None=None


@router.post('/fills')
def fill(payload:FillIn):
    with get_session() as s:
        fill_id=checked(ledger.record_fill,s,**payload.model_dump())
    return {'fill_id':fill_id,'broker_order_submitted':False}


@router.get('/tasks')
def tasks(): return jobs.recent_jobs()


@router.post('/tasks')
def task(payload:jobs.WorkRequest): return checked(jobs.submit,payload)


@router.get('/tasks/{job_id}')
def task_result(job_id:str): return checked(jobs.read_job,job_id)


@router.get('/finmind')
def finmind_data(dataset:Literal['TaiwanStockPrice','TaiwanStockPriceAdj','TaiwanStockPER','TaiwanStockMonthRevenue','TaiwanStockKBar'],
                 stock_id:str=Query(pattern=r'^\d{4}$'),start_date:date=Query(),end_date:date=Query()):
    from app.config import load_config
    from app.finmind import fetch_dataset,FinMindQuotaError,FinMindError
    if end_date<start_date or (end_date-start_date).days>31 or (dataset=='TaiwanStockKBar' and end_date!=start_date):
        raise HTTPException(400,'最多查詢 31 天；分鐘 K 每次限一天')
    try:
        config=load_config()
        frame=fetch_dataset(dataset,start_date,end_date,token=config.finmind_token,
                            data_id=stock_id,requests_per_hour=config.finmind_requests_per_hour)
        return {'rows':json.loads(frame.head(1000).to_json(orient='records',date_format='iso')),'total_rows':len(frame),
                'truncated':len(frame)>1000,**frame.attrs}
    except FinMindQuotaError as exc:
        raise HTTPException(429,str(exc),headers={'Retry-After':str(int(exc.retry_after_seconds)+1)}) from None
    except FinMindError as exc:
        raise HTTPException(502,str(exc)) from None
