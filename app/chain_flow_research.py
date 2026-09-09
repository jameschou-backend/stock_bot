"""Bounded Sponsor snapshots for current industry participation research."""
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
import pandas as pd
from sqlalchemy import select
from app.db import get_session
from app.finmind import fetch_dataset
from app.models import RawPrice, Stock
from app.news_research import atomic_json, overview as news_overview
from skills.chain_flow import summarize

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT/'.cache/chain-flow-research'


def read_or_fetch(name, dataset, day, config, stats, *, fetch=False):
    path, meta = CACHE/(name+'.parquet'), CACHE/(name+'.meta.json')
    if path.exists() and meta.exists():
        value = json.loads(meta.read_text())
        if value.get('sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('族群快取內容或版本不符，需明確修復：'+name)
        # Past daily snapshots are preserved; memberships are refreshed weekly on explicit fetch.
        stale = name=='members' and time.time()-value['retrieved_at'] > 7*86400
        if not stale or not fetch:
            stats['file_cache_hits'] += 1
            stats['inputs'][name] = value
            return pd.read_parquet(path)
    if not fetch: raise ValueError('缺少族群研究快取，請先執行「更新族群資金」：'+name)
    frame = fetch_dataset(dataset,day,token=config.finmind_token,
                          requests_per_hour=config.finmind_requests_per_hour,max_retries=0,timeout=30)
    if frame.empty: raise ValueError(f'{dataset} {day} 尚無資料；不跳過該交易日')
    if name!='members' and set(pd.to_datetime(frame['date']).dt.date) != {day}:
        raise ValueError('供應商回傳日期與請求不符')
    if name=='members' and frame.stock_id.nunique()<1000:
        raise ValueError('產業鏈成分疑似只取得近期異動；需要完整名單')
    tmp = path.with_suffix('.tmp');frame.to_parquet(tmp,index=False);tmp.replace(path)
    value = dict(frame.attrs)
    value.update(dataset=dataset,requested_date=str(day),rows=len(frame),
                 sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    atomic_json(meta,value)
    stats['gateway_cache_hits' if frame.attrs.get('cache_hit') else 'network_requests'] += 1
    stats['inputs'][name] = value
    print(f'{name}: {len(frame)} rows',flush=True)
    return frame


def run(*, fetch=False, config=None):
    started=time.perf_counter();CACHE.mkdir(parents=True,exist_ok=True)
    if fetch and config is None: raise ValueError('更新需要 FinMind 設定')
    with get_session() as session:
        days=session.execute(select(RawPrice.trading_date).distinct().order_by(RawPrice.trading_date.desc()).limit(25)).scalars().all()[::-1]
        companies=session.execute(select(Stock.stock_id,Stock.name).where(
            Stock.security_type=='stock',Stock.market.in_(['TWSE','TPEX']),Stock.is_listed==True)).all()
        if len(days)!=25: raise ValueError('本機不足 25 個交易日')
        prices=pd.read_sql(select(RawPrice.stock_id,RawPrice.trading_date.label('date'),RawPrice.close).where(
            RawPrice.trading_date.between(days[-5],days[-1])),session.get_bind())
    names={s:n for s,n in companies if s.isascii() and s.isdigit() and len(s)==4 and n and '-DR' not in n}
    stats={'network_requests':0,'file_cache_hits':0,'gateway_cache_hits':0,'inputs':{}}
    def get(name,dataset,day): return read_or_fetch(name,dataset,day,config,stats,fetch=fetch)
    members=get('members','TaiwanStockIndustryChain',date(1900,1,1))
    flow=pd.concat([get('flow-'+str(d),'TaiwanStockIndustryChainMoneyFlow',d) for d in days],ignore_index=True)
    quotes=get('prices-'+str(days[-1]),'TaiwanStockPrice',days[-1])
    institution=pd.concat([get('institution-'+str(d),'TaiwanStockInstitutionalInvestorsBuySell',d) for d in days[-5:]],ignore_index=True)
    news=news_overview()
    report=summarize(flow,members,quotes,prices,institution,names,news if news['available'] else None)
    report.update(schema=1,experiment='chain_flow_v1',research_only=True,live_qualified=False,
                  analyzed_at=datetime.now(timezone.utc).isoformat(),collection=stats,
                  code_sha256=code_sha(),members_update_min=str(members.date.min()),members_update_max=str(members.date.max()),
                  input_sha256={key:hashlib.sha256(frame.to_json(orient='records',date_format='iso').encode()).hexdigest()
                                for key,frame in [('flow',flow),('members',members),('quotes',quotes),('raw_closes',prices),('institution',institution)]},
                  limitations=['成交金額代表交易活躍度，不等於市場淨流入；不同產業鏈成分重疊，占比不可相加。',
                    '法人金額為淨股數乘收盤價近似，只含連續五日完整成分；外資含外資自營商，投信另列。',
                    '收紅為收盤高於開盤，不是較前收盤上漲，也不是含息報酬。',
                    '產業鏈使用目前分類回溯，沒有歷史成分版本，不能直接作無偏差回測。',
                    '歷史日快照保留原取得版本；分類每週更新可能造成成分與總額差異，須看對帳覆蓋。',
                    '新聞只連結日期不晚於資金表且標題有點名的文章，未確認盤中時序或直接受惠。'])
    report['elapsed_seconds']=round(time.perf_counter()-started,3)
    report['summary']={'groups':report['available_groups'],'elapsed_seconds':report['elapsed_seconds']}
    atomic_json(CACHE/'report.json',report)
    return report


def code_sha():
    return hashlib.sha256((ROOT/'skills/chain_flow.py').read_bytes()+(ROOT/'app/chain_flow_research.py').read_bytes()).hexdigest()


def overview():
    path=CACHE/'report.json'
    if not path.exists(): return {'available':False,'note':'尚無族群資金研究，請先更新'}
    try:
        d=json.loads(path.read_text())
        if d['code_sha256']!=code_sha() or d['live_qualified'] is not False: raise ValueError('stale')
        return {**d,'available':True}
    except (KeyError,ValueError,TypeError): return {'available':False,'note':'研究程式已變更或結果不完整，請重算'}
