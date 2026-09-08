"""Shared read models for the workbench UI, API and MCP."""
from __future__ import annotations
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
import json

from sqlalchemy import func, select, text
from app.db import get_session
from app.models import Job, Pick, RawPrice, Stock
from app.market_calendar import get_latest_trading_day
from app.rate_limiter import get_rate_limiter
from app.workbench_models import WorkbenchAccount
from app.workbench_ledger import fills_for, plans_for, ledger_state, reservation
from skills.feature_store import FeatureStore
from skills.price_coverage import recent_market_counts, incomplete_dates

ROOT = Path(__file__).resolve().parents[1]


def latest_quotes(session, stock_ids):
    if not stock_ids:
        return {}
    dates=select(RawPrice.stock_id,func.max(RawPrice.trading_date).label('d')).where(
        RawPrice.stock_id.in_(stock_ids)).group_by(RawPrice.stock_id).subquery()
    rows=session.execute(select(RawPrice).join(dates,
        (RawPrice.stock_id==dates.c.stock_id)&(RawPrice.trading_date==dates.c.d))).scalars()
    return {row.stock_id:{'close':float(row.close),'date':row.trading_date}
            for row in rows if row.close is not None and row.close>0}


def data_status():
    with get_session() as s:
        latest = s.query(func.max(RawPrice.trading_date)).scalar()
        expected = get_latest_trading_day(s)
        counts = recent_market_counts(s, latest) if latest else None
        gaps = incomplete_dates(counts) if counts is not None else []
        markets = []
        if counts is not None and not counts.empty:
            for market, frame in counts.groupby('market'):
                row = frame.sort_values('trading_date').iloc[-1]
                markets.append({'market': market, 'date': str(row.trading_date), 'stocks': int(row.rows_count)})
        jobs = [{'name': j.job_name, 'status': j.status, 'started_at': str(j.started_at),
                 'ended_at': str(j.ended_at) if j.ended_at else None}
                for j in s.query(Job).order_by(Job.started_at.desc()).limit(12)]
        fs = FeatureStore()
        feature_date = fs.get_max_date()
        feature_count = len(fs.read(latest, latest)) if latest else 0
        latest_count = (int(counts[counts.trading_date == latest].rows_count.sum())
                        if counts is not None and not counts.empty else 0)
    problems = []
    if not latest or (expected and latest < expected):
        problems.append('股價尚未更新至最近已收盤交易日')
    recent_gaps = [str(d) for d in gaps if latest and d >= latest - timedelta(days=7)]
    if recent_gaps:
        problems.append('上市／上櫃覆蓋不足：' + '、'.join(recent_gaps))
    if not feature_date or (latest and feature_date < latest) or feature_count < latest_count * .9:
        problems.append('特徵尚未覆蓋最新股價，請完成資料更新')
    from app.config import load_config
    quota = asdict(get_rate_limiter(load_config().finmind_requests_per_hour).get_stats())
    return {'price_date': str(latest) if latest else None, 'expected_date': str(expected) if expected else None,
            'feature_date': str(feature_date) if feature_date else None, 'feature_stocks': feature_count,
            'markets': markets, 'data_ready': not problems, 'problems': problems,
            'strategy_ready': False,
            'strategy_note': '目前策略尚未完成新的成本後驗證；候選名單與計畫供研究／紙上追蹤。',
            'adjustment_note': '還原價來源與歷史修正尚需完整對帳，不能以因子表最新日期推定正確。',
            'quota': quota, 'jobs': jobs, 'observed_at': datetime.utcnow().isoformat() + 'Z'}


def candidates(limit=20):
    with get_session() as s:
        latest = s.query(func.max(Pick.pick_date)).scalar()
        if latest is None:
            return []
        rows = s.execute(select(Pick, Stock.name, Stock.market).join(Stock, Stock.stock_id == Pick.stock_id)
            .where(Pick.pick_date == latest, Stock.security_type == 'stock')
            .order_by(Pick.score.desc(), Pick.stock_id).limit(min(max(limit, 1), 50))).all()
        quotes=latest_quotes(s,[row[0].stock_id for row in rows])
        out=[]
        for rank, (p, name, market) in enumerate(rows, 1):
            quote = quotes.get(p.stock_id)
            out.append({'stock_id':p.stock_id,'name':name,'market':market,'rank':rank,
                        'score':float(p.score) if p.score is not None else None,
                        'signal_date':str(p.pick_date),'price':quote['close'] if quote else None,
                        'price_date':str(quote['date']) if quote else None,
                        'purpose':'research_only'})
        return out


def portfolio(account_id='paper'):
    if account_id not in ('paper','real'):
        raise ValueError('未知帳本')
    with get_session() as s:
        account=s.get(WorkbenchAccount,account_id)
        if account is None:
            return {'initialized':False,'account_id':account_id}
        fills=fills_for(s,account_id)
        ids=sorted({f.stock_id for f in fills})
        quotes=latest_quotes(s,ids)
        result=ledger_state(account.initial_cash,fills,quotes)
        plans=plans_for(s,account_id)
        reserved=float(sum(reservation(p) for p in plans))
        result.update(initialized=True,account_id=account_id,initial_cash=float(account.initial_cash),
                      reserved_cash=reserved,available_cash=max(0,result['cash']-reserved),
                      funding_shortfall=max(0,reserved-result['cash']))
        result['plans']=[{'plan_id':p.plan_id,'stock_id':p.stock_id,'entry_price':float(p.entry_price),
                          'stop_price':float(p.stop_price),'qty':p.qty,'filled_qty':p.filled_qty,
                          'status':p.status,'reason':p.reason} for p in plans]
        result['fills']=[{'fill_id':f.fill_id,'stock_id':f.stock_id,'side':f.side,'qty':f.qty,
                          'price':float(f.price),'fee':float(f.fee),'tax':float(f.tax),
                          'executed_at':f.executed_at.isoformat()+'Z'} for f in fills]
        return result


def strategy_evidence():
    docs=[]
    for path in sorted((ROOT/'docs').glob('prereg*.md')):
        docs.append({'name':path.name,'path':str(path),'status':'historical_research'})
    return {'live_qualified':False,'note':'目前沒有通過新驗證的實盤策略。歷史回測不等於實際獲利。',
            'documents':docs,'validation_requirements':['扣除稅費與滑價','訊號延遲至少一交易日',
                '歷史資料可用時間與還原價對帳','與同期間基準比較','未用來調參的測試期與向前紙上追蹤']}
