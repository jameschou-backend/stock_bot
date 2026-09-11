"""Explicit halt evidence and labelled estimates, outside the frozen trading engine."""
from datetime import datetime
from decimal import Decimal as D
import re
from app import forward_journal as j, forward_portfolio as p
from app.forward_corporate_resolution import validate_evidence

VERSION='halt-valuation-v1'


def notices(rows):
    return {r['body']['stock_id']:r for r in rows if r['kind']=='halt_notice'}


def save_notice(path,terms,evidence,clock=j.now):
    if set(terms)!={'stock_id','halt_start','resume_date','withdrawn'} or not re.fullmatch(r'\d{4}',terms['stock_id']):
        raise ValueError('停牌紀錄需四碼代號、停牌起日、恢復日及撤回狀態')
    start=p.iso(terms['halt_start']);resume=p.iso(terms['resume_date']) if terms['resume_date'] else None
    if (resume and resume<=start) or type(terms['withdrawn']) is not bool:raise ValueError('恢復日須晚於停牌起日；僅支援完整交易日停牌')
    proof=validate_evidence(evidence,clock)
    if str(j.timestamp(evidence['published_at']).astimezone(p.TZ).date())>str(clock().astimezone(p.TZ).date()):raise ValueError('不能保存未來公告')
    with j.connection(path) as con:
        p.initialize(con,clock);rows=j.read_events(con)
        # A revision appends a new notice; never overwrites the source or prior NAV.
        body=dict(version=VERSION,**terms,evidence=proof)
        old=notices(rows).get(terms['stock_id'])
        if old and all(old['body'].get(k)==v for k,v in body.items()):return old
        body['supersedes']=old['hash'] if old else None
        return j.append(con,'halt_notice:'+j.digest(body),'halt_notice',body,clock)


def active(rows,day):
    return {sid:r for sid,r in notices(rows).items() if not r['body']['withdrawn']
            and r['body']['halt_start']<=day and (not r['body']['resume_date'] or day<r['body']['resume_date'])}


def estimated(rows):
    mark=p.state(rows)['mark']
    return mark['body'].get('estimated_prices',{}) if mark else {}


def valuation_prices(rows,day,observed):
    """Only a dated halt permits carry-forward. Unknown ordinary prices still fail.

    Rebase the last actually observed quote for confirmed cash rights and delivered
    splits since that observation. Never recursively adjust yesterday's estimate.
    """
    prices=dict(observed);details={};s=p.state(rows);halted=active(rows,day)
    for sid in s['holdings']:
        if sid in prices:
            if sid in halted:raise ValueError(sid+' 停牌公告與當日成交價格衝突，須先核對恢復交易日期')
            continue
        notice=halted.get(sid)
        if not notice:raise ValueError(sid+' 缺少當日價格且無有效完整日停牌證據，不可沿用舊價')
        prior=next((r for r in reversed(rows) if r['kind']=='close' and r['body']['date']<day
                    and sid in r['body']['prices'] and sid not in r['body'].get('estimated_prices',{})),None)
        if not prior:raise ValueError(sid+' 缺少停牌前已觀察價格')
        quote_day=prior['body']['date'];price=p.num(prior['body']['prices'][sid],True)
        # An unrelated data gap before the halt cannot be hidden by a later notice.
        if any(quote_day<d<notice['body']['halt_start'] for d in s['calendar']):
            raise ValueError(sid+' 停牌前仍有行情缺口，不能跨越缺漏估價')
        adjustments=[]
        for right in sorted(s['rights'].values(),key=lambda x:x['ex_date']):
            if right['stock_id']!=sid or not quote_day<right['ex_date']<=day:continue
            if right['action_type']=='cash':price-=D(right['cash_per_share'])
            elif right['action_type']=='split':
                if not right['paid']:raise ValueError(sid+' 分割股尚未交付，估值仍未知')
                price/=D(right['ratio'])
            adjustments.append(right['action_id'])
        if price<=0:raise ValueError('公司行動調整後估價非正值，需另行對帳')
        prices[sid]=str(price)
        details[sid]=dict(price=str(price),price_date=quote_day,valuation_date=day,
            source_close_hash=prior['hash'],halt_notice_hash=notice['hash'],adjustments=adjustments,
            method='last_observed_close_adjusted_for_recorded_rights',tradable=False)
    return prices,details


def require_observed(path):
    with j.connection(path) as con:
        p.initialize(con);rows=j.read_events(con)
    today=str(j.now().astimezone(p.TZ).date())
    if estimated(rows) or active(rows,today):raise ValueError('持股含停牌估值，暫停新買進及資金再配置；恢復交易並取得完整行情後再建立計畫')
    return rows


def save_plans(path,signal_path=j.DEFAULT_PATH,benchmark=False,clock=j.now):
    from app import forward_comparison as c
    with j.connection(path) as con:
        p.initialize(con,clock);rows=j.read_events(con)
    s=p.state(rows)
    next_day=p.next_session(s,s['mark']['body']['date']) if s['mark'] else str(clock().astimezone(p.TZ).date())
    if not estimated(rows) and not active(rows,next_day):
        return c.reinvest_dividends(path,clock) if benchmark else c.save_strategy_plans(path,signal_path,clock)
    if benchmark:raise ValueError('0050含停牌估值，股息再投入暫停')
    # Preserve exits for independently observed stocks; an estimated price never triggers one.
    saved=[]
    with j.connection(path) as con:
        c.protocol(con,'strategy',clock);rows=j.read_events(con);s=p.state(rows);mark=s['mark']
        day=str(clock().astimezone(p.TZ).date())
        if not mark or mark['body']['date']!=day or p.valuation(s) is None:raise ValueError('出場需今日完整核對估值')
        session=p.next_session(s,day);_,reserved=p.reserves(s,day)
        blocked=set(estimated(rows))|set(active(rows,session))
        for sid,pos in s['holdings'].items():
            if sid=='0050' or sid in blocked:continue
            n=pos['qty']-reserved.get(sid,0)
            if n<=0:continue
            price=D(mark['body']['prices'][sid]);held=sum(pos['entry_date']<=d<=day for d in s['calendar'])
            reason='stop12_close' if price<=pos['stop_basis']*D('.88') else 'holding63' if held>=63 else None
            decision=s['decisions'].get(sid)
            if not reason and not decision:continue
            if not decision:decision=p.submit(con,dict(kind='decision',id=day+':'+sid,stock_id=sid,date=day,reason=reason),clock)
            for channel,size in [('board',n//1000*1000),('odd',n%1000)]:
                if size:
                    key=f'exit:{session}:{sid}:{channel}'
                    saved.append(p.submit(con,dict(kind='order',id=key,order_id=key,stock_id=sid,side='sell',channel=channel,qty=size,limit_price=str(price),session=session,reason=decision['hash']),clock))
    return dict(exits=saved,new_plans=[],entry_block='持股含停牌估值，僅保留其他有當日行情個股的出場，新買進暫停')


def record(path,command,benchmark=False,clock=j.now):
    from app import forward_comparison as c
    if command['kind'] not in ('fill','cancel','entitlement','delivery'):raise ValueError('只接受成交、取消、權益及交付')
    with j.connection(path) as con:
        p.initialize(con,clock);rows=j.read_events(con)
        is_benchmark=benchmark or any(r['kind']=='comparison_anchor' for r in rows)
        if is_benchmark:
            if not any(r['kind']=='comparison_anchor' for r in rows):raise ValueError('基準尚未初始化')
            c.protocol(con,'benchmark',clock)
            if command['kind']=='entitlement' and command['stock_id']!='0050':raise ValueError('基準只能持有0050')
        if command['kind']=='fill':
            order=p.state(rows)['orders'].get(command['order_id'])
            day=str(j.timestamp(command['executed_at']).astimezone(p.TZ).date())
            if order and order['stock_id'] in active(rows,day):raise ValueError('成交日有有效停牌證據，須先核對公告或回報；不可直接成交')
        return p.submit(con,command,clock)


def compare(strategy,benchmark):
    from app.forward_comparison import comparison
    special=[]
    for path in (strategy,benchmark):
        if not path.exists():continue
        with j.connection(path) as con:rows=j.read_events(con)
        if any(r['kind']=='restatement_lineage' for r in rows):special.append('含事後更正版本，僅供會計對帳，不列原始前向績效')
        if any(r['kind']=='close' and r['body'].get('estimated_prices') for r in rows):special.append('期間含停牌估值，不能視為全部每日價格均可成交的比較')
    if special:return dict(ready=False,reasons=sorted(set(special)),points=[],live_qualified=False)
    return comparison(strategy,benchmark)
