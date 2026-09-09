"""Descriptive sector participation, never a buy score or total net money flow."""
from __future__ import annotations
import numpy as np
import pandas as pd

FOCUS = [
    ('半導體', '記憶體IC', '記憶體 IC', 'memory'),
    ('通信網路', '光通訊設備(如光纖電纜、光傳輸設備)', '光通訊設備', 'photonics'),
    ('電腦及週邊設備', '伺服器', '伺服器', 'ai_server'),
    ('電腦及週邊設備', '散熱片、風扇馬達、散熱模組', '散熱元件／模組', 'cooling'),
    ('通信網路', '無線通訊設備(如行動電話、衛星定位系統、衛星通訊設備、微波通訊設備、數位機上盒)',
     '無線通訊（含衛星，非純低軌）', 'leo'),
]


def institutional_net(frame):
    required = {'date', 'stock_id', 'name', 'buy', 'sell'}
    if not required.issubset(frame): raise ValueError('法人來源缺少必要欄位')
    frame = frame.copy()
    frame = frame[frame.stock_id.astype(str).str.fullmatch(r'[0-9]{4}')]
    if frame.duplicated(['date', 'stock_id', 'name']).any(): raise ValueError('法人分類重複，禁止重複加總')
    for c in ['buy', 'sell']:
        frame[c] = pd.to_numeric(frame[c], errors='coerce')
        frame.loc[frame[c] < 0, c] = np.nan
    frame['net'] = frame['buy']-frame['sell']
    values = frame.pivot(index=['date', 'stock_id'], columns='name', values='net')
    for c in ['Investment_Trust', 'Foreign_Investor', 'Foreign_Dealer_Self']:
        if c not in values: values[c] = np.nan
    return pd.DataFrame({'trust_net': values.Investment_Trust,
                         'foreign_net': values.Foreign_Investor+values.Foreign_Dealer_Self}).reset_index()


def summarize(flow, members, prices, raw_closes, institution, names, news=None):
    """25 supplied sessions; 5-day recent vs separate 20-day base. Missing stays unknown."""
    flow, prices, raw_closes, institution = flow.copy(), prices.copy(), raw_closes.copy(), institution.copy()
    for frame in [flow, prices, raw_closes, institution]:
        frame['date'] = pd.to_datetime(frame['date']).dt.strftime('%Y-%m-%d')
    days = sorted(flow.date.unique())
    if len(days) != 25: raise ValueError('必須正好有 25 個交易日的產業鏈資料')
    end = days[-1]
    if flow.duplicated(['date','industry','sub_industry']).any(): raise ValueError('產業鏈日期重複')
    if prices.date.nunique() != 1 or prices.date.iloc[0] != end: raise ValueError('當日行情與產業鏈日期不同')
    if prices.stock_id.duplicated().any() or raw_closes.duplicated(['date','stock_id']).any():
        raise ValueError('行情重複')
    if set(institution.date.unique()) != set(days[-5:]): raise ValueError('法人日期不完整')
    for c in ['trading_money','trading_money_pct','stock_count']:
        flow[c] = pd.to_numeric(flow[c], errors='raise')
        if flow[c].isna().any() or (flow[c] < 0).any(): raise ValueError('產業鏈數值缺漏或為負')
    prices = prices.set_index('stock_id')
    for c in ['Trading_money','open','close']:
        prices[c] = pd.to_numeric(prices[c], errors='coerce')
    close = raw_closes.pivot(index='date', columns='stock_id', values='close').reindex(days[-5:]).apply(pd.to_numeric)
    net = institutional_net(institution)
    amounts, complete, daily_amounts = {}, {}, {}
    for kind in ['trust','foreign']:
        shares = net.pivot(index='date', columns='stock_id', values=kind+'_net').reindex(index=close.index, columns=close.columns)
        cash = shares * close.where(close > 0)
        complete[kind] = cash.notna().all(axis=0)
        amounts[kind] = cash.sum(axis=0, min_count=5)
        daily_amounts[kind] = cash
    current = flow[flow.date.eq(end)]
    groups = [(i, '', i, None) for i in sorted(current[current.sub_industry.eq('')].industry.unique())] + FOCUS
    result = []
    for industry, sub, label, topic in groups:
        series = flow[flow.industry.eq(industry) & flow.sub_industry.eq(sub)].set_index('date').reindex(days)
        if series.trading_money_pct.isna().any():
            result.append({'name':label, 'industry':industry, 'sub_industry':sub, 'available':False,
                           'note':'25 日產業鏈資料不完整'})
            continue
        m = members[members.industry.eq(industry)]
        if sub: m = m[m.sub_industry.eq(sub)]
        # Current provider membership; never substitute headline-related stock tags.
        provider_ids = sorted(set(m.stock_id.astype(str)))
        ids = [s for s in provider_ids if s in names]
        quote = prices.reindex(ids)
        traded = quote[quote.Trading_money.gt(0) & quote['close'].gt(0) & quote.open.gt(0)]
        expected = int(series.iloc[-1].stock_count)
        reported_money = float(series.iloc[-1].trading_money)
        member_money = float(traded.Trading_money.sum())
        money_gap = abs(member_money/reported_money-1) if reported_money > 0 else None
        coverage = len(traded)/expected if expected else 0
        reconciled = .95 <= coverage <= 1.05 and money_gap is not None and money_gap <= .02
        share = series.trading_money_pct
        recent, baseline = float(share.iloc[-5:].mean()), float(share.iloc[:-5].mean())
        concentration = float(traded.Trading_money.max()/member_money) if member_money > 0 else None
        breadth = float(traded['close'].gt(traded.open).mean()) if len(traded) else None
        item = {'name':label, 'industry':industry, 'sub_industry':sub, 'topic':topic, 'available':True,
                'excluded_provider_ids':[s for s in provider_ids if s not in names],
                'missing_quote_ids':[s for s in ids if s not in traded.index],
                'as_of':end, 'members':len(ids), 'quoted_members':len(traded), 'provider_traded_members':expected,
                'member_coverage':coverage, 'money_reconciliation_gap':money_gap, 'components_reconciled':reconciled,
                'today_money':reported_money, 'today_share_pct':float(share.iloc[-1]),
                'share_5d_pct':recent, 'share_previous20_pct':baseline,
                'share_change_pp':recent-baseline, 'share_multiple':recent/baseline if baseline>0 else None,
                'intraday_up_fraction':breadth, 'top1_share':concentration,
                'observed_money_ex_top1':member_money-float(traded.Trading_money.max()) if len(traded) else None,
                'share_history':[{'date':d,'share_pct':float(v)} for d,v in share.items()]}
        cash_ok = True
        for kind in ['trust','foreign']:
            valid = [s for s in traded.index if s in complete[kind] and complete[kind][s]]
            ratio = len(valid)/len(traded) if len(traded) else 0
            ok = reconciled and ratio >= .9
            cash_ok = cash_ok and ok
            item[kind+'_coverage'] = ratio
            item[kind+'_observed_stocks'] = len(valid)
            # Amount is for complete observed members only, never a fabricated group total.
            item[kind+'_observed_net_est_5d'] = float(amounts[kind].reindex(valid).sum()) if valid else None
            item[kind+'_positive_days'] = int((daily_amounts[kind].reindex(columns=valid).sum(axis=1)>0).sum()) if valid else None
        item['flow_data_ready'] = cash_ok
        if not cash_ok: item['reading'] = '資料待核對'
        elif recent <= baseline: item['reading'] = '成交占比未增加'
        elif concentration is not None and concentration > .5: item['reading'] = '成交集中單一龍頭'
        elif breadth is not None and breadth <= .5: item['reading'] = '成交增加，當日收紅未過半'
        elif item['trust_observed_net_est_5d'] > 0 or item['foreign_observed_net_est_5d'] > 0:
            item['reading'] = '成交擴散，至少一類法人偏買'
        else: item['reading'] = '成交擴散，兩類法人未偏買'
        leaders = traded.sort_values('Trading_money',ascending=False).head(5)
        item['leaders'] = [{'stock_id':sid,'name':names[sid], 'money':float(p.Trading_money),
                            'intraday_return':float(p['close']/p.open-1),
                            'trust_net_est_5d':None if pd.isna(amounts['trust'].get(sid)) else float(amounts['trust'][sid]),
                            'foreign_net_est_5d':None if pd.isna(amounts['foreign'].get(sid)) else float(amounts['foreign'][sid])}
                           for sid,p in leaders.iterrows()]
        stories = (news or {}).get('stories',[])
        matched = [s for s in stories if s['source_date'] <= end and set(s['headline_named_ids']) & set(ids)
                   and (not topic or topic in s['themes'])]
        item['news_named_articles'] = len(matched)
        result.append(item)
    return {'as_of':end, 'start':days[0], 'recent_start':days[-5], 'groups':result,
            'available_groups':sum(g['available'] for g in result),
            'news_analyzed_at':(news or {}).get('analyzed_at'),
            'news_end':(news or {}).get('end')}
