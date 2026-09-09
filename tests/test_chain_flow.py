from datetime import date
from types import SimpleNamespace
import pandas as pd
import pytest
from app import chain_flow_research as service
from app.finmind import FinMindQuotaError
from skills.chain_flow import institutional_net, summarize


def inputs():
    days=pd.bdate_range('2026-07-01',periods=25).strftime('%Y-%m-%d').tolist()
    flow=pd.DataFrame([{'date':day,'industry':'測試','sub_industry':'','stock_count':2,
                        'trading_money':1000,'trading_money_pct':1 if i<20 else 2}
                       for i,day in enumerate(days)])
    members=pd.DataFrame([{'stock_id':'2408','industry':'測試','sub_industry':'甲'},
                          {'stock_id':'2408','industry':'測試','sub_industry':'乙'},
                          {'stock_id':'2344','industry':'測試','sub_industry':'甲'}])
    prices=pd.DataFrame([{'stock_id':sid,'date':days[-1],'Trading_money':500,'open':9,'close':10}
                         for sid in ['2408','2344']])
    closes=pd.DataFrame([{'stock_id':sid,'date':day,'close':10} for day in days[-5:] for sid in ['2408','2344']])
    inst=pd.DataFrame([{'stock_id':sid,'date':day,'name':kind,'buy':2,'sell':1}
                       for day in days[-5:] for sid in ['2408','2344']
                       for kind in ['Investment_Trust','Foreign_Investor','Foreign_Dealer_Self','Dealer']])
    return flow,members,prices,closes,inst,{'2408':'南亞科','2344':'華邦電'}


def test_nonoverlapping_share_comparison_deduplicates_members_and_separates_investor_groups():
    result=summarize(*inputs())['groups'][0]
    assert result['share_multiple']==2 and result['share_change_pp']==1
    assert result['members']==2 and result['top1_share']==.5
    assert result['trust_observed_net_est_5d']==100
    assert result['foreign_observed_net_est_5d']==200  # Dealer excluded, foreign self included once.
    assert result['reading']=='成交擴散，至少一類法人偏買'


def test_missing_category_or_day_is_unknown_and_incomplete_components_block_reading():
    args=list(inputs());inst=args[4]
    args[4]=inst[~((inst.stock_id=='2408') & (inst.name=='Investment_Trust') & (inst.date==inst.date.max()))]
    result=summarize(*args)['groups'][0]
    assert result['trust_coverage']==.5
    assert result['trust_observed_net_est_5d']==50
    assert not result['flow_data_ready'] and result['reading']=='資料待核對'
    assert next(x for x in result['leaders'] if x['stock_id']=='2408')['trust_net_est_5d'] is None
    args=list(inputs());args[2]=args[2].iloc[:1]
    result=summarize(*args)['groups'][0]
    assert not result['components_reconciled']
    assert result['reading']=='資料待核對'


def test_different_dates_duplicate_rows_and_absent_foreign_component_fail_closed():
    args=list(inputs());args[2]['date']='2026-12-31'
    with pytest.raises(ValueError,match='日期不同'): summarize(*args)
    inst=inputs()[4]
    with pytest.raises(ValueError,match='重複'): institutional_net(pd.concat([inst,inst.iloc[:1]]))
    net=institutional_net(inst[inst.name!='Foreign_Dealer_Self'])
    assert net.foreign_net.isna().all()


def test_quota_stop_keeps_completed_snapshots_and_offline_reuses_them(tmp_path,monkeypatch):
    monkeypatch.setattr(service,'CACHE',tmp_path)
    cfg=SimpleNamespace(finmind_token='test',finmind_requests_per_hour=5400)
    stats={'network_requests':0,'file_cache_hits':0,'gateway_cache_hits':0,'inputs':{}}
    calls=[]
    def fetch(dataset,day,**kwargs):
        calls.append(day)
        assert kwargs['max_retries']==0
        if day==date(2026,9,8): raise FinMindQuotaError(600)
        frame=pd.DataFrame([{'date':str(day),'stock_id':'2408'}]);frame.attrs={'retrieved_at':1,'cache_hit':False}
        return frame
    monkeypatch.setattr(service,'fetch_dataset',fetch)
    service.read_or_fetch('day7','test',date(2026,9,7),cfg,stats,fetch=True)
    with pytest.raises(FinMindQuotaError): service.read_or_fetch('day8','test',date(2026,9,8),cfg,stats,fetch=True)
    service.read_or_fetch('day7','test',date(2026,9,7),None,stats,fetch=False)
    assert len(calls)==2 and stats['file_cache_hits']==1
    with pytest.raises(ValueError,match='缺少'): service.read_or_fetch('day8','test',date(2026,9,8),None,stats,fetch=False)
