"""Candidate/financial seams: preserve matching populations and stop mismatches."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from scripts.research_rally_rank_accounts import (
    map_ranked_entries, verify_financial_prefix, audit_ranked_account, preserve_failure,
)
from skills.rally_ranking import build_rankings


def fixture_events():
    return pd.DataFrame([
        dict(cohort='original_red',event_id='red-2024-01-02-1101',stock_id='1101',
             signal_date='2024-01-02',relative_return20=.4,peer_breadth_value=.1,
             peer_turnover_multiple=1.,flow_ratio5_lag1=-.1),
        dict(cohort='original_red',event_id='red-2024-01-02-1102',stock_id='1102',
             signal_date='2024-01-02',relative_return20=.2,peer_breadth_value=.8,
             peer_turnover_multiple=2.,flow_ratio5_lag1=.2),
        dict(cohort='original_red',event_id='red-2024-01-02-1103',stock_id='1103',
             signal_date='2024-01-02',relative_return20=.5,peer_breadth_value=np.nan,
             peer_turnover_multiple=2.,flow_ratio5_lag1=.3),
    ])


def test_ranks_reorder_same_candidates_before_reservations_and_slots():
    ranked=build_rankings(fixture_events())
    groups,audit=map_ranked_entries(ranked,pd.bdate_range('2024-01-01','2024-01-05'),
                                   start='2024-01-02',end='2024-01-05')
    rs=groups['original_red__rs__3'];context=groups['original_red__context__3']
    assert [e['members'][0] for e in rs]==['1101','1102']
    assert [e['members'][0] for e in context]==['1102','1101']
    assert all(e['entry_date']=='2024-01-03' for e in rs+context)
    assert groups['original_red__rs__5']==rs
    assert {e['event_id'] for e in context}=={e['event_id'] for e in rs}
    assert audit['unknown_evidence']==1
    assert all(e['leader_evidence']['information_cutoff']=='2024-01-02' for e in rs)


def test_financial_end_cannot_manufacture_next_session():
    groups,audit=map_ranked_entries(build_rankings(fixture_events()),
        pd.bdate_range('2024-01-01','2024-01-02'),start='2024-01-02',end='2024-01-02')
    assert not any(groups.values())
    assert audit['outside_financial_entry_scope']==2


def test_duplicate_coordinate_fails_even_if_event_ids_differ():
    events=fixture_events();duplicate=events.iloc[[0]].copy();duplicate['event_id']='duplicate'
    ranked=build_rankings(pd.concat([events,duplicate],ignore_index=True))
    with pytest.raises(ValueError,match='Duplicate signal coordinate'):
        map_ranked_entries(ranked,pd.bdate_range('2024-01-01','2024-01-05'),
                           start='2024-01-02',end='2024-01-05')


def prefix_fixture():
    days=pd.bdate_range('2024-01-01',periods=3)
    quotes=pd.DataFrame([dict(date=d,stock_id='1101',open=10.,high=11.,low=9.,close=10.,volume=100.) for d in days])
    adjusted=pd.DataFrame({'1101':[10.,11.,np.nan]},index=days)
    eligible=pd.DataFrame({'1101':[True,True,False]},index=days)
    return quotes,quotes.copy(),adjusted,adjusted.copy(),eligible,eligible.copy()


def test_exact_prefix_including_nan_is_allowed():
    assert verify_financial_prefix(*prefix_fixture())['raw_coordinates']==3


@pytest.mark.parametrize('field',['raw','adjusted','eligible','missing_raw','extra_raw'])
def test_financial_source_conflicts_stop_account_preparation(field):
    inputs=list(prefix_fixture())
    if field=='raw':inputs[1].loc[0,'close']=10.1
    if field=='adjusted':inputs[3].iloc[0,0]=10.1
    if field=='eligible':inputs[5].iloc[0,0]=False
    if field=='missing_raw':inputs[1]=inputs[1].iloc[1:]
    if field=='extra_raw':
        extra=inputs[1].iloc[[0]].copy();extra['stock_id']='1102';inputs[1]=pd.concat([inputs[1],extra])
    with pytest.raises(ValueError,match='differs|extra'):
        verify_financial_prefix(*inputs)


def test_failed_account_never_exposes_partial_return():
    class Partial:
        daily=[dict(date='2024-01-02',nav=1500000.)]
        holdings={};trades=[];orders=[];cash_ledger=[];actions=[];holding_rows=[]
        cohorts=[];receivables=[];resource_plans=[];tick_plans=[];day_plans={}
    result=preserve_failure(RuntimeError('Missing quote evidence'),Partial())
    assert result['summary'] is None and result['completed'] is False
    assert result['last_date']=='2024-01-02'
    assert result['partial_journal']['daily'][0]['nav']==1500000.


def test_audit_detects_reordered_reservations_or_idle_etf():
    groups,_=map_ranked_entries(build_rankings(fixture_events()),
        pd.bdate_range('2024-01-01','2024-01-05'),start='2024-01-02',end='2024-01-05')
    entries=groups['original_red__rs__3'];ids=[e['event_id'] for e in entries]
    account=dict(cohorts=[],trades=[],selection_decisions=[dict(date='2024-01-03',
        original_event_ids=ids,selected_event_ids=ids)])
    assert audit_ranked_account(account,entries)['ranking_before_reservations']
    bad=deepcopy(account);bad['selection_decisions'][0]['original_event_ids']=ids[::-1]
    with pytest.raises(ValueError,match='order differs'):audit_ranked_account(bad,entries)
    bad=deepcopy(account);bad['trades']=[dict(side='buy',stock_id='0050')]
    with pytest.raises(ValueError,match='Idle 0050'):audit_ranked_account(bad,entries)


def test_strict_execution_stops_inherited_missing_source_exclusion(monkeypatch):
    from scripts.research_rally_rank_accounts import StrictRankRangeOrders, RangeGapOrders
    from skills.replay_market_feeds import ReplayDataUnavailable
    engine=object.__new__(StrictRankRangeOrders);engine.data_gap_exclusions=[]
    def underlying(self,*args,**kwargs):
        self.data_gap_exclusions.append(dict(date='2024-01-03',stock_id='1101',
            side='buy',failure_reason='Official odd source missing'))
        return 0
    monkeypatch.setattr(RangeGapOrders,'_execute_order',underlying)
    with pytest.raises(ReplayDataUnavailable,match='Official odd source missing'):
        engine._execute_order()
    assert len(engine.data_gap_exclusions)==1  # diagnostic preserved, no fabricated return


@pytest.mark.parametrize('quantity',[0,100,1000])
def test_strict_execution_allows_observed_capacity_or_limit_results(monkeypatch,quantity):
    from scripts.research_rally_rank_accounts import StrictRankRangeOrders, RangeGapOrders
    engine=object.__new__(StrictRankRangeOrders);engine.data_gap_exclusions=[]
    monkeypatch.setattr(RangeGapOrders,'_execute_order',lambda self,*a,**kw:quantity)
    assert engine._execute_order()==quantity


def test_completed_account_cannot_retain_inherited_gap_metadata(monkeypatch):
    from scripts.research_rally_rank_accounts import StrictRankRangeOrders, RangeGapOrders
    from skills.replay_market_feeds import ReplayDataUnavailable
    engine=object.__new__(StrictRankRangeOrders);engine.data_gap_exclusions=[]
    monkeypatch.setattr(RangeGapOrders,'run',lambda self:dict(settings=dict(posthoc_data_exclusion=True)))
    account=engine.run()
    assert account['settings']['posthoc_data_exclusion'] is False
    assert account['settings']['missing_odd_policy']=='halt_account_without_partial_return'
    engine.data_gap_exclusions.append(dict(failure_reason='missing'))
    with pytest.raises(ReplayDataUnavailable,match='cannot complete'):engine.run()
