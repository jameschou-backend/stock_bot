import numpy as np
import pandas as pd
import pytest
from scripts.research_signal_rank_20261003 import (
    rank_events,rank_band,score_bin,matched_days,arm_result,
)


def fixture(n=12):
    days=pd.bdate_range('2024-01-01',periods=45)
    ids=[str(2000+i) for i in range(n)]
    frame=pd.DataFrame(100.,index=days,columns=['0050',*ids])
    events=[]
    for i,sid in enumerate(ids):
        score=(n-i)/100
        frame.loc[days[30],sid]=100*(1+score)
        events.append(dict(event_id='event-'+sid,signal_date=str(days[30].date()),members=[sid],priority=score))
    return frame,events


def test_rank_uses_all_original_rows_before_future_outcome_filters():
    frame,events=fixture(12)
    events[0]['status']='open';events[1]['status']='unresolved';events[2]['net_return']=-1
    ranks=rank_events(events,frame)
    assert ranks['event-2000']['daily_rank']==1 and ranks['event-2001']['daily_rank']==2
    assert {r['daily_candidate_count'] for r in ranks.values()}=={12}
    altered=[dict(e,status='closed',net_return=100,exit_date='2099-12-31') for e in reversed(events)]
    assert ranks==rank_events(altered,frame)


def test_prefix_and_modified_future_do_not_change_ranks():
    frame,events=fixture();cut=pd.Timestamp(events[0]['signal_date']);base=rank_events(events,frame)
    assert base==rank_events(events,frame.loc[:cut])
    changed=frame.copy();changed.loc[changed.index>cut]=np.nan
    assert base==rank_events(events,changed)
    future=dict(events[0],event_id='future-event',signal_date=str(frame.index[35].date()),priority=.2)
    extended=frame.copy();extended.at[frame.index[35],future['members'][0]]=120
    ranked=rank_events([*events,future],extended)
    assert base=={k:v for k,v in ranked.items() if k!='future-event'}


def test_equal_priority_ties_use_event_id_and_not_input_order():
    frame,events=fixture(2);frame.at[frame.index[30],'2001']=frame.at[frame.index[30],'2000'];events[1]['priority']=events[0]['priority']
    events[0]['event_id']='z';events[1]['event_id']='a'
    values=rank_events(events,frame)
    assert values['a']['daily_rank']==1 and values['z']['daily_rank']==2


@pytest.mark.parametrize('bad',[None,float('nan'),float('inf'),0,-.1,True])
def test_missing_invalid_priority_is_an_error_not_a_dropped_candidate(bad):
    frame,events=fixture();events[0]['priority']=bad
    with pytest.raises(ValueError):rank_events(events,frame)


def test_priority_must_match_recomputed_signal_close_relative20():
    frame,events=fixture();events[0]['priority']+=.01
    with pytest.raises(ValueError,match='differs from T0'):rank_events(events,frame)


def test_quintiles_use_midpoint_percentile_and_sparse_days_remain_unclassified():
    frame,events=fixture(10);ranked=rank_events(events,frame)
    assert [r['rank_quintile'] for r in ranked.values()]==['Q1_top20pct']*2+['Q2']*2+['Q3']*2+['Q4']*2+['Q5_bottom20pct']*2
    assert ranked['event-2000']['daily_midpoint_percentile']==.05
    frame,events=fixture(1);value=rank_events(events,frame)['event-2000']
    assert value['daily_rank']==1 and value['daily_candidate_count']==1
    assert value['rank_quintile'] is None and value['rank_keep_Q1_top20pct'] is None
    assert value['rank_keep_top5'] is True


def test_fixed_rank_and_score_boundaries():
    assert [rank_band(i) for i in [1,2,3,4,5,6,10,11,20,21]]==['rank1','rank2','rank3','rank4_5','rank4_5','rank6_10','rank6_10','rank11_20','rank11_20','rank21_plus']
    assert [score_bin(x) for x in [.1,.10001,.2,.20001,.4,.40001]]==['score_0_10pp','score_10_20pp','score_10_20pp','score_20_40pp','score_20_40pp','score_above40pp']


def with_outcomes(ranks):
    return [dict(r,status='closed',net_return=.4 if r['daily_rank']<=3 else -.1,
                 holding_days=5,exit_reason='three_black') for r in ranks.values()]


def test_matched_comparison_rejects_day_if_any_original_top10_outcome_unknown():
    frame,events=fixture(12);rows=with_outcomes(rank_events(events,frame));rows[11]['status']='open'
    matched=matched_days(rows)
    assert matched['coverage']['matched_days']==1
    assert matched['scopes']['all']['mean_paired_difference']==pytest.approx(.5)
    rows[9]['status']='unresolved'
    matched=matched_days(rows)
    assert matched['coverage']['matched_days']==0
    assert matched['excluded'][0]['reason']=='not_all_original_top10_closed'


def test_top3_conditionals_and_equal_opportunity_denominator_are_distinct():
    frame,events=fixture(10);rows=with_outcomes(rank_events(events,frame))
    value=arm_result(rows,'top3')
    assert value['p_win_given_selected']==1 and value['p_selected_given_win']==1
    assert value['selected_return30_count']==3 and value['selected_return30_rate']==1
    assert value['equal_units_opportunities']['equal_units_mean_per_all_original_opportunity']==pytest.approx(.12)
    assert value['kept']['mean_net_return']==pytest.approx(.4)
    rows[9]['net_return']=.2
    value=arm_result(rows,'top3')
    assert value['p_win_given_selected']==1 and value['p_selected_given_win']==.75


def test_quintile_population_reports_low_candidate_day_exclusion():
    frame,events=fixture(10);rows=with_outcomes(rank_events(events,frame))
    frame2,events2=fixture(1);extra=with_outcomes(rank_events(events2,frame2))[0]
    extra['signal_date']='2024-02-14';rows.append(extra)
    value=arm_result(rows,'Q1_top20pct')
    assert value['scope_coverage']['all_signals']==11
    assert value['scope_coverage']['population_signals']==10
    assert value['scope_coverage']['excluded_low_candidate_days']==1
