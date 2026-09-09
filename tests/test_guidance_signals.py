import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from skills.guidance_signals import build_signals, quarterly_comparison

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    quarterly = json.loads((ROOT/'docs/guidance_quarterly_sources_20260910.json').read_text())['events']
    annual = json.loads((ROOT/'docs/guidance_annual_sources_20260910.json').read_text())['records']
    dates = pd.bdate_range('2022-11-01','2026-06-23')
    close = pd.DataFrame(100., index=dates, columns=['2330','0050'])
    volume = pd.Series(100., index=dates)
    return quarterly, annual, close, volume


def test_compare_actual_only_with_previously_published_same_quarter_company():
    q,_,_,_ = fixture()
    matched = quarterly_comparison(q[5],q[4])
    assert matched['beat'] and matched['prior_usd_high']==18.8
    assert quarterly_comparison(q[5],q[3])['status']=='target_mismatch'
    other = {**q[4], 'stock_id':'2303'}
    with pytest.raises(ValueError,match='same company'):
        quarterly_comparison(q[5],other)
    assert not quarterly_comparison({**q[5],'actual_revenue_usd_billion':18.8},q[4])['beat']
    assert not quarterly_comparison({**q[5],'actual_gross_margin_pct':51.9},q[4])['beat']
    assert quarterly_comparison({**q[5],'availability_status':'missing'},q[4])['status']=='missing_source'
    with pytest.raises(ValueError,match='precede'):
        quarterly_comparison(q[5],{**q[4],'meeting_date':q[5]['meeting_date']})


def test_price_confirmation_uses_past_mean_and_enters_next_session():
    q,a,c,v = fixture()
    c.loc['2024-04-19','2330']=103.
    v.loc['2024-04-19']=121.
    entries, events = build_signals(c,v,q,a)
    event = next(e for e in events if e['event_id']=='2330-2024Q1')
    assert event['confirmation_date']=='2024-04-19'
    assert event['entries']['beat_confirm']=='2024-04-22'
    assert event['confirmation_volume_ratio']==pytest.approx(1.21)
    c.loc['2024-04-22':,'2330']=1000
    _, modified = build_signals(c,v,q,a)
    assert event==next(e for e in modified if e['event_id']==event['event_id'])
    v.loc['2024-04-19']=119.
    _, failed = build_signals(c,v,q,a)
    assert 'beat_confirm' not in next(e for e in failed if e['event_id']==event['event_id'])['entries']


def test_annual_version_delay_and_complete_comparison_key():
    q,a,c,v = fixture()
    _,events=build_signals(c,v,q,a)
    late=next(e for e in events if e['event_id']=='2330-2024Q2')
    assert late['entries']['guidance_up']>'2024-08-31'
    amendment=next(e for e in events if e['event_id']=='2330-2025Q2')
    assert amendment['entries']['guidance_up']>'2025-07-21'
    changed=copy.deepcopy(a)
    changed[6]['period_key']['currency']='TWD'
    with pytest.raises(ValueError,match='currency'):
        build_signals(c,v,q,changed)
    changed=copy.deepcopy(a)
    changed[6]['known_version_date']='2024-12-01'
    _,events=build_signals(c,v,q,changed)
    # The later Q3 revision cannot use a Q2 reference version before it exists.
    assert next(e for e in events if e['event_id']=='2330-2024Q3')['entries']['guidance_up']>'2024-12-01'


def test_missing_quarter_is_retained_and_never_treated_as_beat():
    q,a,c,v=fixture()
    q[5]['availability_status']='missing'
    _,events=build_signals(c,v,q,a)
    assert len(events)==13
    assert events[5]['comparison']['status']=='missing_source'
    assert not events[5]['comparison']['beat']
    assert events[6]['comparison']['status']=='missing_source'
    with pytest.raises(ValueError,match='13 consecutive'):
        build_signals(c,v,q[1:],a)
