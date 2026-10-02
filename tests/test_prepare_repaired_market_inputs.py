import pandas as pd
import pytest
from scripts.prepare_repaired_market_inputs import required_history_days,candidate_prefix_inputs
from skills.market_input_validation import MarketEvidenceError


def test_new_candidate_requires_full_prior_history_without_future_sessions():
    days=pd.bdate_range('2018-01-01','2019-02-01')
    entry=dict(signal_date='2019-01-02')
    required=required_history_days(days,[entry],start='2019-01-02',end='2019-01-02')
    i=days.get_loc('2019-01-02')
    assert required==[str(d.date()) for d in days[i-126:i+1]]
    assert min(required)<'2018-08-15'
    assert max(required)=='2019-01-02'


def test_whole_account_period_stays_required_even_without_candidates():
    days=pd.bdate_range('2018-12-20','2019-01-10')
    required=required_history_days(days,[],start='2019-01-02',end='2019-01-04')
    assert required==['2019-01-02','2019-01-03','2019-01-04']


def test_history_cannot_shorten_warmup_or_accept_duplicate_calendar():
    days=pd.bdate_range('2018-12-20','2019-01-10')
    with pytest.raises(MarketEvidenceError,match='126-session'):
        required_history_days(days,[dict(signal_date='2019-01-02')])
    with pytest.raises(MarketEvidenceError,match='Unique ordered'):
        required_history_days(list(days)+[days[-1]],[])


def test_prefix_retains_only_next_session_calendar_not_any_future_observations():
    days=pd.to_datetime(['2023-12-28','2023-12-29','2024-01-02','2024-01-03'])
    frames={name:pd.DataFrame({'1234':[1.,2.,999.,1000.]},index=days)
            for name in ('raw-close','raw-volume','close-quality','close-official')}
    frames['eligibility']=pd.DataFrame({'1234':[True]*4},index=days)
    groups=dict(entries=[dict(signal_date='2023-12-29'),dict(signal_date='2024-01-02')],
        diffusion=dict(groups=[dict(month='2023-12'),dict(month='2024-01')]))
    prefix,known,signal,entry=candidate_prefix_inputs(frames,groups,'2023-12-31')
    assert (signal,entry)==('2023-12-29','2024-01-02')
    for name,frame in prefix.items():
        assert list(frame.index)==list(days[:3])
        assert frame.iloc[:2].equals(frames[name].iloc[:2])
        assert not frame.iloc[-1].any() if name=='eligibility' else frame.iloc[-1].isna().all()
    assert known['entries']==[dict(signal_date='2023-12-29')]
    assert known['diffusion']['groups']==[dict(month='2023-12')]
    assert frames['raw-close'].iloc[2,0]==999. and len(groups['entries'])==2


def test_prefix_rejects_absent_following_calendar_without_inventing_entry_date():
    frame=pd.DataFrame({'1234':[1.]},index=pd.to_datetime(['2023-12-29']))
    with pytest.raises(MarketEvidenceError,match='following calendar'):
        candidate_prefix_inputs({'raw-close':frame},{},'2023-12-29')
