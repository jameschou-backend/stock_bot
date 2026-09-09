from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from app import news_research as service
from app.finmind import FinMindQuotaError
from skills.news_radar import analyze, classify, safe_link


def row(title, sid='2408', source='媒體甲', stamp='2025-07-01T04:00:00'):
    return {'stock_id': sid, 'title': title, 'source': source, 'link': 'https://example.com/news',
            'provider_datetime': stamp, 'first_recorded_at': '2026-09-09T04:00:00+00:00'}


def test_negative_and_speculative_headlines_cannot_be_confirmed_beneficiaries():
    negative = classify('南亞科否認DDR4漲價傳言，訂單尚未確認')
    assert negative['status'] == 'negative_or_mixed'
    assert 'memory' in negative['themes']
    assert classify('衛星訂單可望成長 明年量產')['status'] == 'expectation'
    assert classify('【12:20 即時新聞】南亞科漲停 DDR4訂單題材發酵')['status'] == 'price_commentary'


def test_syndicated_title_and_multi_stock_tags_count_once_and_keep_association_uncertainty():
    rows = [row('南亞科 DDR4 出貨增加 - 媒體甲'), row('南亞科DDR4出貨增加 - 媒體乙', source='媒體乙'),
            row('南亞科DDR4出貨增加 - 媒體乙', sid='2344', source='媒體乙')]
    result = analyze(rows, {'2408': '南亞科', '2344': '華邦電'})
    assert result['unique_articles'] == 1 and result['duplicates_collapsed'] == 2
    story = result['stories'][0]
    assert story['stock_ids'] == ['2344', '2408']
    assert story['headline_named_ids'] == ['2408']
    assert story['evidence_level'] == 'headline_only_unverified'
    assert result['themes'][0]['publisher_count'] == 2


def test_etf_and_bad_ids_are_excluded_and_unknown_topics_are_not_promoted():
    result = analyze([row('量子運算題材受矚目'), row('記憶體', sid='00988A'), row('記憶體', sid='0050')],
                     {'2408': '南亞科'})
    assert result['invalid_or_non_stock_rows'] == 2
    assert not result['themes']
    assert result['unclassified_topic_phrases'][0]['phrase'] == '量子運算'
    assert safe_link('javascript:alert(1)') == ''
    assert safe_link('https://secret@example.com') == ''


def test_day_normalization_rejects_wrong_dates_and_preserves_first_observation(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'CACHE', tmp_path)
    now = datetime(2026, 9, 9, 4, tzinfo=timezone.utc)
    frame = pd.DataFrame([{'stock_id': '2408', 'date': '2026-09-08 05:00', 'title': 'DDR4出貨增加'},
                          {'stock_id': '2408', 'date': '2026-09-07 05:00', 'title': '昨日新聞'},
                          {'stock_id': '00988A', 'date': '2026-09-08 05:00', 'title': 'ETF'}])
    a = service.store_day(date(2026, 9, 8), frame, now)
    b = service.store_day(date(2026, 9, 8), frame, now+timedelta(hours=2))
    assert len(b['rows']) == 1 and b['invalid_or_out_of_day_rows'] == 2
    assert b['rows'][0]['first_recorded_at'] == a['rows'][0]['first_recorded_at']


def test_quota_error_stops_immediately_and_completed_days_resume_from_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'CACHE', tmp_path)
    now = datetime(2026, 9, 9, 4, tzinfo=timezone.utc)
    calls = []
    def fetch(dataset, day, **kwargs):
        calls.append(day)
        assert kwargs['max_retries'] == 0
        if day == date(2026, 9, 9): raise FinMindQuotaError(600)
        return pd.DataFrame([{'stock_id': '2408', 'date': str(day)+' 05:00', 'title': 'DDR4漲價'}])
    monkeypatch.setattr(service, 'fetch_dataset', fetch)
    cfg = SimpleNamespace(finmind_token='test', finmind_requests_per_hour=5400)
    with pytest.raises(FinMindQuotaError): service.collect_recent(date(2026,9,9), 2, cfg, now=now)
    with pytest.raises(FinMindQuotaError): service.collect_recent(date(2026,9,9), 2, cfg, now=now)
    assert calls == [date(2026,9,8), date(2026,9,9), date(2026,9,9)]
    assert (tmp_path/'days/2026-09-08.json').exists()


def test_review_cutoff_excludes_same_day_and_never_fetches(monkeypatch, tmp_path):
    monkeypatch.setattr(service, 'CACHE', tmp_path)
    monkeypatch.setattr(service, 'company_names', lambda: {'2408': '南亞科'})
    seen = []
    def window(start, end, sid):
        seen.append((start, end, sid))
        return [row('南亞科DDR4出貨增加')], {'legacy_latest': '2026-05-25'}
    monkeypatch.setattr(service, 'load_window', window)
    monkeypatch.setattr(service, 'price_context', lambda *args: {'available': False})
    monkeypatch.setattr(service, 'fetch_dataset', lambda *a, **k: pytest.fail('offline review fetched data'))
    result = service.run('review', date(2025,7,10), 30, stock_id='2408')
    assert seen == [(date(2025,6,10), date(2025,7,9), '2408')]
    assert result['collection']['network_requests'] == 0
    assert result['live_qualified'] is False


def test_only_today_and_yesterday_use_hourly_refresh_in_taipei(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'CACHE', tmp_path)
    now = datetime(2026, 9, 9, 16, 30, tzinfo=timezone.utc)  # Taipei Sep 10.
    for day in [date(2026,9,8),date(2026,9,9),date(2026,9,10)]:
        frame=pd.DataFrame([{'stock_id':'2408','date':str(day)+' 00:00','title':'記憶體'}])
        service.store_day(day,frame,now-timedelta(hours=2))
    calls=[]
    def fetch(dataset,day,**kwargs):
        calls.append(day)
        return pd.DataFrame([{'stock_id':'2408','date':str(day)+' 00:00','title':'記憶體'}])
    monkeypatch.setattr(service,'fetch_dataset',fetch)
    cfg=SimpleNamespace(finmind_token='test',finmind_requests_per_hour=5400)
    stats=service.collect_recent(date(2026,9,10),3,cfg,now=now)
    assert calls==[date(2026,9,9),date(2026,9,10)]
    assert stats['day_cache_hits']==1


def test_price_context_uses_only_pre_publication_closes_and_keeps_missing_unknown(tmp_path, monkeypatch):
    import numpy as np
    from scripts import research_flow
    monkeypatch.setattr(research_flow, 'INPUT_DIR', tmp_path)
    monkeypatch.setattr(research_flow, 'verify_inputs', lambda: {'files_sha256': {'quotes.parquet': 'test'}})
    days = pd.bdate_range('2025-01-01', periods=80)
    close = pd.DataFrame({'2408': np.linspace(100, 200, 80), '0050': np.linspace(100, 130, 80)}, index=days)
    def save():
        prices=close.rename_axis('trading_date').reset_index().melt('trading_date',var_name='stock_id',value_name='adj_close')
        prices.to_parquet(tmp_path/'quotes.parquet',index=False)
    pd.DataFrame({'stock_id':['2408'],'listed_date':[pd.Timestamp('2000-01-01')]}).to_parquet(tmp_path/'companies.parquet')
    save()
    story={'source_date':str(days[30].date())}
    service.price_context([story],'2408',days[50].date())
    original=story['price_context'].copy()
    assert original['as_of']==str(days[29].date())
    close.iloc[30:,0]=10000  # Includes same-day closing price and everything later.
    save()
    service.price_context([story],'2408',days[50].date())
    assert story['price_context']==original
    close.iloc[9,1]=np.nan
    save()
    service.price_context([story],'2408',days[50].date())
    assert story['price_context'] is None
