from datetime import datetime, timezone
from copy import deepcopy
import pandas as pd
import pytest
from app import forward_journal as j

CLOCK=lambda:datetime(2026,9,11,12,tzinfo=timezone.utc)
STATUS=dict(data_ready=True,problems=[],price_date='2026-09-11')
SIGNALS=dict(strategy=j.RULES['selection'],signal_end='2026-09-11',source_kind='original_rule_forward_extension',next_session='2026-09-14',entries=[
    dict(signal_date='2026-09-11',entry_date='2026-09-14',members=['2492'])])


def test_freeze_idempotence_and_no_overwrite(tmp_path):
    with j.connection(tmp_path/'j.db') as con:
        a=j.freeze(con,STATUS,SIGNALS,'source',CLOCK)
        assert j.freeze(con,STATUS,SIGNALS,'source',CLOCK)==a
        with pytest.raises(ValueError,match='overwritten'):
            j.freeze(con,STATUS,SIGNALS,'revised',CLOCK)
        assert len(j.read_events(con))==2


def test_stale_signal_and_missing_market_are_gaps(tmp_path):
    with j.connection(tmp_path/'j.db') as con:
        old=deepcopy(SIGNALS);old['signal_end']='2026-09-09'
        e=j.freeze(con,STATUS|{'data_ready':False},old,'source',CLOCK)
        assert e['kind']=='blocked'
        with pytest.raises(ValueError,match='prospective'):
            j.order(con,e['hash'],'2492','buy','odd',10,100,'2026-09-14',CLOCK)


def test_order_budget_channels_cancellation(tmp_path):
    with j.connection(tmp_path/'j.db') as con:
        signal=j.freeze(con,STATUS,SIGNALS,'source',CLOCK)
        order=j.order(con,signal['hash'],'2492','buy','board',1000,300,'2026-09-14',CLOCK)
        assert j.order(con,signal['hash'],'2492','buy','board',1000,300,'2026-09-14',CLOCK)==order
        with pytest.raises(ValueError,match='Combined'):
            j.order(con,signal['hash'],'2492','buy','odd',200,300,'2026-09-14',CLOCK)
        j.cancel(con,order['hash'],'no fill before close',CLOCK)
        with pytest.raises(ValueError,match='1000'):
            j.order(con,signal['hash'],'2492','buy','odd',1000,100,'2026-09-14',CLOCK)


def test_hash_chain_tampering_detected(tmp_path):
    with j.connection(tmp_path/'j.db') as con:
        j.freeze(con,STATUS,SIGNALS,'source',CLOCK)
        con.execute("UPDATE events SET body='{}' WHERE seq=2")
        with pytest.raises(ValueError,match='integrity'): j.read_events(con)


def test_quote_is_never_fill_and_old_trade_visible(tmp_path):
    frame=pd.DataFrame([dict(stock_id='0050', date='2026-09-11 13:30:00',
        buy_price=100,buy_volume=3,sell_price=101,sell_volume=2)])
    with j.connection(tmp_path/'j.db') as con:
        e=j.snapshot(con,frame,CLOCK)
        assert not e['body']['fill_evidence']
        assert e['body']['quotes'][0]['stale_trade']
        assert not e['body']['quotes'][0]['odd_lot_compatible']
    assert j.summary(tmp_path/'j.db')['confirmed_fills']==0


def test_partial_fill_reports_deduplicate_and_respect_remaining(tmp_path):
    with j.connection(tmp_path/'j.db') as con:
        signal=j.freeze(con,STATUS,SIGNALS,'source',CLOCK)
        order=j.order(con,signal['hash'],'2492','buy','odd',500,100,'2026-09-14',CLOCK)
        at='2026-09-14T02:00:00+00:00'
        clock=lambda:datetime(2026,9,14,2,1,tzinfo=timezone.utc)
        args=(con,order['hash'],200,99,20,0,at,{'source':'paper_execution_report','report_id':'a'},clock)
        first=j.record_fill(*args)
        assert j.record_fill(*args)==first
        with pytest.raises(ValueError,match='remaining'):
            j.record_fill(con,order['hash'],400,99,20,0,at,{'source':'paper_execution_report','report_id':'b'},clock)
        with pytest.raises(ValueError,match='quotes'):
            j.record_fill(con,order['hash'],100,99,20,0,at,{'source':'FinMind','report_id':'c'},clock)
