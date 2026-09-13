from copy import deepcopy
import pytest
from tests.test_capacity_forward import prepare
from tests.test_forward_portfolio import clock
from app import capacity_execution_review as review,capacity_forward as cap,forward_portfolio as p


def report(rows):
    o=next(o for o in p.state(rows)['orders'].values() if o['order_id'].startswith('capacity:') and o['channel']=='board')
    return dict(execution_id='broker-1',order_id=o['order_id'],stock_id=o['stock_id'],side='buy',channel='board',
                qty='1000',price='99',fee='141',tax='0',executed_at='2026-09-14T10:00:00+08:00')


def test_missing_report_is_unknown_and_csv_cannot_turn_simulation_into_real(tmp_path):
    root,_,_=prepare(tmp_path);path=root/'strategy.sqlite3';rows=cap.verify(path)
    r=review.compare(rows,[report(rows)],clock('2026-09-14',11))
    assert r['report_count']==1 and not r['broker_transport_verified']
    assert any(x['reported_qty'] is None for x in r['rows'])
    assert rows==cap.verify(path)


@pytest.mark.parametrize('problem',['duplicate','future','side','overfill','fraction'])
def test_reject_invalid_execution_evidence(tmp_path,problem):
    root,_,_=prepare(tmp_path);rows=cap.verify(root/'strategy.sqlite3');e=report(rows);data=[e]
    if problem=='duplicate':data.append(deepcopy(e))
    if problem=='future':e['executed_at']='2026-09-15T10:00:00+08:00'
    if problem=='side':e['side']='sell'
    if problem=='overfill':e['qty']='99999000'
    if problem=='fraction':e['qty']='1.1'
    with pytest.raises(ValueError):review.compare(rows,data,clock('2026-09-14',11))


def test_limit_violation_is_visible_not_hidden_or_posted(tmp_path):
    root,_,_=prepare(tmp_path);rows=cap.verify(root/'strategy.sqlite3');e=report(rows);e['price']='101'
    r=review.compare(rows,[e],clock('2026-09-14',11))
    assert any(x['limit_breach'] for x in r['rows'])
