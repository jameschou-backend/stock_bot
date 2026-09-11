"""End-to-end accounting on isolated fixtures; never writes real paper books."""
from decimal import Decimal as D
from streamlit.testing.v1 import AppTest
from app import forward_journal as j, forward_portfolio as p, forward_halts as h, forward_restatement as r
from tests.test_forward_portfolio import setup,buy,fill,cancel,close,clock
from tests.test_forward_halts import EVIDENCE
from tests.test_forward_restatement import PROOF


def test_partial_fill_halt_dividend_split_delivery_and_restatement(tmp_path):
    path=tmp_path/'original'
    with j.connection(path) as con:
        setup(con);o=buy(con,n=999);trade=fill(con,o,400,'100');cancel(con,o)
        close(con,'2026-09-14',{'2492':'100','0050':'100'})
    h.save_notice(path,dict(stock_id='2492',halt_start='2026-09-15',resume_date='2026-09-17',withdrawn=False),EVIDENCE,clock('2026-09-14'))
    with j.connection(path) as con:
        p.submit(con,dict(kind='entitlement',id='div',action_id='div',action_type='cash',stock_id='2492',ex_date='2026-09-15',delivery_date='2026-09-17',eligible_qty=400,cash_per_share='10',amount='4000',evidence='fixture'),clock('2026-09-15',8))
        for day in ['2026-09-15','2026-09-16']:
            rows=j.read_events(con);prices,details=h.valuation_prices(rows,day,{'0050':'100'})
            body=dict(date=day,prices=prices,estimated_prices=details,actions_reviewed=True,source='fixture')
            nav=p.valuation({**p.state(rows),'mark':{'body':body}})
            assert nav==D('999980')
            p.submit(con,dict(kind='close',id=day,**body,nav=str(nav)),clock(day))
        p.submit(con,dict(kind='entitlement',id='split',action_id='split',action_type='split',stock_id='2492',ex_date='2026-09-17',delivery_date='2026-09-17',eligible_qty=400,ratio='4',result_qty=1600,evidence='fixture'),clock('2026-09-17',8))
        assert p.valuation(p.state(j.read_events(con))) is None
        for key in ['div','split']:
            p.submit(con,dict(kind='delivery',id='pay-'+key,action_id=key,date='2026-09-17',evidence='actual fixture receipt'),clock('2026-09-17',8))
        close(con,'2026-09-17',{'2492':'22.5','0050':'100'})
    original=r.read(path)
    preview=r.preview(path,[dict(op='replace',target=trade['hash'],body=dict(trade['body'],fee='30'))],PROOF,clock('2026-09-18'))
    assert preview['after']['nav']=='999970.0'
    derived=r.materialize(path,preview['command'],tmp_path/'versions',clock('2026-09-18'))
    assert r.read(path)==original
    after=p.summary(derived,clock('2026-09-18'))
    assert after['cash']=='963970' and after['holdings'][0]['qty']==1600
    assert h.compare(derived,derived)['ready'] is False


def test_evidence_ui_initial_render_no_side_effects(tmp_path,monkeypatch):
    from app import forward_evidence_ui as ui
    path=tmp_path/'book'
    with j.connection(path) as con:setup(con);buy(con,n=999)
    original=r.read(path)
    monkeypatch.setattr(ui.odd,'history',lambda:[])
    src=f'''from pathlib import Path
from app.forward_evidence_ui import render_halts,render_odd,render_restatement
p=Path({str(path)!r})
render_halts(p,'test')
render_odd(p,'test')
render_restatement(p,'test')
'''
    app=AppTest.from_string(src).run()
    assert not app.exception
    labels=[x.label for x in app.expander]
    assert len(labels)==3
    assert r.read(path)==original
    assert any('取得此股票零股五檔'==b.label for b in app.button)


def test_restatement_ui_preview_and_save_are_separate(tmp_path,monkeypatch):
    from app import forward_evidence_ui as ui
    from tests.test_forward_restatement import account
    path,f=account(tmp_path);original=r.read(path)
    actual_preview=ui.restatement.preview;actual_save=ui.restatement.materialize
    monkeypatch.setattr(ui.restatement,'preview',lambda path,ops,proof:actual_preview(path,ops,proof,clock('2026-09-15')))
    monkeypatch.setattr(ui.restatement,'materialize',lambda path,cmd:actual_save(path,cmd,tmp_path/'versions',clock('2026-09-15')))
    app=AppTest.from_string(f'from pathlib import Path\nfrom app.forward_evidence_ui import render_restatement\nrender_restatement(Path({str(path)!r}),"test")').run()
    app.selectbox(key='repair_target_test').select(f).run()
    import json
    app.text_area(key='repair_body_test'+f['hash']).set_value(json.dumps(dict(f['body'],fee='30')))
    for suffix,field in [('ref','reference'),('reviewer','reviewer'),('reason','reason')]:app.text_input(key='repair_test'+suffix).set_value(PROOF[field])
    app.text_area(key='repair_testtext').set_value(PROOF['text'])
    app.button(key='repair_preview_btn_test').click().run()
    assert not app.exception and not app.error and r.read(path)==original
    assert not list((tmp_path/'versions').glob('*.sqlite3'))
    app.button(key='repair_save_test').click().run()
    assert not app.exception and not app.error
    assert len(list((tmp_path/'versions').glob('*.sqlite3')))==1 and r.read(path)==original
