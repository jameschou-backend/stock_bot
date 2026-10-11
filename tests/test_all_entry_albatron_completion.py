from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skills.all_entry_accounts import digest
from skills.all_entry_albatron_completion import KEY, URLS, load_albatron, install_albatron
from skills.scenario_exit_replay import FractionalCashActions

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = 'docs/all_entry_5386_delivery_20261011.json'


@pytest.fixture
def package(tmp_path):
    doc = json.loads((ROOT/DOCUMENT).read_text())
    for kind, item in doc['evidence'].items():
        path = tmp_path/item['path']; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'%PDF synthetic fixture')
        receipt = tmp_path/item['receipt_path']
        receipt.write_text(json.dumps(dict(url=URLS[kind],http_status=200,redirected=False,sha256=digest(path))))
        item.update(sha256=digest(path),receipt_sha256=digest(receipt))
        if kind == 'annual':
            text = tmp_path/item['text_path']
            text.write_text('盈餘轉增資發行新股案。執行情形：決議通過。訂定113年8月30日為除權基準日，113年9月27日為股票發放日。')
            item['text_sha256'] = digest(text)
    target = tmp_path/DOCUMENT; target.parent.mkdir()
    def save(): target.write_text(json.dumps(doc))
    save()
    return tmp_path,doc,save


@pytest.mark.parametrize('field,value', [
    ('pay_date','2024-08-30'),('ordinary_share_available_date','2024-08-22'),
    ('shares_per_share',.9),('fractional_cash_per_share',10),
    ('fractional_net_zero_verified',False),('fractional_cash_pay_date','2024-09-27'),
    ('fractional_cash_available_for_trading',True),('use_scope','selection'),
])
def test_no_early_delivery_or_invented_fractional_cash(package,field,value):
    root,doc,save = package; doc['replacement'][field] = value; save()
    with pytest.raises(ValueError):load_albatron(root,root/DOCUMENT)


@pytest.mark.parametrize('change',['bytes','wrong_issuer','wrong_payment_text','unapproved','certificate'])
def test_primary_evidence_is_rechecked(package,change):
    root,doc,save = package
    item = doc['evidence']['annual']
    if change == 'bytes': (root/item['path']).write_bytes(b'%PDF changed')
    elif change == 'wrong_issuer':
        p=root/item['receipt_path']; r=json.loads(p.read_text());r['url']='https://example.com/report.pdf'
        p.write_text(json.dumps(r));item['receipt_sha256']=digest(p)
    elif change == 'wrong_payment_text':
        p=root/item['text_path'];p.write_text('113年9月27日僅為現金股利發放日');item['text_sha256']=digest(p)
    else:
        doc['evidence']['minutes']['reviewed_facts'][
            'approved' if change == 'unapproved' else 'same_rights_as_ordinary'] = False
    save()
    with pytest.raises(ValueError):load_albatron(root,root/DOCUMENT)


def test_fractional_book_entry_right_does_not_create_cash(package):
    root,doc,_=package; row,_=load_albatron(root,root/DOCUMENT)
    provider=SimpleNamespace(corporate_overrides={})
    install_albatron(provider,root/DOCUMENT,root)
    source=SimpleNamespace(overrides={KEY:row},on_date=lambda s,d:[dict(row,kind='stock_dividend',action_id='test',stock_id=s,date=d)])
    actions=FractionalCashActions(source,SimpleNamespace(holdings={'5386':{'qty':123}})).on_date('5386','2024-08-22')
    assert len(actions)==1 and actions[0]['pay_date']=='2024-09-27'
    assert actions[0]['fractional_settlement']['gross_cash_amount']==0
    assert actions[0]['fractional_settlement']['fraction']==.07
    assert doc['replacement']==row and provider.corporate_overrides=={KEY:row}
    provider.corporate_overrides[KEY]=deepcopy(row);provider.corporate_overrides[KEY]['pay_date']='2024-08-23'
    with pytest.raises(ValueError,match='rewrite'):install_albatron(provider,root/DOCUMENT,root)


def test_actual_primary_documents():
    if not (ROOT/'.cache/all-entry-accounts-20261011/corporate-evidence/5386/annual-report-2024.pdf').exists():
        pytest.skip('Primary PDF evidence is retained in the local research cache')
    row,refs=load_albatron(ROOT,ROOT/DOCUMENT)
    assert len(refs)==6 and row['pay_date']=='2024-09-27'
