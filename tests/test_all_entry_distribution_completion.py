from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from skills.all_entry_accounts import digest
from skills.all_entry_distribution_completion import load_distributions, install_distributions
from skills.scenario_exit_replay import FractionalCashActions

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = 'docs/all_entry_distribution_completion_20261011.json'


@pytest.fixture
def package(tmp_path):
    doc = json.loads((ROOT/DOCUMENT).read_text())
    for key, facts in doc['evidence'].items():
        sid = key[:4]
        issuer, exchange = facts['issuer'], facts['listing']
        for name in facts['source_sha256']:
            (tmp_path/name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path/issuer['path']).write_bytes(b'%PDF synthetic issuer fixture')
        (tmp_path/issuer['text_path']).write_text('\n'.join(issuer['required_text']))
        host = 'www.makalot.com.tw' if sid == '1477' else 'ir.tbb.com.tw'
        (tmp_path/issuer['receipt_path']).write_text(json.dumps(dict(
            url='https://'+host+'/minutes.pdf', http_status=200, redirected=False,
            sha256=digest(tmp_path/issuer['path']))))
        # This is a synthetic exchange response, not evidence for a real event.
        pay, announced, delivered = ('113年10月18日','113年10月09日','113年10月04日') if sid == '1477' else (
            '115年09月10日','115年09月07日','115年08月28日')
        rows = [[sid, 'fixture', '(股票上市)增資新股上市', 1, 'fixture', announced, pay,
                 str(exchange['ordinary_quantity']), '1', '與原已發行之股份相同 '+delivered]]
        if sid == '2834':
            rows.append([sid,'fixture','新股權利證書上市',2,'fixture','115年08月27日','115年09月02日',
                         str(exchange['ordinary_quantity']),'1','fixture'])
        (tmp_path/exchange['path']).write_text(json.dumps(dict(stat='OK', data=rows)))
        (tmp_path/exchange['receipt_path']).write_text(json.dumps(dict(
            url='https://www.twse.com.tw/rwd/zh/announcement/LT185', http_status=200,
            accepted=True, security_denied=False, automatic_redirects_disabled=True,
            params={'stockNo':sid}, raw_sha256=digest(tmp_path/exchange['path']))))
        facts['source_sha256'] = {name:digest(tmp_path/name) for name in facts['source_sha256']}
    target = tmp_path/DOCUMENT
    target.parent.mkdir(exist_ok=True)
    def save(): target.write_text(json.dumps(doc))
    save()
    return tmp_path, doc, save


@pytest.mark.parametrize('key,field,value', [
    ('2834-2026-08-04','pay_date','2026-09-02'),
    ('2834-2026-08-04','ordinary_share_available_date','2026-09-02'),
    ('2834-2026-08-04','certificate_trading_modeled',True),
    ('2834-2026-08-04','fractional_cash_net_amount_verified',True),
    ('2834-2026-08-04','fractional_cash_available_for_trading',True),
    ('2834-2026-08-04','fractional_cash_pay_date','2026-09-10'),
    ('2834-2026-08-04','shares_per_share',.7),
    ('1477-2024-09-12','fractional_cash_per_share',10),
    ('1477-2024-09-12','fractional_net_zero_verified',False),
])
def test_no_early_ordinary_shares_or_unverified_cash(package,key,field,value):
    root,doc,save = package
    doc['overrides'][key][field] = value
    save()
    with pytest.raises(ValueError):load_distributions(root,root/DOCUMENT)


@pytest.mark.parametrize('change',['bytes','rehashed_http_error','listing_quantity','listing_date','missing_text'])
def test_primary_source_identity_and_facts_are_rechecked(package,change):
    root,doc,save = package
    facts = doc['evidence']['2834-2026-08-04']
    if change == 'bytes':
        (root/facts['issuer']['path']).write_bytes(b'%PDF modified')
    elif change == 'missing_text':
        path = root/facts['issuer']['text_path']; path.write_text('unrelated primary text')
        facts['source_sha256'][str(path.relative_to(root))] = digest(path)
    elif change == 'rehashed_http_error':
        path = root/facts['listing']['receipt_path']; meta = json.loads(path.read_text())
        meta['http_status'] = 403; path.write_text(json.dumps(meta))
        facts['source_sha256'][str(path.relative_to(root))] = digest(path)
    else:
        path = root/facts['listing']['path']; data = json.loads(path.read_text())
        data['data'][0][7 if change == 'listing_quantity' else 6] = '1' if change == 'listing_quantity' else '115年09月02日'
        path.write_text(json.dumps(data)); facts['source_sha256'][str(path.relative_to(root))] = digest(path)
        receipt = root/facts['listing']['receipt_path']; meta = json.loads(receipt.read_text())
        meta['raw_sha256'] = digest(path); receipt.write_text(json.dumps(meta))
        facts['source_sha256'][str(receipt.relative_to(root))] = digest(receipt)
    save()
    with pytest.raises(ValueError):load_distributions(root,root/DOCUMENT)


def test_actual_held_quantities_keep_fractional_cash_and_certificates_separate(package):
    root,_,_ = package
    overrides,_ = load_distributions(root,root/DOCUMENT)
    for sid,day,quantity in [('1477','2024-09-12',127),('2834','2026-08-04',13312)]:
        terms = overrides[sid+'-'+day]
        provider = SimpleNamespace(overrides=overrides, on_date=lambda s,d:[dict(
            action_id=s+'-'+d+'-stock',stock_id=s,date=d,kind='stock_dividend',**deepcopy(terms))])
        before = deepcopy(overrides)
        result = FractionalCashActions(provider,SimpleNamespace(holdings={sid:{'qty':quantity}})).on_date(sid,day)
        stock = result[-1]
        assert stock['fractional_cash_per_share'] == 0 and overrides == before
        if sid == '1477':
            assert len(result) == 1 and stock['fractional_settlement']['gross_cash_amount'] == 0
            assert stock['fractional_settlement']['fraction'] == .54
        else:
            assert result[0]['gross_cash_amount'] == 8 and result[0]['pay_date'] is None
            assert stock['certificate_restriction']['certificate_delivery_date'] == '2026-09-02'
            assert stock['pay_date'] == '2026-09-10'


def test_existing_settlement_cannot_be_overwritten(package):
    root,_,_ = package
    provider = SimpleNamespace(corporate_overrides={'1477-2024-09-12':{'pay_date':'2024-09-13'}})
    with pytest.raises(ValueError,match='rewrite'):install_distributions(provider,root/DOCUMENT,root)


def test_real_primary_minutes_and_exchange_notices():
    if not (ROOT/'.cache/all-entry-accounts-20261011/required-listings-v1/outcome.json').exists():
        pytest.skip('Primary evidence is retained in the local research cache')
    overrides,refs = load_distributions(ROOT,ROOT/DOCUMENT)
    assert len(refs) == 11 and overrides['1477-2024-09-12']['pay_date'] == '2024-10-18'
