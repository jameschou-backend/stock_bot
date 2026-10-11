import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from skills.all_entry_accounts import digest
from skills.all_entry_bank_distributions import EXPECTED, load_distributions, install_distributions
from skills.scenario_exit_replay import FractionalCashActions

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = 'docs/all_entry_bank_distributions_20261011.json'


@pytest.fixture
def package(tmp_path):
    doc = json.loads((ROOT/DOCUMENT).read_text())
    for key, event in doc['events'].items():
        spec = EXPECTED[key]; sid = key[:4]
        for name in event['source_sha256']: (tmp_path/name).parent.mkdir(parents=True,exist_ok=True)
        for kind, host in [('issuer',spec['issuer']),('calendar',spec['calendar']),('exchange','www.twse.com.tw')]:
            item=event[kind]; path=tmp_path/item['path']
            if kind=='issuer':
                path.write_bytes(b'%PDF synthetic unit-test fixture')
                (tmp_path/item['text_path']).write_text('\n'.join(spec['required']))
            elif kind=='calendar':
                path.write_text('<table><tr>'+''.join('<td>'+v+'</td>' for v in spec['calendar_row'])+'</tr></table>')
            else:
                roc=lambda d:str(int(d[:4])-1911)+'年'+d[5:7]+'月'+d[8:]+'日'
                path.write_text(json.dumps(dict(stat='OK',data=[[sid,'fixture','(股票上市)增資新股上市',1,'fixture',
                    roc(spec['announcement']),roc(spec['pay']),str(spec['quantity']),'1',
                    f"與原已發行之股份相同 含員工酬勞轉增資 {spec['employee']:,} 股"]])))
            url='https://'+host+('/rwd/zh/announcement/LT185' if kind=='exchange' else '/fixture')
            (tmp_path/item['receipt_path']).write_text(json.dumps(dict(url=url,final_url=url,http_status=200,
                automatic_redirects_disabled=True,accepted=True,security_denied=False,redirect_statuses=[],
                params={'stockNo':sid},sha256=digest(path))))
        feed=tmp_path/event['existing_dividend_feed']
        pd.DataFrame([dict(stock_id=sid,StockExDividendTradingDate=spec['raw_ex'],
            StockEarningsDistribution=float(spec['rate'])*10,StockStatutorySurplus=0,
            ParticipateDistributionOfTotalShares=spec['total'])]).to_parquet(feed)
        feed.with_suffix('.json').write_text('{}')
        event['source_sha256']={name:digest(tmp_path/name) for name in event['source_sha256']}
    path=tmp_path/DOCUMENT;path.parent.mkdir(exist_ok=True)
    def save():path.write_text(json.dumps(doc))
    save();return tmp_path,path,doc,save


def test_final_ratio_and_restricted_fractional_cash(package):
    root,path,_,_=package;rows,refs=load_distributions(root,path);assert len(refs)==19
    row=rows['2884-2024-07-26'];assert row['shares_per_share']==.020033197
    provider=SimpleNamespace(overrides=rows,on_date=lambda s,d:[dict(action_id='test',kind='stock_dividend',stock_id=s,date=d,**row)])
    events=FractionalCashActions(provider,SimpleNamespace(holdings={'2884':{'qty':1000}})).on_date('2884','2024-07-26')
    assert events[-1]['pay_date']=='2024-08-29'
    assert events[-1]['fractional_settlement']['fraction']==.033197
    assert events[-1]['fractional_cash_per_share']==0
    assert not any(e.get('pay_date')=='2024-07-26' for e in events)


@pytest.mark.parametrize('field,value',[('shares_per_share',.02),('pay_date','2024-08-02'),
    ('ordinary_share_available_date','2024-08-02'),('fractional_cash_available_for_trading',True),
    ('fractional_cash_payment_date_verified',True),('fractional_cash_net_amount_verified',True),
    ('fractional_cash_pay_date','2024-08-29'),('certificate_delivery_date','2024-08-29')])
def test_no_rounded_ratio_record_date_or_unverified_cash(package,field,value):
    root,path,doc,save=package;doc['events']['2884-2024-07-26']['override'][field]=value;save()
    with pytest.raises(ValueError):load_distributions(root,path)


@pytest.mark.parametrize('change',['bytes','employee','certificate','calendar','ex_date'])
def test_primary_facts_and_feed_rechecked(package,change):
    root,path,doc,save=package;event=doc['events']['2884-2024-07-26']
    if change=='bytes':(root/event['issuer']['path']).write_bytes(b'%PDF changed')
    elif change=='ex_date':
        f=root/event['existing_dividend_feed'];data=pd.read_parquet(f);data['StockExDividendTradingDate']='2024-08-02';data.to_parquet(f)
        event['source_sha256'][str(f.relative_to(root))]=digest(f)
    else:
        item=event['calendar' if change=='calendar' else 'exchange'];f=root/item['path']
        if change=='calendar':f.write_text(f.read_text().replace('0.20033197','0.2'))
        else:
            data=json.loads(f.read_text())
            if change=='employee':data['data'][0][9]=data['data'][0][9].replace('18,000,000','0')
            else:data['data'][0][2]='新股權利證書上市'
            f.write_text(json.dumps(data))
        event['source_sha256'][item['path']]=digest(f)
        r=root/item['receipt_path'];data=json.loads(r.read_text());data['sha256']=digest(f);r.write_text(json.dumps(data))
        event['source_sha256'][item['receipt_path']]=digest(r)
    save()
    with pytest.raises(ValueError):load_distributions(root,path)


def test_conflicting_existing_event_rejected(package):
    root,path,_,_=package
    with pytest.raises(ValueError,match='rewrite'):
        install_distributions(SimpleNamespace(corporate_overrides={'2884-2024-07-26':{}}),path,root)


def test_archived_primary_package():
    doc=json.loads((ROOT/DOCUMENT).read_text())
    if not (ROOT/doc['events']['2884-2024-07-26']['issuer']['path']).exists():pytest.skip('Primary archive is local')
    rows,refs=load_distributions(ROOT,ROOT/DOCUMENT);assert len(rows)==2 and len(refs)==19
