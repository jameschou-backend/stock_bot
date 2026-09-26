"""Compare issuer-visible distribution dates to unchanged benchmark cash journals."""
from datetime import date
from decimal import Decimal
from pathlib import Path
from app.backtest_tool_ui import verified_bytes
from app.index_continuous_ui import load as continuous
from scripts.research_exit_scenarios import read,sha

ROOT=Path(__file__).resolve().parents[1]
URL='https://www.yuantafunds.com/myfund/information/1066?tab=3'
CAPTURE=Path('.cache/issuer-dividends-20260927/visible-tables-v1.json')


def issuer_rows(value):
    if (value.get('schema')!='issuer_visible_dividend_tables_v1' or value.get('source_url')!=URL
            or value.get('security_id')!='0050' or value.get('fund_heading')!='0050 元大台灣卓越50基金'
            or value.get('columns',[])[:4]!=['配息年月','除息日','發放日','每單位配息金額']):
        raise ValueError('Wrong issuer instrument or table columns')
    if [p['page'] for p in value['pages']]!=[1,2,3] or any(len(p['rows'])!=10 for p in value['pages']):
        raise ValueError('Incomplete visible issuer pages')
    result={};ordered=[]
    for page in value['pages']:
        for row in page['rows']:
            if len(row)!=6:raise ValueError('Incomplete distribution row')
            month,ex,payment,amount=row[:4]
            ex=date.fromisoformat(ex.replace('/','-')).isoformat()
            payment=date.fromisoformat(payment.replace('/','-')).isoformat();amount=Decimal(amount)
            if (month.replace('/','-')!=ex[:7] or payment<=ex or not amount.is_finite() or amount<=0 or ex in result):
                raise ValueError('Invalid or duplicate distribution')
            result[ex]=dict(ex_date=ex,pay_date=payment,cash_per_share=float(amount),page=page['page'])
            ordered.append(ex)
    if ordered!=sorted(ordered,reverse=True):raise ValueError('Issuer pagination overlaps or is unordered')
    return result


def audit_account(account,rows,start,end):
    expected={k:v for k,v in rows.items() if start<=k<=end}
    actions=[a for a in account['corporate_actions'] if a['kind']=='cash_dividend']
    payments=[a for a in account['corporate_actions'] if a['kind']=='payment']
    ledger=[r for r in account['cash_ledger'] if r['kind']=='dividend_payment']
    if (len(actions)!=len(expected) or {a['date'] for a in actions}!=set(expected)
            or len(payments)!=len(expected) or len(ledger)!=len(expected)):
        raise ValueError('Missing or duplicated dividend entitlement/payment')
    seen=set();matched=[]
    for action in actions:
        ex=action['date'];official=expected[ex];key=action['action_id']
        if key in seen or action['stock_id']!='0050':raise ValueError('Duplicate action or wrong security')
        seen.add(key)
        if (action['pay_date']!=official['pay_date']
                or Decimal(str(action['cash_per_share']))!=Decimal(str(official['cash_per_share']))):
            raise ValueError('Issuer dividend date or amount differs: '+ex)
        cash=Decimal(str(action['entitled_qty']))*Decimal(str(official['cash_per_share']))
        if Decimal(str(action['entitlement_value']))!=cash:raise ValueError('Dividend entitlement amount differs')
        paid=[p for p in payments if p['action_id']==key];posted=[p for p in ledger if p['action_id']==key]
        if len(paid)!=1 or len(posted)!=1:raise ValueError('Missing or duplicate payment receipt')
        payment=paid[0];entry=posted[0]
        if (payment['stock_id']!='0050' or entry['stock_id']!='0050' or payment['ex_date']!=ex
                or payment['pay_date']!=official['pay_date'] or payment['date']!=official['pay_date']
                or entry['date']!=official['pay_date'] or Decimal(str(payment['amount']))!=cash
                or Decimal(str(entry['cash_change']))!=cash):raise ValueError('Dividend cash posted on wrong date or amount')
        matched.append(dict(official,action_id=key,entitled_qty=action['entitled_qty'],cash_received=float(cash)))
    return dict(matched_dividends=len(matched),payments_on_issuer_dates=True,rows=sorted(matched,key=lambda r:r['ex_date']))


def reconcile(root=ROOT):
    root=Path(root);capture=read(root/CAPTURE);rows=issuer_rows(capture);pub=continuous(root)
    accounts={**{'continuous_'+k:v for k,v in pub['benchmarks'].items()},**pub['reference_controls']}
    checks={}
    for key,descriptor in accounts.items():
        import json
        value=json.loads(verified_bytes(descriptor['result'],root,'.json'))
        account=value['account'];start=account['daily'][0]['date'];end=account['daily'][-1]['date']
        check=audit_account(account,rows,start,end)
        checks[key]=dict(check,result=descriptor['result'],start=start,end=end,summary=value['summary'])
    return dict(schema='index_dividend_reconciliation_v1',passed=True,security_id='0050',
        source_url=URL,observed_at_utc=capture['observed_at_utc'],
        capture=dict(path=str(CAPTURE),sha256=sha(root/CAPTURE)),
        benchmark_payment_dates_issuer_matched=True,unique_period_dividends=len([d for d in rows if pub['start']<=d<=pub['end']]),
        account_checks=checks,changed_accounts=0,new_backtests=0,finmind_requests=0,
        strict_data_ready=False,live_qualified=False,unseen_validation=False,
        limitations=['Issuer history observed on 2026-09-27; this does not prove original announcement timestamps.',
            'This is an additive reconciliation; sealed accounts and their original limitations are preserved.',
            'Derived daily price limits and daily-bar execution limitations remain unresolved.'])
