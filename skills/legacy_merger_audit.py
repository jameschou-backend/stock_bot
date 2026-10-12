"""Independent exchange arithmetic and private, guarded journal/slot auditors."""
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
import inspect
from pathlib import Path
from types import FunctionType

import pandas as pd

from skills.all_entry_accounts import digest
from skills.legacy_merger_settlement import (ROOT, DOCUMENT, OLD, NEW, EXCHANGE, ORDINARY,
    REFERENCE, ACTION, NOTE, load_terms, MergerSettlement)
from skills.top3_cost_audit import private_cash_audit


def exchange_actions(account):
    return [r for r in account['corporate_actions'] if r['kind']=='share_exchange']


def audit_exchange(account, quotes):
    """Recompute old units, new units, noncash fractions and restricted inventory."""
    actions = exchange_actions(account)
    if not actions: return dict(events=[],certificate_trades=0)
    if len(actions) != 1: raise ValueError('Unexpected additional share exchange')
    action = actions[0]
    source = quotes.copy(); source['date'] = pd.to_datetime(source.date).dt.strftime('%Y-%m-%d')
    price_rows = source.loc[source.stock_id.eq(NEW) & source.date.eq(REFERENCE)]
    if len(price_rows) != 1: raise ValueError('Fraction reference is not a unique observed close')
    price = Decimal(str(price_rows.iloc[0]['close']))
    eid = action['event_id']
    old = sum(t['qty']*(1 if t['side']=='buy' else -1) for t in account['trades']
        if t['event_id']==eid and t['date']<EXCHANGE)
    if old <= 0 or any(r['event_id']==eid and r['date']<EXCHANGE
        and r['kind'] in ('split','stock_dividend','capital_reduction','share_delivery')
        for r in account['corporate_actions']):
        raise ValueError('Exchange source units require unambiguous reconstruction')
    exact = Decimal(old)*Decimal('0.2596'); whole = int(exact); fraction = exact-whole
    gross = int((fraction*price).quantize(Decimal(1),rounding=ROUND_HALF_UP))
    fields = dict(action_id=ACTION,date=EXCHANGE,stock_id=OLD,target_stock_id=NEW,
        entitled_qty=old,qty_after=whole,ratio=.2596,fractional_right=float(fraction),
        fractional_gross_amount=gross,fractional_reference_price=float(price),
        fractional_reference_date=REFERENCE,fractional_rounding='half_up_ntd',
        fractional_cash_available=False,certificate_trading_modeled=False,
        certificate_delivery_date=EXCHANGE,ordinary_share_available_date=ORDINARY,
        valuation_basis='ordinary_close_proxy')
    if any(action.get(k)!=v for k,v in fields.items()): raise ValueError('Independent exchange entitlement differs')
    cohort = next(c for c in account['cohorts'] if c['event_id']==eid)
    if (cohort['stock_id'] != OLD or cohort.get('current_stock_id') != NEW
            or cohort.get('share_exchange_date') != EXCHANGE):
        raise ValueError('Exchange changed candidate identity or lost target ownership')
    claims = [r for r in account['receivables'] if r['action_id']==ACTION+'-fractional-cash']
    if ((gross and len(claims)!=1) or (not gross and claims)
            or any(r['amount']!=gross or r['pay_date'] is not None or r['stock_id']!=NEW for r in claims)
            or any(r.get('action_id')==ACTION+'-fractional-cash' for r in account['cash_ledger'])):
        raise ValueError('Unknown net fraction cash was paid or omitted')
    if any(t['stock_id']==OLD and t['date']>='2026-08-20' for t in account['trades']):
        raise ValueError('Source shares traded during suspension or after delisting')
    journal=[]
    for day in [r['date'] for r in account['daily'] if r['date']>=EXCHANGE]:
        rows=[h for h in account['holdings'] if h['event_id']==eid and h['date']==day]
        trades=[t for t in account['trades'] if t['event_id']==eid and t['date']==day]
        if day<ORDINARY and trades: raise ValueError('Merger certificate was traded')
        whole += sum(t['qty']*(1 if t['side']=='buy' else -1) for t in trades)
        instrument='nontraded_merger_certificate' if day<ORDINARY else 'ordinary_share'
        if sum(h['qty'] for h in rows)!=whole or any(h['stock_id']!=NEW or h.get('instrument_kind')!=instrument for h in rows):
            raise ValueError('Exchange inventory differs or certificate became early ordinary units')
        journal.append(dict(date=day,units=whole,instrument_kind=instrument))
    return dict(events=[dict(event_id=eid,old_units=old,new_units=int(exact),fractional_gross=gross,
        fractional_cash_available=False,certificate_trading_modeled=False,daily=journal)],certificate_trades=0)


def _unit_exchange(action, units):
    if (action['stock_id'],action['target_stock_id'],action['date']) != (OLD,NEW,EXCHANGE):
        raise ValueError('Unexpected journal exchange identity')
    whole = int(Decimal(units[OLD])*Decimal('.2596'))
    if units[NEW] or whole != action['qty_after']:
        raise ValueError('Exchange overlaps target ownership or has wrong units')
    units[OLD],units[NEW] = 0,whole


def corporate_auditor(mode):
    from skills.corporate_account_audit import audit_corporate_account
    original=private_cash_audit(audit_corporate_account,mode)
    code=inspect.getsource(audit_corporate_account)
    old="            if kind == 'capital_reduction':"
    if code.count(old)!=1: raise ValueError('Corporate audit exchange seam differs')
    code=code.replace(old,"            if kind == 'share_exchange':\n                _unit_exchange(action, units)\n            elif kind == 'capital_reduction':")
    code=code.replace("raw_gross*Decimal('.001425')","raw_gross*Decimal(str(_top3_commission))")
    ns=dict(original.__globals__,_unit_exchange=_unit_exchange)
    exec(compile(code,str(ROOT/'skills/corporate_account_audit.py'),'exec'),ns)
    return ns['audit_corporate_account']


def _opening_sid(cohort,day):
    return (cohort['current_stock_id'] if cohort.get('share_exchange_date','9999')<day else cohort['stock_id'])


def _exchange_opening(account,day,opening,active,wanted):
    for action in exchange_actions(account):
        if action['date'] != day: continue
        old,new=action['stock_id'],action['target_stock_id']
        if old not in opening or new in opening or old in wanted:
            raise ValueError('Exchange slot needs one unreleased source position')
        holding=opening.pop(old)
        if holding['event_id']!=action['event_id'] or holding['qty']!=action['entitled_qty']:
            raise ValueError('Exchange opening slot disagrees with units')
        opening[new]=dict(holding,qty=action['qty_after'])
        if old in active: active.remove(old);active.add(new)


def resource_auditor(mode):
    from skills.high_return_audit import audit_high_return_resources
    from skills.execution_resources import audit_resources
    corporate=corporate_auditor(mode)
    if digest(ROOT/'skills/high_return_audit.py') != '00630cab1fb8c65b83fcbdc3241968711be4e6f5e3c4702d0de32916d1a7ef36':
        raise ValueError('Frozen resource source changed')
    code=inspect.getsource(audit_high_return_resources)
    old="open_cohorts = {c['stock_id']: c for c in account['cohorts']"
    site="        day_decisions = [r for r in decisions if r['date']==day]"
    if code.count(old)!=1 or code.count(site)!=1: raise ValueError('Resource exchange seams differ')
    code=code.replace(old,"open_cohorts = {_opening_sid(c,day): c for c in account['cohorts']")
    code=code.replace(site,"        _exchange_opening(account,day,opening,active,wanted)\n"+site)
    def audit(account,plans,decisions,board,snapshots,quotes):
        proof=audit_exchange(account,quotes)
        checked=corporate(account)
        view=dict(account,corporate_actions=deepcopy(account['corporate_actions']))
        for row in view['corporate_actions']:
            if row['kind']=='capital_reduction':row['kind']='split'
        def validated(value):
            if value is not view: raise ValueError('Unexpected corporate resource view')
            return dict(checked)
        resources=FunctionType(audit_resources.__code__,dict(audit_resources.__globals__,audit_stress=validated),
            audit_resources.__name__,audit_resources.__defaults__,audit_resources.__closure__)
        resources.__kwdefaults__=audit_resources.__kwdefaults__
        ns=dict(audit_high_return_resources.__globals__,audit_resources=resources,
            _opening_sid=_opening_sid,_exchange_opening=_exchange_opening)
        exec(compile(code,str(ROOT/'skills/high_return_audit.py'),'exec'),ns)
        result=ns['audit_high_return_resources'](view,plans,decisions,board,snapshots,quotes)
        result.update(capital_resources_independently_reconciled=True,share_exchange=proof)
        return result
    return audit


def install(runner,mode):
    document,refs=load_terms()
    read=runner.read
    target=ROOT/'docs/stock_universe_2019_corporate_terms.json'
    def augmented(path):
        value=read(path)
        if Path(path).resolve()!=target:return value
        value=deepcopy(value)
        if any(h['stock_id']==OLD and h['start']<EXCHANGE and '2026-08-20'<h['end'] for h in value['verified_halts']):
            raise ValueError('Merger halt overlaps sealed terms')
        evidence=document['evidence']['issuer0']
        value['verified_halts'].append(dict(stock_id=OLD,market='TWSE',kind='trading_suspension',
            start='2026-08-20',end=EXCHANGE,announcement_date='2026-08-14',
            source_path=evidence['path'],source_sha256=digest(ROOT/evidence['path']),
            source='ESUN issuer primary shareholder QA'))
        value['source_sha256'].update(refs)
        return value
    runner.read=augmented
    engines=runner.engine_types
    def types(*args,**kwargs):
        return tuple(type('Merger'+cls.__name__,(MergerSettlement,cls),{}) for cls in engines(*args,**kwargs))
    runner.engine_types=types
    runner.audit_face_resources=resource_auditor(mode)
    writer=runner.write
    def write(path,value):
        if value.get('completed') and 'account' in value and exchange_actions(value['account']):
            value=dict(value,merger_settlement_note=NOTE,
                settlement_evidence_notes=[*value.get('settlement_evidence_notes',[]),NOTE])
        return writer(path,value)
    runner.write=write
    return refs
