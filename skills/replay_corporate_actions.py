"""Dated cash/share rights for the integer-share research account."""
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd

from app.finmind import fetch_dataset
from skills.million_replay import UnresolvedAction


START, END = date(2021,1,1), date(2026,12,31)
SPLIT_0050 = {'action_id':'0050-split-20250618','stock_id':'0050','date':'2025-06-18',
    'kind':'split','multiplier':4,
    'source':'https://www.twse.com.tw/zh/ETFortune/announcement?company=A00005&date=20250617&fund=0050&seq=1&type=other'}


def optional_number(value):
    if value is None or pd.isna(value) or value == '':
        return 0.
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError('Corporate amount must be finite and nonnegative')
    return result


def optional_date(value):
    if value is None or pd.isna(value) or value == '':
        return None
    if not isinstance(value,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):
        raise ValueError('Corporate date must be YYYY-MM-DD')
    date.fromisoformat(value)
    return value


class CorporateActions:
    def __init__(self, events, cache_dir, token=None, *, offline=False, overrides=None):
        self.events = events.copy()
        self.events['event_date'] = pd.to_datetime(self.events.event_date).dt.strftime('%Y-%m-%d')
        self.directory = Path(cache_dir)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.token, self.offline = token, offline
        self.overrides = overrides or {}
        self.loaded = {}
        self.requests = 0
        self.sources = {}

    def prepare(self,sid):
        if sid in self.loaded:
            return
        path = self.directory/f'{sid}.parquet'
        if path.exists():
            frame = pd.read_parquet(path)
        elif self.offline:
            raise ValueError('Frozen dividend source missing: '+sid)
        else:
            frame = fetch_dataset('TaiwanStockDividend',START,END,token=self.token,data_id=sid,
                                  timeout=30,max_retries=0)
            self.requests += int(not frame.attrs.get('cache_hit',False))
            frame.to_parquet(path,index=False)
        self.sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = []
        for item in frame.to_dict('records'):
            if str(item.get('stock_id'))!=sid:
                raise ValueError('Wrong stock in dividend feed')
            ex = optional_date(item.get('CashExDividendTradingDate'))
            payment = optional_date(item.get('CashDividendPaymentDate'))
            amount = optional_number(item.get('CashEarningsDistribution'))+optional_number(item.get('CashStatutorySurplus'))
            if ex and amount>0:
                if payment and pd.Timestamp(payment)<pd.Timestamp(ex):
                    raise ValueError('Dividend payment precedes ex date: '+sid)
                row = dict(action_id=f'{sid}-cash-{ex}',stock_id=sid,date=ex,kind='cash_dividend',
                    cash_per_share=amount,pay_date=payment,source='FinMind TaiwanStockDividend',
                    announcement_date=item.get('AnnouncementDate') or None)
                override = self.overrides.get(f'{sid}-{ex}',{})
                if override.get('cash_rounding'):
                    row['cash_rounding'] = override['cash_rounding']
                existing = [r for r in rows if r['action_id']==row['action_id']]
                if existing and any(r['cash_per_share']!=amount or r['pay_date']!=payment for r in existing):
                    raise ValueError('Conflicting dividend revisions: '+row['action_id'])
                if not existing:
                    rows.append(row)
        for event in self.events[self.events.stock_id.eq(sid)].to_dict('records'):
            day = event['event_date']
            payload = json.loads(event['payload_json'])
            key = f'{sid}-{day}'
            match = next((r for r in rows if r['kind']=='cash_dividend' and r['date']==day),None)
            if event['source']=='ex_rights':
                expected_cash = payload.get('cash_dividend')
                if expected_cash is None and event['event_type']=='息':
                    expected_cash = payload.get('value_amount')
                if expected_cash and (match is None or abs(match['cash_per_share']-expected_cash)>.015):
                    rows.append(dict(action_id=key+'-missing-cash',stock_id=sid,date=day,
                        kind='unresolved_cash_dividend',source='official action disagrees with dividend policy'))
                stock_rate = optional_number(payload.get('stock_dividend_per_1000'))/1000
                policy = [r for r in frame.to_dict('records') if r.get('StockExDividendTradingDate')==day]
                policy_stock = any(optional_number(r.get('StockEarningsDistribution'))+optional_number(r.get('StockStatutorySurplus'))>0 for r in policy)
                if stock_rate or policy_stock:
                    override = self.overrides.get(key,{})
                    delivery = optional_date(override.get('pay_date'))
                    if delivery and delivery < day:
                        raise ValueError('Stock delivery precedes ex date')
                    rows.append(dict(action_id=key+'-stock',stock_id=sid,date=day,kind='stock_dividend',
                        shares_per_share=override.get('shares_per_share',stock_rate or None),
                        pay_date=delivery,fractional_cash_per_share=override.get('fractional_cash_per_share'),
                        source=override.get('source','official share entitlement; delivery unresolved')))
                cash_increase = event.get('cash_increase_suspected')
                cash_increase = bool(cash_increase) if pd.notna(cash_increase) else False
                if cash_increase or any(optional_number(p.get('TotalNumberOfCashCapitalIncrease'))>0 for p in policy):
                    rows.append(dict(action_id=key+'-waive',stock_id=sid,date=day,kind='waive_subscription',
                                     source='official and FinMind capital subscription; fixed no-subscription policy'))
            else:
                override = self.overrides.get(key)
                if override:
                    rows.append(dict(action_id=key,stock_id=sid,date=day,**override))
                else:
                    rows.append(dict(action_id=key,stock_id=sid,date=day,kind='unresolved_'+event['source'],
                                     source=event['source']))
        for item in frame.to_dict('records'):
            ex = optional_date(item.get('StockExDividendTradingDate'))
            amount = optional_number(item.get('StockEarningsDistribution'))+optional_number(item.get('StockStatutorySurplus'))
            if ex and amount and not any(row['date']==ex and row['kind']=='stock_dividend' for row in rows):
                rows.append(dict(action_id=f'{sid}-{ex}-missing-stock-source',stock_id=sid,date=ex,
                                 kind='unresolved_stock_dividend',source='FinMind stock distribution lacks exact official terms'))
        if sid=='0050':
            rows.append(dict(SPLIT_0050))
        self.loaded[sid] = sorted(rows,key=lambda r:(r['date'],r['action_id']))

    def on_date(self,sid,day):
        self.prepare(sid)
        return [dict(row) for row in self.loaded[sid] if row['date']==day]

    def reference_price(self,sid,day,prior):
        """Known ex-date opening reference for planning, never a future close."""
        events = self.events[self.events.stock_id.eq(sid)&self.events.event_date.eq(day)]
        if len(events)>1:
            raise UnresolvedAction(f'Multiple corporate opening references: {sid} {day}')
        if len(events):
            row = events.iloc[0]
            ref = row['opening_ref'] if pd.notna(row['opening_ref']) else row['ref_price']
            if ref and math.isfinite(ref) and ref>0:
                return float(ref)
            raise UnresolvedAction(f'Corporate opening reference missing: {sid} {day}')
        return prior/4 if sid=='0050' and day=='2025-06-18' else prior

    def manifest(self):
        for name,digest in self.sources.items():
            if hashlib.sha256((self.directory/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Frozen corporate source changed: '+name)
        return dict(dataset='TaiwanStockDividend',start=START.isoformat(),end=END.isoformat(),
                    files_sha256=self.sources,requests=self.requests,overrides=self.overrides,
                    explicit_split=SPLIT_0050)
