"""One evidenced cross-security exchange; preserve the original investment clock.

The 2867 certificate is held as restricted 2884 units from September 1, marked
with an explicitly labeled ordinary-close proxy. It cannot trade until the
September 8 ordinary conversion. Fractional cash is never spendable without
the shareholder's remittance details. No candidate is removed for this event.
"""
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path

import pandas as pd

from scripts.summarize_poc_broker_account import Sources
from skills.all_entry_accounts import digest
from skills.all_entry_bank_distributions import normalized
from skills.scenario_exit_replay import ExitSignals

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = ROOT/'docs/legacy_2867_exchange_20261012.json'
OLD, NEW = '2867', '2884'
EXCHANGE, ORDINARY, REFERENCE = '2026-09-01', '2026-09-08', '2026-08-19'
RATIO = Decimal('.2596')
ACTION = '2867-2026-09-01-share-exchange'
NOTE = ('三商壽2867按0.2596換玉山金2884；9/1～9/7為不可交易證書庫存，'
        '按普通股收盤代理估值，9/8才可交易普通股。畸零款保留毛額應收，未假設匯費豁免或入帳。')


def load_terms():
    sources = Sources(ROOT)
    document = json.loads(sources.bind(str(DOCUMENT.relative_to(ROOT)), digest(DOCUMENT)).read_text())
    wanted = dict(schema='legacy_2867_exchange_v1', source_stock_id=OLD, target_stock_id=NEW,
        ratio='.2596', halt_start='2026-08-20', exchange_date=EXCHANGE, ordinary_date=ORDINARY,
        fraction_reference_date=REFERENCE, fraction_rounding='half_up_ntd',
        fraction_payment_date_verified=True, fraction_net_amount_verified=False,
        fraction_cash_available_for_trading=False, certificate_trading_modeled=False,
        certificate_valuation='ordinary_close_proxy', selection_parameters_changed=False, live_qualified=False)
    if any(document.get(k) != v for k,v in wanted.items()):
        raise ValueError('2867 settlement terms changed')
    for name,sha in document['source_sha256'].items(): sources.bind(name,sha)
    urls = {
        'issuer0':'https://www.esunbank.com/zh-tw/-/media/ESUNFHC/Files/Investor-Relations/mna/mna09QA.pdf',
        'issuer1':'https://www.esunbank.com/zh-tw/-/media/ESUNFHC/Files/Investor-Relations/mna/mna_20260713.pdf',
        'exchange0':'https://www.twse.com.tw/rwd/zh/announcement/LT185'}
    for key,item in document['evidence'].items():
        if key not in urls or any(item[k] not in sources.used for k in ('path','receipt_path')):
            raise ValueError('Unbound merger evidence')
        receipt = json.loads((ROOT/item['receipt_path']).read_text())
        if (receipt.get('url') != urls[key] or receipt.get('http_status') != 200
                or receipt.get('accepted') is not True or receipt.get('security_denied') is not False
                or receipt.get('automatic_redirects_disabled') is not True or receipt.get('redirect_statuses') != []
                or receipt.get('raw_sha256') != digest(ROOT/item['path'])):
            raise ValueError('Merger evidence receipt differs')
    if set(document['evidence']) != set(urls) or document['reviewed_pdf_pages'] != {'qa':[1],'notice':[1]}:
        raise ValueError('Merger primary evidence is incomplete')
    folder = ROOT/'.cache/legacy-return-comparison-20261012/merger-pdfs-v1'
    qa, notice = (normalized((folder/(name+'.txt')).read_text()) for name in ('qa','notice'))
    for fact in ('115年8月19日','115年8月20日-115年9月1日','0.2596','四捨五入','匯費10元'):
        if normalized(fact) not in qa: raise ValueError('Issuer QA fact missing: '+fact)
    if any(normalized(fact) not in notice for fact in ('115/07/13','0.2596','115年9月1日')):
        raise ValueError('Issuer final merger ratio/date missing')
    listing = json.loads((ROOT/document['evidence']['exchange0']['path']).read_text())
    rows = listing['data']
    if (listing['stat'] != 'OK' or len(rows) != 2 or any(r[0] != NEW for r in rows)
            or rows[0][2] != '新股權利證書上市'
            or rows[0][5:8] != ['115年08月27日','115年09月01日','1,531,510,471']
            or rows[1][2] != '(股票上市)新股權利證書終止上市，增資股股票上市'
            or rows[1][5:8] != ['115年09月03日','115年09月08日','1,531,510,471']):
        raise ValueError('Merger certificate and ordinary listing dates differ')
    return document, dict(sources.used)


def entitlement(quantity, reference_price):
    if type(quantity) is not int or quantity <= 0 or not reference_price or reference_price <= 0:
        raise ValueError('Exchange requires positive integer shares and the dated fraction reference')
    exact = quantity*RATIO
    whole = int(exact)
    gross = int(((exact-whole)*Decimal(str(reference_price))).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    return whole, float(exact-whole), gross


class MergerActions:
    def __init__(self, provider, account):
        self.provider, self.account = provider, account

    def __getattr__(self, name): return getattr(self.provider, name)

    def on_date(self, sid, day):
        if sid == OLD and day == EXCHANGE:
            self.account.exchange_units()
            return []
        return self.provider.on_date(sid, day)


class MergerSettlement:
    def __init__(self, *args, **kwargs):
        self.merger_event_id = None
        super().__init__(*args, **kwargs)
        if self.benchmark: return
        self.corporate = MergerActions(self.corporate, self)
        # Only liquidation signals need a continuous economic unit. Entry
        # candidates and all executable prices retain their sealed raw inputs.
        adjusted = self.exit_signals.adjusted_close.copy()
        if OLD not in adjusted or NEW not in adjusted:
            raise ValueError('Merger requires both source and target price histories')
        ref = pd.Timestamp(REFERENCE)
        scale = adjusted.at[ref,OLD]/self.raw(ref,OLD)
        if not pd.notna(scale) or scale <= 0: raise ValueError('Missing pre-exchange adjusted/raw anchor')
        future = adjusted.index >= EXCHANGE
        adjusted.loc[future,OLD] = self.fields['close'].loc[future,NEW]*float(RATIO)*scale
        self.exit_signals = ExitSignals(adjusted, self.days)
        self.merger_signal_scale = float(scale)

    @property
    def opening_members(self): return self._merger_opening_members

    @opening_members.setter
    def opening_members(self, value):
        value = set(value)
        if self.merger_event_id and OLD in value:
            value.remove(OLD); value.add(NEW)
        self._merger_opening_members = value

    def identity(self, day, sid):
        if sid == OLD and str(day.date()) == EXCHANGE and self.holdings.get(OLD,{}).get('qty',0):
            # Dated legal ownership is settled by on_date below, before orders.
            # This exception grants neither a quotation nor trade permission.
            return dict(status='official_trading_suspension',market='TWSE',category='股票')
        return super().identity(day,sid)

    def exchange_units(self):
        if self.merger_event_id or NEW in self.holdings:
            raise ValueError('Repeated or overlapping target merger holdings require separate review')
        holding = self.holdings[OLD]
        if any(r.get('qty',0) for r in self.receivables if r['stock_id'] == OLD):
            raise ValueError('Pending source stock rights require separate exchange review')
        if self.drawdown_arm != 'control':
            raise ValueError('Cross-security three-black signals need separate validation')
        self.corporate.prepare(NEW)
        if self.corporate.provider.on_date(NEW,EXCHANGE):
            raise ValueError('Simultaneous target corporate actions require separate exchange review')
        price = self.raw(pd.Timestamp(REFERENCE),NEW)
        qty, fraction, gross = entitlement(holding['qty'],price)
        if not qty: raise ValueError('All-fraction merger requires separate slot closure review')
        cohort = next(c for c in self.cohorts if c['event_id'] == holding['event_id'])
        self.merger_event_id = holding['event_id']
        self.actions.append(dict(action_id=ACTION,date=EXCHANGE,kind='share_exchange',stock_id=OLD,
            target_stock_id=NEW,event_id=holding['event_id'],entitled_qty=holding['qty'],qty_after=qty,
            ratio=float(RATIO),fractional_right=fraction,fractional_gross_amount=gross,
            fractional_reference_date=REFERENCE,fractional_reference_price=price,
            fractional_rounding='half_up_ntd',fractional_cash_available=False,
            certificate_delivery_date=EXCHANGE,ordinary_share_available_date=ORDINARY,
            certificate_trading_modeled=False,valuation_basis='ordinary_close_proxy',
            original_due_index=holding['due_index'],signal_scale=self.merger_signal_scale))
        if gross:
            self.receivables.append(dict(stock_id=NEW,kind='cash',amount=float(gross),pay_date=None,
                action_id=ACTION+'-fractional-cash',event_id=holding['event_id'],ex_date=EXCHANGE,
                cash_flow_nature='fractional_exchange_gross',net_amount_verified=False))
        self.holdings[NEW] = dict(holding,qty=qty)
        del self.holdings[OLD]
        prior = self.days[self.positions[pd.Timestamp(EXCHANGE)]-1]
        price = self.raw(prior,NEW)
        if not price: raise ValueError('Acquirer prior valuation is missing')
        self.marks[NEW] = dict(price=price,date=str(prior.date()))
        cohort.update(current_stock_id=NEW,share_exchange_date=EXCHANGE)

    def _plan(self,day,sid,side,eid,signal,qty,budget,opening_cash,failure):
        if self.merger_event_id == eid and sid == NEW and str(day.date()) < ORDINARY:
            qty,failure = 0,'merger_certificate_not_executable'
        return super()._plan(day,sid,side,eid,signal,qty,budget,opening_cash,failure)

    def run(self):
        account = super().run()
        for row in account['holdings']:
            if row['event_id'] == self.merger_event_id and row['date'] >= EXCHANGE:
                row['instrument_kind'] = 'nontraded_merger_certificate' if row['date'] < ORDINARY else 'ordinary_share'
        return account
