"""Reject an unheld 2888 buy during the preannounced merger suspension."""
from copy import deepcopy
import json
from pathlib import Path

from scripts.summarize_poc_broker_account import Sources
from skills.all_entry_accounts import digest
from skills.all_entry_bank_distributions import normalized

ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = 'docs/all_entry_2888_halt_20261012.json'
START, END = '2025-07-14', '2025-07-24'
POLICY = 'preannounced_2888_unheld_buy_rejection_v1'
URL = 'https://www.twse.com.tw/staticFiles/news/news/tsecnews/8a8216d697769dcf0197b0ab734100e2.pdf'
REASON = 'preannounced_merger_suspension_no_buy'


def load_halt(root, document):
    root,document = Path(root).resolve(),Path(document).resolve(); sources = Sources(root)
    doc=json.loads(sources.bind(str(document.relative_to(root)),digest(document)).read_text())
    if (doc.get('schema') != POLICY or doc.get('strategy_parameters_changed') is not False
            or doc.get('live_qualified') is not False or doc.get('reviewed_pdf_pages') != [1]):
        raise ValueError('Merger halt evidence scope differs')
    for name,sha in doc['source_sha256'].items():sources.bind(name,sha)
    notice,receipt,text=(root/doc[k] for k in ('notice_path','receipt_path','text_path'))
    if any(doc[k] not in sources.used for k in ('notice_path','receipt_path','text_path')):
        raise ValueError('Merger halt evidence is unbound')
    meta=json.loads(receipt.read_text())
    if (meta.get('url') != URL or meta.get('http_status') != 200 or meta.get('accepted') is not True
            or meta.get('security_denied') is not False or meta.get('automatic_redirects_disabled') is not True
            or meta.get('redirect_statuses') != [] or meta.get('raw_sha256') != digest(notice)
            or not notice.read_bytes().startswith(b'%PDF')):
        raise ValueError('Merger halt primary receipt differs')
    required=['中華民國114年6月27日','新光金融控股股份有限公司','2888',
        '自114年7月14日起停止買賣','114年7月24日終止上市']
    if any(normalized(x) not in normalized(text.read_text()) for x in required):
        raise ValueError('Merger halt dates are not supported')
    return dict(stock_id='2888',market='TWSE',kind='trading_suspension',start=START,end=END,
        announcement_date='2025-06-27',source_path=doc['notice_path'],source_sha256=digest(notice),
        source='TWSE primary merger suspension announcement'),dict(sources.used)


class MergerHaltBuy:
    def _plan(self,day,sid,side,eid,signal,qty,budget,opening_cash,failure):
        stamp=str(day.date())
        if sid == '2888' and START <= stamp < END and self.official_halt(day,sid):
            if side != 'buy' or self.holdings.get(sid,{}).get('qty',0):
                raise ValueError('Held 2888 merger rights require a separate conversion model')
            # The outer planner still locks the original attempted budget/slot.
            # Zero order quantities need no fabricated suspension-day limits.
            qty,failure=0,REASON
        return super()._plan(day,sid,side,eid,signal,qty,budget,opening_cash,failure)


def audit_halt(case):
    if not case.get('completed'):raise ValueError('Partial account has no merger halt proof')
    account=case['account']
    if any(r['stock_id']=='2888' and r['date']>=START and r['qty']>0
           for r in [*account['holdings'],*account['trades']]):
        raise ValueError('Unmodeled merger ownership cannot be published')
    plans=[p for p in account['tick_plans'] if p['stock_id']=='2888' and START<=p['date']<END]
    for p in plans:
        if (p['side']!='buy' or p['rejection']!=REASON
                or any(p[k]!=0 for k in ('planned_qty','board_qty','odd_qty'))
                or p['signal_date']!=p['reference_date'] or not '2025-06-27'<p['signal_date']<p['date']):
            raise ValueError('Suspension plan was not rejected before execution')
        orders=[r for r in account['orders'] if (r['date'],r['event_id'],r['side'])==(p['date'],p['event_id'],p['side'])]
        if (len(orders)>1 or any(r['filled_qty']!=0 or r['channel']!='event' for r in orders)
                or any(r.get('date')==p['date'] and r.get('event_id')==p['event_id']
                       and r.get('cash_change',0)!=0 for r in account['cash_ledger'])):
            raise ValueError('Suspension rejection moved shares or cash')
    return dict(schema=POLICY,plans_checked=len(plans),held_merger_rights=False,
        suspension_trades=0,legal_limits_fabricated=False)


def verify_halt_proof(case,root=ROOT):
    account=case['account']
    affected=any(r['stock_id']=='2888' and r['date']>=START and r['qty']>0
        for r in [*account.get('holdings',[]),*account.get('trades',[])])
    affected=affected or any(p['stock_id']=='2888' and START<=p['date']<END
        for p in account.get('tick_plans',[]))
    if not affected and 'merger_halt_audit' not in case:return
    load_halt(root,Path(root)/DOCUMENT)
    if case.get('merger_halt_audit')!=audit_halt(case):
        raise ValueError('Independent merger suspension proof is absent or differs')


def install_halt(runner,document,root):
    halt,refs=load_halt(root,document);read=runner.read
    target=Path(root).resolve()/'docs/stock_universe_2019_corporate_terms.json'
    def augmented_read(path):
        value=read(path)
        if Path(path).resolve()!=target:return value
        value=deepcopy(value)
        if any(h['stock_id']=='2888' and h['start']<END and START<h['end'] for h in value['verified_halts']):
            raise ValueError('Merger suspension overlaps prior terms')
        value['verified_halts'].append(deepcopy(halt))
        for name,sha in refs.items():
            if name in value['source_sha256'] and value['source_sha256'][name]!=sha:
                raise ValueError('Merger halt source conflicts')
        value['source_sha256'].update(refs)
        return value
    runner.read=augmented_read
    original=runner.engine_types
    def engine_types(*args,**kwargs):
        return tuple(type('MergerHalt'+cls.__name__,(MergerHaltBuy,cls),{}) for cls in original(*args,**kwargs))
    runner.engine_types=engine_types
    writer=runner.write
    def write(path,value):
        if isinstance(value,dict) and value.get('completed') is True and 'account' in value:
            value=dict(value,merger_halt_audit=audit_halt(value))
        return writer(path,value)
    runner.write=write
    return refs
