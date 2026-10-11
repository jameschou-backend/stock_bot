"""The newly encountered 8,346-share holding keeps its actual delivery clock."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from skills.all_entry_accounts import digest
from skills.all_entry_corporate import load_terms
from skills.million_replay import Replay
from skills.scenario_exit_replay import FractionalCashActions


def test_5871_delivery_preserves_166_shares_and_unspendable_fraction(tmp_path):
    document = json.loads((Path(__file__).resolve().parents[1] /
        'docs/all_entry_corporate_terms_v2_20261011.json').read_text())
    # Synthetic test notices isolate the loader test from local evidence caches.
    for key, evidence in document['evidence'].items():
        for notice in evidence['sources']:
            path = tmp_path/notice['path'];path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('Synthetic test notice: '+' '.join(notice['required_text']))
            notice['sha256'] = digest(path)
            day = pd.Timestamp(notice['announcement_date'])
            receipt = dict(url=notice['url'],http_status=200,redirected=False,sha256=digest(path),
                announcement_row=[key[:4],'synthetic',f'{day.year-1911:03d}/{day.month:02d}/{day.day:02d}'])
            (tmp_path/notice['receipt_path']).write_text(json.dumps(receipt))
            notice['receipt_sha256'] = digest(tmp_path/notice['receipt_path'])
    path = tmp_path/'terms.json';path.write_text(json.dumps(document))
    terms, _ = load_terms(tmp_path,path)
    row = terms['5871-2026-07-29']
    action = dict(action_id='5871-2026-07-29-stock',stock_id='5871',kind='stock_dividend',
        shares_per_share=row['shares_per_share'],pay_date=row['pay_date'],fractional_cash_per_share=10.)
    provider = SimpleNamespace(overrides=terms,on_date=lambda sid,day:[action] if day=='2026-07-29' else [])
    account = Replay.__new__(Replay)
    account.holdings = {'5871':dict(qty=8346,event_id='poc_up_red-2026-07-21-5871',due_index=9)}
    account.cohorts = [dict(event_id='poc_up_red-2026-07-21-5871',stock_id='5871',entry_date='2026-07-22',due_index=9)]
    account.receivables=[];account.actions=[];account.cash_ledger=[];account.cash=1000.
    account.marks={'5871':dict(price=120.)};account.raw=lambda day,sid:dict(close=120.)
    account.corporate=FractionalCashActions(provider,account)
    account.corporate_day(pd.Timestamp('2026-07-29'))
    shares,=[r for r in account.receivables if r['kind']=='shares']
    assert shares['qty']==166 and shares['pay_date']=='2026-09-01'
    account.holdings={}
    account.corporate_day(pd.Timestamp('2026-08-31'))
    assert not account.holdings
    account.corporate_day(pd.Timestamp('2026-09-01'))
    assert account.holdings['5871']['qty']==166
    assert account.cash==1000. and account.receivables[0]['amount']==9.
    assert account.receivables[0]['pay_date'] is None
