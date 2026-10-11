"""The newly encountered 7,444-share holding keeps its actual delivery clock."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from skills.all_entry_accounts import digest
from skills.all_entry_corporate import load_terms
from skills.million_replay import Replay
from skills.scenario_exit_replay import FractionalCashActions


def test_2887_delivery_preserves_74_shares_and_unspendable_fraction(tmp_path):
    document = json.loads((Path(__file__).resolve().parents[1] /
        'docs/all_entry_corporate_terms_v3_20261011.json').read_text())
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
    row = terms['2887-2026-07-21']
    action = dict(action_id='2887-2026-07-21-stock',stock_id='2887',kind='stock_dividend',
        shares_per_share=row['shares_per_share'],pay_date=row['pay_date'],fractional_cash_per_share=10.)
    provider = SimpleNamespace(overrides=terms,on_date=lambda sid,day:[action] if day=='2026-07-21' else [])
    account = Replay.__new__(Replay)
    account.holdings = {'2887':dict(qty=7444,event_id='donchian55-2026-06-12-2887',due_index=9)}
    account.cohorts = [dict(event_id='donchian55-2026-06-12-2887',stock_id='2887',entry_date='2026-06-15',due_index=9)]
    account.receivables=[];account.actions=[];account.cash_ledger=[];account.cash=1000.
    account.marks={'2887':dict(price=120.)};account.raw=lambda day,sid:dict(close=120.)
    account.corporate=FractionalCashActions(provider,account)
    account.corporate_day(pd.Timestamp('2026-07-21'))
    shares,=[r for r in account.receivables if r['kind']=='shares']
    assert shares['qty']==74 and shares['pay_date']=='2026-08-19'
    account.holdings={}
    account.corporate_day(pd.Timestamp('2026-08-18'))
    assert not account.holdings
    account.corporate_day(pd.Timestamp('2026-08-19'))
    assert account.holdings['2887']['qty']==74
    assert account.cash==1000. and account.receivables[0]['amount']==4.
    assert account.receivables[0]['pay_date'] is None
