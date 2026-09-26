#!/usr/bin/env python3
"""Publish the additive issuer payment-date check without changing any account."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from skills.index_dividend_reconciliation import reconcile
from scripts.research_exit_scenarios import write,sha
from skills.verified_backtest_tool import offline_only
from skills.backtest_case_cache import file_identities

if __name__=='__main__':
    output=ROOT/'artifacts/forward_simulation/index_dividends_20260927.json'
    if output.exists():raise ValueError('Additive publication already exists')
    with offline_only():
        value=reconcile();value['audit_code']=file_identities([Path(__file__),ROOT/'skills/index_dividend_reconciliation.py'],ROOT)
        write(output,value);output.with_suffix('.sha256').write_text(sha(output)+'\n')
    print('Matched',value['unique_period_dividends'],'distribution dates across',len(value['account_checks']),'accounts')
