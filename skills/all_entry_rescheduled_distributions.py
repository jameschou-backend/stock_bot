"""Bind reviewed settlement terms before the existing holiday calendar shifts dates."""
from copy import deepcopy
import json
from pathlib import Path

from scripts.summarize_poc_broker_account import Sources
from skills.all_entry_accounts import digest
from skills.all_entry_bank_distributions import load_distributions as load_banks
from skills.all_entry_issuer_distributions import load_distributions as load_issuers

DOCUMENTS = {
    'docs/all_entry_bank_distributions_20261011.json': 'd93e19817bbb359d0bcd0af549ca7232e33aac6e11f3ef9b49e7d2ce571fc258',
    'docs/all_entry_issuer_distributions_20261011.json': '92e7a5bf66be93a63e155aa3454bd387ccf23f4f9f8fe5ac3ee143718c1e66c2',
    'docs/liquidity_universe_corporate_terms_v2_20261001.json': 'cab2b521ba1ea4ec557db9ce9ab644180b8d9489a33dfbc1dd2c56b67a72f4be',
}
EVENTS = ('1519', '2884')
SCHEDULED, EFFECTIVE = '2024-07-25', '2024-07-26'


def bind_scheduled_aliases(existing, verified, corrections):
    result = deepcopy(existing)
    for sid in EVENTS:
        old, new = sid+'-'+SCHEDULED, sid+'-'+EFFECTIVE
        matched = [r for r in corrections if r['stock_id'] == sid and r['scheduled_date'] == SCHEDULED]
        if (len(matched) != 1 or matched[0]['effective_date'] != EFFECTIVE
                or matched[0]['official_url'] != 'https://www.twse.com.tw/rwd/zh/exRight/TWT49U'
                or matched[0]['official_row'][:2] != ['113年07月25日', sid]):
            raise ValueError('Scheduled distribution has no exact reviewed calendar move')
        if new not in verified or existing.get(new) != verified[new]:
            raise ValueError('Effective-date distribution is absent or changed')
        if old in existing and existing[old] != verified[new]:
            raise ValueError('Scheduled-date terms conflict with reviewed distribution')
        # Preparation looks up the original official event date. The frozen
        # calendar moves that one action to the effective date afterwards.
        # The effective key remains for per-account fractional cash rounding.
        result[old] = deepcopy(verified[new])
    return result


def install_distributions(provider, root):
    root = Path(root).resolve(); sources = Sources(root)
    for path, sha in DOCUMENTS.items(): sources.bind(path, sha)
    names = list(DOCUMENTS)
    banks, bank_refs = load_banks(root, root/names[0])
    issuers, issuer_refs = load_issuers(root, root/names[1])
    calendar = json.loads((root/names[2]).read_text())
    for path,sha in {**bank_refs, **issuer_refs, **calendar['source_sha256']}.items():
        sources.bind(path, sha)
    provider.corporate_overrides = bind_scheduled_aliases(provider.corporate_overrides,
        {**banks, **issuers}, calendar['rescheduled'])
    return dict(sources.used)
