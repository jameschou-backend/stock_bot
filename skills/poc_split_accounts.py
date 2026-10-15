"""Fixed 500k funding, preserving all original sizing and accounting checks."""
from copy import deepcopy

VARIANTS = ('split50',)
CAPITAL = 500000


def audit_funding(account):
    if account['settings']['initial_cash'] != CAPITAL:
        raise ValueError('Split sleeve must start with exactly 500000')
    deposits = [r for r in account['cash_ledger'] if r['kind'] == 'initial_deposit']
    if len(deposits) != 1 or deposits[0]['cash_change'] != CAPITAL:
        raise ValueError('Split sleeve initial funding differs')
    if any(r['kind'] in ('deposit', 'withdrawal', 'transfer', 'rebalance') for r in account['cash_ledger']):
        raise ValueError('Transfers or rebalancing are not registered')
    if any(r['cash'] < -.011 for r in account['daily']):
        raise ValueError('Sleeve borrowed cash from another account')
    return dict(initial_cash=CAPITAL, independent_cash=True, rebalance=False,
                scaled_original_result=False, checked_days=len(account['daily']))


def install(runner, variant, metadata):
    if variant != 'split50' or metadata['initial_cash'] != CAPITAL:
        raise ValueError('Unregistered split allocation')
    original = runner.engine_types

    class Funding:
        def __init__(self, *args, **kwargs):
            # The sealed orchestration passes 1m. Change only the constructor's
            # funding input before any integer sizing, fees, or reservations.
            if kwargs.get('initial_cash') != 1000000:
                raise ValueError('Frozen funding seam changed')
            kwargs['initial_cash'] = CAPITAL
            super().__init__(*args, **kwargs)

        def run(self):
            result = super().run()
            result['split_funding_audit'] = audit_funding(result)
            return result

    def engines(*args, **kwargs):
        stock, benchmark = original(*args, **kwargs)
        return type('HalfCapital'+stock.__name__, (Funding, stock), {}), benchmark
    runner.engine_types = engines
    writer = runner.write

    def write(path, value):
        if 'account' in value or str(path).endswith('/report.json'):
            value['split_accounts'] = deepcopy(metadata)
        if str(path).endswith('/report.json'):
            if value.get('initial_cash') != 1000000:
                raise ValueError('Frozen report funding seam changed')
            value['initial_cash'] = CAPITAL
        return writer(path, value)
    runner.write = write
