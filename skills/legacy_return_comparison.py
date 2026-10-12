"""Fixed old strategy identities under the current common account assumptions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'.cache/legacy-return-comparison-20261012'
SPEC = ROOT/'docs/prereg_legacy_return_comparison_20261012.md'
ARMS = ('liquid_universe', 'poc_priority', 'benchmark')
NAMES = dict(liquid_universe='擴大流動股池（原949%）',
             poc_priority='POC優先＋紅K（原576%）', benchmark='0050含息基準')


def apply_original_exit(engine, arm):
    """Keep the 949 strategy's original loss12/time63 without adding black bars."""
    if arm not in ARMS:
        raise ValueError('Unregistered legacy strategy')
    if arm == 'liquid_universe':
        engine.drawdown_arm = 'control'


def audit_original_exit(account):
    if account.get('black_log') or any(t['reason'] == 'three_black' for t in account['trades']):
        raise ValueError('949 strategy must not acquire the later three-black exit')
    return dict(three_black_enabled=False, exit_policy='loss12_time63')
