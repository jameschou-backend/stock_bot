"""User risk design beside the sealed 20% simulation, never a live order gate."""
from decimal import Decimal as D
import json
from pathlib import Path

PATH=Path(__file__).resolve().parents[1]/'docs/trading_risk_preference_20260914.json'


def load(path=PATH):
    profile=json.loads(Path(path).read_text())
    capital=D(profile['initial_capital_twd']); limit=D(profile['maximum_loss_fraction'])
    if not capital.is_finite() or capital<=0 or not limit.is_finite() or not 0<limit<1:
        raise ValueError('本金與最大虧損比例不合法')
    if profile['measurement']!='close_nav_high_water_mark_drawdown':
        raise ValueError('未支援的虧損衡量方式')
    if profile['applied_to_existing_simulation'] or profile['broker_orders_authorized']:
        raise ValueError('偏好檔不能自行切換封存策略或取得下單授權')
    return profile


def assess(nav_history, profile=None, external_cashflows=False):
    profile=load() if profile is None else profile
    if external_cashflows or not nav_history or any(x is None for x in nav_history):
        return dict(status='unknown',reason='缺少估值或入出金尚未校正',live_qualified=False)
    values=[D(str(v)) for v in nav_history]
    if any(not v.is_finite() or v<0 for v in values):
        return dict(status='unknown',reason='估值不是有效非負數',live_qualified=False)
    peak=D(profile['initial_capital_twd']); limit=D(profile['maximum_loss_fraction'])
    breached=False; first_breach=None
    for index,nav in enumerate(values):
        peak=max(peak,nav)
        if nav<=peak*(1-limit):
            breached=True
            if first_breach is None: first_breach=index
    return dict(status='review_required' if breached else 'below_threshold',
                peak=str(peak),trigger_nav=str(peak*(1-limit)),current_nav=str(values[-1]),
                drawdown=str(1-values[-1]/peak),first_breach_index=first_breach,
                applied_to_existing_simulation=False,live_qualified=False)
