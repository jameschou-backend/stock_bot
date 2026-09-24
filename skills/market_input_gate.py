"""Source preflight shared by batch building and direct candidate publication."""
from datetime import date
from skills.price_coverage import require_market_coverage


def require_market_inputs(config, session):
    result = require_market_coverage(session)
    if getattr(config, 'market_filter_enabled', False):
        from skills.daily_pick import _load_market_price_df
        target = date.fromisoformat(result['market_coverage_target'])
        frame = _load_market_price_df(session, target, max(getattr(config, 'market_filter_ma_days', 60), 200))
        result['market_index_date'] = str(frame.trading_date.max())
    return result
