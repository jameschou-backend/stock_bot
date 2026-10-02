"""Derived adjusted closes from independent price factors, never raw-close aliases."""
import math
import numpy as np
import pandas as pd
from skills.market_input_validation import require


def derive_adjusted_close(stamp, raw_close, independent, official_raw, frozen_quality, events):
    """Use nearest independently observed factors bracketing a missing quote.

    A dated corporate action in the bracket, factor discontinuity, missing
    official anchor, or basis inconsistency blocks derivation. The result is
    explicitly derived and must not be counted as a direct provider observation.
    """
    day = pd.Timestamp(stamp)
    require(math.isfinite(float(raw_close)) and raw_close > 0, 'Missing positive official raw close')
    require(all(s.index.is_unique and s.index.is_monotonic_increasing
                for s in (independent, official_raw, frozen_quality)), 'Invalid independent factor axes')
    require(day not in independent.index or not pd.notna(independent.at[day]), 'Target already independently observed')
    joined = pd.concat([independent.rename('independent'),official_raw.rename('raw'),
                        frozen_quality.rename('frozen')],axis=1)
    good = np.isfinite(joined).all(axis=1) & joined.gt(0).all(axis=1)
    before,after = joined.index[good & (joined.index < day)],joined.index[good & (joined.index > day)]
    require(len(before)>0 and len(after)>0, 'Independent factor needs two official anchors')
    anchors = [before[-1],after[0]]
    require((anchors[1]-anchors[0]).days <= 10, 'Independent factor anchor gap is too wide')
    dates = pd.to_datetime(events)
    require(not ((dates>anchors[0]) & (dates<=anchors[1])).any(), 'Corporate action crosses factor bracket')
    data = joined.loc[anchors]
    factors = data.independent/data.raw
    scales = data.frozen/data.independent
    factor,scale = float(factors.mean()),float(scales.mean())
    require(np.isfinite(factors).all() and np.isfinite(scales).all()
            and (abs(factors/factor-1)<1e-6).all() and (abs(scales/scale-1)<1e-6).all(),
            'Independent factor or adjustment basis changed across target')
    value = float(raw_close)*factor*scale
    require(math.isfinite(value) and value>0, 'Invalid derived adjusted value')
    return dict(value=value,method='derived_from_bracketed_independent_adjustment_factors',
        direct_provider_observation=False,source_is_revised_snapshot=True,
        anchor_dates=[str(d.date()) for d in anchors],
        anchor_independent_prices=[float(x) for x in data.independent],
        anchor_official_raw_prices=[float(x) for x in data.raw],
        anchor_frozen_quality_prices=[float(x) for x in data.frozen],
        independent_factor=factor,frozen_basis_scale=scale,corporate_action_in_bracket=False)
