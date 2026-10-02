import pandas as pd
import pytest
from skills.independent_factor_repair import derive_adjusted_close
from skills.market_input_validation import MarketEvidenceError


def inputs():
    idx=pd.to_datetime(['2022-12-20','2022-12-23'])
    return dict(stamp='2022-12-22',raw_close=60.,independent=pd.Series([40.,44.],index=idx),
        official_raw=pd.Series([50.,55.],index=idx),frozen_quality=pd.Series([20.,22.],index=idx),events=[])


def test_derivation_uses_independent_factor_and_frozen_basis_not_raw_alias():
    result=derive_adjusted_close(**inputs())
    assert result['value']==24.
    assert result['independent_factor']==.8 and result['frozen_basis_scale']==.5
    assert result['direct_provider_observation'] is False


@pytest.mark.parametrize('mutation',['no_after','factor_jump','basis_jump','action','wide','missing_raw'])
def test_missing_or_discontinuous_factor_evidence_fails_closed(mutation):
    values=inputs()
    if mutation=='no_after': values['independent']=values['independent'].iloc[:1]
    elif mutation=='factor_jump': values['independent'].iloc[-1]=45.
    elif mutation=='basis_jump': values['frozen_quality'].iloc[-1]=23.
    elif mutation=='action': values['events']=['2022-12-22']
    elif mutation=='wide':
        for key in ('independent','official_raw','frozen_quality'):
            values[key].index=pd.to_datetime(['2022-12-01','2022-12-23'])
    else: values['official_raw'].iloc[-1]=float('nan')
    with pytest.raises(MarketEvidenceError):derive_adjusted_close(**values)
