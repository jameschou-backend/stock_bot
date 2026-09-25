"""Restore only independently evidenced pre-transfer quotes into a new snapshot."""
import numpy as np
import pandas as pd

from scripts.prepare_million_signals import official_adjusted


def restore(frames, companies, events, prefix, quality, plan):
    result = {key: frame.copy() for key, frame in frames.items()}
    companies = companies.copy()
    prefix = prefix.copy()
    prefix['date'] = pd.to_datetime(prefix.date)
    anchors = []
    for row in plan['rows']:
        sid = row['stock_id']
        dates = (result['raw-close'].index >= pd.Timestamp(row['start'])) & (
            result['raw-close'].index < pd.Timestamp(row['end_exclusive']))
        if result['raw-close'].loc[dates, sid].notna().any():
            raise ValueError('Prefix repair would overwrite an existing frozen price: ' + sid)
        source = prefix[prefix.stock_id.eq(sid)].set_index('date')
        if source.empty or source.index.has_duplicates or not ((source.index >= pd.Timestamp(row['start'])) & (
                source.index < pd.Timestamp(row['end_exclusive']))).all():
            raise ValueError('Prefix source lies outside the verified transfer interval: ' + sid)
        for name, col in (('raw-close', 'close'), ('raw-volume', 'volume')):
            result[name].loc[dates, sid] = source[col].reindex(result[name].index[dates]).to_numpy()
        reconstructed = official_adjusted(result['raw-close'][[sid]], events)
        result['close-official'].loc[dates, sid] = reconstructed.loc[dates, sid]
        fresh = quality[sid].reindex(result['close-quality'].index)
        old = result['close-quality'][sid]
        common = old.notna() & fresh.notna() & old.gt(0) & fresh.gt(0)
        if not old.notna().any():
            if not fresh.gt(0).any():
                raise ValueError('Both adjusted-price versions are absent: ' + sid)
            # Explicit missing-column repair, not an official-price substitute.
            # The independent fresh provider column supplies the observations.
            result['close-quality'][sid] = fresh.where(result['raw-close'][sid].gt(0)
                & result['raw-volume'][sid].gt(0))
            companies.loc[companies.stock_id.eq(sid), 'listed_date'] = pd.Timestamp(row['start'])
            anchors.append(dict(stock_id=sid, mode='previous_quality_column_entirely_missing',
                anchor_date=None, unit_scale=1., restored_rows=int(result['raw-close'].loc[dates, sid].notna().sum())))
            continue
        if not common.any():
            raise ValueError('No common adjusted-price unit anchor: ' + sid)
        anchor = common[common].index[0]
        scale = float(old.at[anchor] / fresh.at[anchor])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Invalid adjusted-price unit scale: ' + sid)
        # A constant unit conversion only; do not rewrite original quote history.
        # Relative returns and correlation are invariant to this stockwise scale.
        result['close-quality'].loc[dates, sid] = fresh.loc[dates] * scale
        result['close-quality'].loc[dates, sid] = result['close-quality'].loc[dates, sid].where(
            result['raw-close'].loc[dates, sid].gt(0) & result['raw-volume'].loc[dates, sid].gt(0))
        companies.loc[companies.stock_id.eq(sid), 'listed_date'] = pd.Timestamp(row['start'])
        anchors.append(dict(stock_id=sid, mode='preserve_existing_quality_with_constant_unit_conversion',
                            anchor_date=str(anchor.date()), unit_scale=scale,
                            restored_rows=int(result['raw-close'].loc[dates, sid].notna().sum())))
    return result, companies, anchors
