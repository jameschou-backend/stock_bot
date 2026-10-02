import json

import pandas as pd
import pytest

from skills.ordinary_volume_bundle import digest, load_ordinary_matrices


def evidence(tmp_path):
    frame = pd.DataFrame([
        dict(date='2025-01-02', stock_id='1101', market='TWSE',
             volume_scope='all_daily_sessions', volume=900_000, table_category='股票'),
        dict(date='2025-01-02', stock_id='5314', market='TPEX',
             volume_scope='ordinary_session', volume=200_000, table_category='股票'),
        dict(date='2025-01-03', stock_id='5314', market='TPEX',
             volume_scope='unclassified_daily', volume=999_999, table_category='股票'),
        dict(date='2025-01-02', stock_id='4415', market='TPEX',
             volume_scope='ordinary_session', volume=2000, table_category='管理股票')])
    for field in ('open', 'high', 'low', 'close'):
        frame[field] = 10.
    quote = tmp_path/'official-normalized.parquet'
    frame.to_parquet(quote, index=False)
    raw = tmp_path/'source.json'
    raw.write_text('{}')
    manifest = tmp_path/'official-sources.json'
    manifest.write_text(json.dumps(dict(schema='official_quote_repair_evidence_v1',
        sources={str(i):dict(market='TPEX',date=d) for i,d in enumerate(['2025-01-02','2025-01-03','2025-01-06'])},
        source_sha256={'source.json': digest(raw)},
        output_sha256={quote.name: digest(quote)})))
    manifest.with_suffix('.sha256').write_text(digest(manifest))
    return manifest


def test_only_explicit_ordinary_scope_enters_matrix(tmp_path):
    manifest = evidence(tmp_path)
    days = pd.date_range('2025-01-02', periods=2)
    refs = {}
    frames = load_ordinary_matrices(tmp_path, manifest.name, days, ['1101', '5314', '4415'], refs)
    assert frames['TWSE'].isna().all().all()
    assert frames['TPEX'].at[days[0], '5314'] == 200_000
    assert pd.isna(frames['TPEX'].at[days[1], '5314'])
    assert frames['TPEX']['4415'].isna().all()
    assert 'source.json' in refs and 'official-normalized.parquet' in refs


def test_changed_original_source_invalidates_normalized_cache(tmp_path):
    manifest = evidence(tmp_path)
    (tmp_path/'source.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='source changed'):
        load_ordinary_matrices(tmp_path, manifest.name, pd.date_range('2025-01-02', periods=2), ['5314'], {})


def test_verified_halt_can_fill_volume_zero_but_conflict_is_rejected(tmp_path):
    manifest = evidence(tmp_path)
    notice = tmp_path/'halt.html'
    notice.write_text('advance official suspension notice fixture')
    halt = dict(stock_id='5314', market='TPEX', announcement_date='2024-12-01',
                start='2025-01-06', end='2025-01-07', source_path=notice.name, source_sha256=digest(notice))
    days = pd.bdate_range('2025-01-02', periods=3)
    frames = load_ordinary_matrices(tmp_path, manifest.name, days, ['5314'], {}, halts=[halt])
    assert frames['TPEX'].at[days[2], '5314'] == 0
    halt['start'] = '2025-01-03'
    with pytest.raises(ValueError, match='conflicts'):
        load_ordinary_matrices(tmp_path, manifest.name, days, ['5314'], {}, halts=[halt])
