from copy import deepcopy
from datetime import date
import json

import pandas as pd
import pytest

from skills import historical_identity_repair as repair


DATES = {
    '5292': ('2023-11-21', None, '2020-01-07'),
    '6757': ('2023-08-15', '2024-11-29', '2024-11-29'),
    '6794': ('2024-05-21', '2025-10-16', '2025-10-16'),
    '6869': ('2023-03-14', '2024-06-19', '2024-06-19'),
    '6873': ('2023-03-06', '2024-09-26', '2024-09-26'),
    '6902': ('2023-07-13', '2025-05-16', '2025-05-16'),
    '7780': ('2025-09-09', None, '2026-01-19'),
}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def seal(root, value):
    path = root / 'repair.json'
    write(path, value)
    path.with_suffix('.sha256').write_text(repair.sha(path))
    return path


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    sources, approved, entries, rows, episodes = {}, {}, [], [], []

    def extraction(key, text):
        path = tmp_path / (key + '.json')
        url = 'https://www.twse.com.tw/staticFiles/' + key + '.pdf'
        write(path, dict(schema='official_pdf_web_extraction_v1', source_kind='official_pdf_web_extraction',
            source_url=url, retrieved_at='2026-10-02T01:00:00Z', raw_pdf_bytes_obtained=False,
            http_status=None, pdf_pages_1based=[1], extraction=text))
        sources[path.name] = approved[key] = repair.sha(path)
        return dict(path=path.name, source_url=url, pdf_pages_1based=[1])

    for sid, (start, transfer, snapshot_start) in DATES.items():
        day = date.fromisoformat(start)
        row = [sid, f'{day.year-1911}.{day.month:02}.{day.day:02}', '創新板' if transfer else '']
        rows.append(row)
        ep = dict(stock_id=sid, market='TWSE', category='股票', start=snapshot_start, end=None,
                  start_evidence='current_official_ISIN', snapshot_date='2026-09-14')
        episodes.append(ep)
        e = dict(stock_id=sid, market='TWSE', category='股票', start=start, replaces_episode=ep,
                 listing_source_row=row, initial_board='innovation' if transfer else 'general',
                 general_board_start=transfer, interval='start_inclusive_end_exclusive')
        if transfer:
            d = date.fromisoformat(transfer)
            e['transfer_source'] = extraction(sid, f'創新板 {sid} 普通股 {d.year-1911}年{d.month}月{d.day}日開始改列上市買賣')
        entries.append(e)
    base = dict(episodes=episodes, trading_exclusions=[], coverage_start='2018-01-02',
                coverage_end='2026-09-09', source_sha256={}, live_qualified=False)
    fields = ['公司代號', '股票上市買賣日期', '備註']
    write(tmp_path / 'base.json', base)
    write(tmp_path / 'ipo.json', dict(stat='OK', fields=fields, total=len(rows), data=rows))
    write(tmp_path / 'ipo.receipt.json', dict(url=repair.IPO_URL, params={'response':'json'},
          sha256=repair.sha(tmp_path/'ipo.json'), retrieved_at='2026-09-14T05:34:16+00:00'))
    for name in ('base.json', 'ipo.json', 'ipo.receipt.json'):
        sources[name] = repair.sha(tmp_path/name)
    monkeypatch.setattr(repair, 'REVIEWED_IPO_SOURCE', dict(raw_sha256=sources['ipo.json'],
                        receipt_sha256=sources['ipo.receipt.json']))
    policy = dict(name='ordinary_retail_research', innovation_reform_date='2025-01-06',
        pre_reform_qualified_investor_assumed=False, risk_notice_assumed_by_default=False,
        regular_lot_shares=1000,
        old_rule_source=extraction('tib_rule_2023', '1,000股 可盤後零股交易 不可盤中零股交易'),
        reform_source=extraction('tib_rule_2025', '2025年1月6日 取消合格投資人 初次買進前須已簽署風險預告書 盤中零股交易'))
    monkeypatch.setattr(repair, 'REVIEWED_EXTRACTIONS', approved)
    manifest = dict(schema=repair.SCHEMA, live_qualified=False, complete_historical_universe=False,
        publication_time_archive_complete=False, source_sha256=sources, entries=entries,
        base_identity=dict(path='base.json', sha256=sources['base.json']), account_policy=policy,
        listing_source=dict(raw_path='ipo.json', raw_sha256=sources['ipo.json'],
            receipt_path='ipo.receipt.json', receipt_sha256=sources['ipo.receipt.json'],
            fields=fields, http_status_evidence='legacy_status_unknown'))
    return tmp_path, manifest, base


def load(evidence):
    root, manifest, base = evidence
    return repair.load_identity_repair(seal(root, manifest), root)


def applied(evidence):
    return repair.apply_identity_repair(evidence[2], load(evidence))


def test_listing_boundaries_and_base_immutability(evidence):
    original = deepcopy(evidence[2])
    report = applied(evidence)
    assert evidence[2] == original
    assert len(report['episodes']) == 12
    for sid, (start, transfer, _) in DATES.items():
        eps = [e for e in report['episodes'] if e['stock_id'] == sid]
        assert eps[0]['start'] == start
        assert eps[0]['category'] == '股票'
        if transfer:
            assert eps[0]['board'] == 'innovation'
            assert eps[0]['end'] == eps[1]['start'] == transfer
            assert eps[1]['board'] == 'general'
    assert not repair.account_entry_decision(report, '5292', '2023-11-20')['allowed']
    assert repair.account_entry_decision(report, '5292', '2023-11-21')['allowed']
    assert not repair.account_entry_decision(report, '7780', '2025-09-08')['allowed']
    assert repair.account_entry_decision(report, '7780', '2025-09-09')['allowed']
    assert not report['live_qualified'] and not report['performance_recomputed']


@pytest.mark.parametrize('field,value', [('start','2023-03-05'), ('general_board_start','2024-09-25'),
    ('category','創新板'), ('initial_board','general'), ('market','TPEX'), ('interval','inclusive')])
def test_semantically_changed_repair_rejected_even_resealed(evidence, field, value):
    e = next(e for e in evidence[1]['entries'] if e['stock_id'] == '6873')
    e[field] = value
    with pytest.raises(ValueError):
        load(evidence)


@pytest.mark.parametrize('mutation', ['duplicate','missing','newtarget','source_hash','escape','source_reseal',
    'base_episode','fake_http','promote','policy','page','url'])
def test_bad_evidence_rejected(evidence, mutation):
    root, manifest, _ = evidence
    e = manifest['entries'][1]
    if mutation == 'duplicate': manifest['entries'].append(deepcopy(e))
    elif mutation == 'missing': manifest['entries'].pop()
    elif mutation == 'newtarget': e['stock_id'] = '2330'
    elif mutation == 'source_hash': manifest['source_sha256']['base.json'] = '0'*64
    elif mutation == 'escape': manifest['source_sha256']['../outside'] = '0'*64
    elif mutation == 'source_reseal':
        p = root/'6757.json'; p.write_text(p.read_text().replace('29日','28日'))
        manifest['source_sha256']['6757.json'] = repair.sha(p)
    elif mutation == 'base_episode': e['replaces_episode']['start'] = '2000-01-01'
    elif mutation == 'fake_http': manifest['listing_source']['http_status_evidence'] = 'recorded_http_200'
    elif mutation == 'promote': manifest['live_qualified'] = True
    elif mutation == 'policy': manifest['account_policy']['pre_reform_qualified_investor_assumed'] = True
    elif mutation == 'page': e['transfer_source']['pdf_pages_1based'] = [2]
    elif mutation == 'url': e['transfer_source']['source_url'] = 'https://example.com/a.pdf'
    with pytest.raises(ValueError): load(evidence)


def test_manifest_and_complete_source_closure(evidence):
    root, manifest, _ = evidence
    refs = {}
    path = seal(root, manifest)
    repair.load_identity_repair(path, root, refs)
    assert refs == dict(manifest['source_sha256'], **{'repair.json':repair.sha(path)})
    with pytest.raises(ValueError, match='closure conflict'):
        repair.load_identity_repair(path, root, {'base.json':'0'*64})
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError, match='manifest hash'):
        repair.load_identity_repair(path, root)


def test_edited_ipo_and_receipt_cannot_be_self_resealed(evidence):
    root, manifest, _ = evidence
    path = root/'ipo.json'
    raw = json.loads(path.read_text())
    raw['data'][0][1] = '109.01.07'
    write(path, raw)
    receipt_path = root/'ipo.receipt.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['sha256'] = repair.sha(path)
    write(receipt_path, receipt)
    for name, key in [('ipo.json','raw_sha256'),('ipo.receipt.json','receipt_sha256')]:
        manifest['source_sha256'][name] = manifest['listing_source'][key] = repair.sha(root/name)
    with pytest.raises(ValueError, match='IPO archive needs independent review'):
        load(evidence)


def test_account_policy_no_inferred_qualification_and_exclusive_transfer(evidence):
    report = applied(evidence)
    before = repair.account_entry_decision(report, '6873', '2024-09-25', research_risk_notice_assumed=True)
    assert before['reason'] == 'qualified_investor_status_not_proven'
    assert not before['allowed']
    assert repair.account_entry_decision(report, '6873', '2024-09-26')['allowed']
    assert not repair.account_entry_decision(report, '6902', '2025-01-06')['allowed']
    yes = repair.account_entry_decision(report, '6902', '2025-01-06', research_risk_notice_assumed=True)
    assert yes['allowed'] and yes['assumptions'] and not yes['live_qualified']


def test_innovation_intraday_odd_lot_cannot_execute_before_reform(evidence):
    report = applied(evidence)
    no = repair.account_entry_decision(report, '6902', '2025-01-03', channel='intraday_odd',
                                      research_risk_notice_assumed=True, quantity=90)
    assert no['reason'] == 'innovation_intraday_odd_not_available'
    yes = repair.account_entry_decision(report, '6902', '2025-01-06', channel='intraday_odd',
                                       research_risk_notice_assumed=True, quantity=90)
    assert yes['allowed']
    for channel, qty in [('regular',90),('intraday_odd',1000),('afterhours_odd',1001)]:
        assert not repair.account_entry_decision(report, '6902', '2025-01-06', channel=channel,
            quantity=qty, research_risk_notice_assumed=True)['allowed']


def test_existing_suspensions_unknowns_and_snapshot_are_not_promoted(evidence):
    report = applied(evidence)
    report['trading_exclusions'] = [dict(stock_id='7780', market='TWSE', start='2026-01-05',
                                        end='2026-01-19', kind='trading_suspension')]
    assert not repair.account_entry_decision(report, '7780', '2026-01-18')['allowed']
    assert repair.account_entry_decision(report, '7780', '2026-01-19')['allowed']
    assert not repair.account_entry_decision(report, '9999', '2025-01-01')['allowed']
    assert not repair.account_entry_decision(report, '7780', '2026-10-01')['allowed']
    report['coverage_end'] = '2026-10-01'
    assert not repair.account_entry_decision(report, '7780', '2026-09-15')['allowed']


def test_matrix_policy_copies_and_never_turns_on_mask(evidence):
    report = applied(evidence)
    idx = pd.to_datetime(['2025-01-03','2025-01-06','2025-05-16'])
    mask = pd.DataFrame({'6902':[True,True,False], '5292':[False,True,True]}, index=idx)
    before = mask.copy()
    filtered = repair.apply_account_entry_policy(mask, report, research_risk_notice_assumed=True)
    assert filtered['6902'].tolist() == [False,True,False]
    pd.testing.assert_frame_equal(mask, before)
    assert filtered['5292'].equals(mask['5292'])
    assert not filtered.attrs['live_qualified']


def test_modified_base_cannot_be_applied_twice(evidence):
    loaded = load(evidence)
    output = repair.apply_identity_repair(evidence[2], loaded)
    with pytest.raises(ValueError, match='Base episode changed'):
        repair.apply_identity_repair(output, loaded)
