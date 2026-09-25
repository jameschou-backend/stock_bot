import pytest

from skills.publication_versions import (parse_release, extend_versions, as_of, index_links,
    verify_archive, digest, encoded, INDEX_URL)

URL='https://www.passivecomponent.com/2026/04/09/walsin-technology-global-consolidated-net-sales-for-march-2026/'


def release(body='Original content', day='April 9th, 2026', updated='2020-01-01T00:00:00Z'):
    return (f'<h1 class="entry-title">Walsin Technology global consolidated net sales for March 2026</h1>'
            f'<div class="post-content">{body}</div><div class="fusion-meta-info">'
            f'<span class="updated">{updated}</span><span>{day}</span></div>').encode()


def observation(body, observed):
    return dict(parse_release(release(body),URL), observed_at=observed,path='release.html')


def test_revenue_period_and_cms_update_are_not_publication_time():
    row=parse_release(release(),URL)
    assert row['revenue_period']=='2026-03'
    assert row['claimed_publication_date']=='2026-04-09'
    assert row['official_published_at'] is None
    assert row['publication_date_upper_bound']=='2026-04-10T00:00:00+08:00'


def test_today_capture_cannot_be_inserted_into_april_history():
    rows=extend_versions([], [observation('Original', '2026-09-25T12:00:00+08:00')])
    assert as_of(rows,'2026-04-02T23:59:59+08:00')==[]
    assert as_of(rows,'2026-04-10T12:00:00+08:00')==[]
    assert len(as_of(rows,'2026-09-25T12:00:00+08:00'))==1


def test_later_revision_never_overwrites_prior_asof_content():
    first=observation('Original','2026-09-25T12:00:00+08:00')
    rows=extend_versions([], [first])
    updated=extend_versions(rows,[observation('Correction','2026-09-26T12:00:00+08:00')])
    assert as_of(updated,'2026-09-25T13:00:00+08:00')==rows
    assert as_of(updated,'2026-09-27T00:00:00+08:00')[0]['content_sha256']!=rows[0]['content_sha256']
    assert updated[1]['previous_observation_id']==rows[0]['observation_id']


def test_identical_body_preserves_first_observed_version_time():
    rows=extend_versions([], [observation('Same','2026-09-25T12:00:00+08:00')])
    new=observation('Same','2026-09-26T12:00:00+08:00')
    new['raw_sha256']='a'*64  # Template changed, content did not.
    latest=extend_versions(rows,[new])[-1]
    assert latest['version_id']==rows[0]['version_id']
    assert latest['version_first_observed_at']==rows[0]['version_first_observed_at']
    assert latest['content_changed'] is False


@pytest.mark.parametrize('cutoff',['2026-09-25T11:00:00+08:00','2026-09-25T12:00:00+08:00'])
def test_observation_cannot_be_backdated_or_ambiguous(cutoff):
    rows=extend_versions([], [observation('One','2026-09-25T12:00:00+08:00')])
    with pytest.raises(ValueError,match='follow prior'):
        extend_versions(rows,[observation('Two',cutoff)])


def test_conflicting_visible_date_fails_closed():
    with pytest.raises(ValueError,match='differs'):
        parse_release(release(day='April 8th, 2026'),URL)


@pytest.mark.parametrize('markup', [
    '<span class="updated">April 9th, 2026</span>',
    '<span style="display: none">April 9th, 2026</span>',
    '<div aria-hidden="true"><span>April 9th, 2026</span></div>',
])
def test_hidden_cms_date_cannot_replace_visible_publication_date(markup):
    raw=release().decode().replace('<span>April 9th, 2026</span>', markup).encode()
    with pytest.raises(ValueError,match='Visible official publication date missing'):
        parse_release(raw,URL)


def test_no_timezone_is_not_a_historical_cutoff():
    with pytest.raises(ValueError,match='timezone'):
        as_of([], '2026-04-02T00:00:00')


def test_index_rejects_nonofficial_release_link():
    raw=f'<article><h2 class="entry-title"><a href="{URL.replace("www.passivecomponent.com","example.com")}">global consolidated net sales</a></h2></article>'.encode()
    with pytest.raises(ValueError,match='official'):
        index_links(raw)


def archive_fixture(folder):
    observed='2026-09-25T12:00:00+08:00'
    files={}
    def source(name,raw,url):
        (folder/name).write_bytes(raw)
        receipt=dict(path=name,url=url,observed_at=observed,http_status=200,sha256=digest(raw))
        (folder/(name+'.source.json')).write_bytes(encoded(receipt))
        files[name]=digest(raw)
        files[name+'.source.json']=digest(encoded(receipt))
        return receipt
    index=source('index.html', f'<article><h2 class="entry-title"><a href="{URL}">global consolidated net sales</a></h2></article>'.encode(),INDEX_URL)
    receipt=source('release.html',release(),URL)
    obs=dict(parse_release(release(),URL),path='release.html',observed_at=observed)
    attempts=[dict(url=url,status='received',http_status=200) for url in (INDEX_URL,URL)]
    (folder/'attempts.json').write_bytes(encoded(attempts))
    files['attempts.json']=digest(encoded(attempts))
    value=dict(format='issuer_publication_versions_v1',historical_complete=False,live_qualified=False,
        finmind_requests=0,network_requests=2,files_sha256=files,index=index,observations=[obs],
        versions=extend_versions([], [obs]),parent=None)
    return value


def seal(folder,value):
    path=folder/'archive.json'
    path.write_bytes(encoded(value))
    path.with_suffix('.sha256').write_text(digest(path.read_bytes()))
    return path


def test_archive_reparses_raw_and_binds_receipt_timestamp(tmp_path):
    value=archive_fixture(tmp_path)
    assert verify_archive(seal(tmp_path,value))==value
    value['observations'][0]['observed_at']='2026-04-10T00:00:00+08:00'
    value['versions']=extend_versions([],value['observations'])
    with pytest.raises(ValueError,match='source receipt'):
        verify_archive(seal(tmp_path,value))


def test_archive_does_not_trust_unbound_receipts(tmp_path):
    value=archive_fixture(tmp_path)
    del value['files_sha256']['release.html.source.json']
    with pytest.raises(ValueError,match='lacks bound'):
        verify_archive(seal(tmp_path,value))


def test_changed_raw_content_invalidates_archive(tmp_path):
    value=archive_fixture(tmp_path)
    path=seal(tmp_path,value)
    (tmp_path/'release.html').write_bytes(release('Correction'))
    with pytest.raises(ValueError,match='source changed'):
        verify_archive(path)


def test_archive_cannot_claim_requests_or_source_coverage_it_did_not_capture(tmp_path):
    value=archive_fixture(tmp_path)
    value['network_requests']=0
    with pytest.raises(ValueError,match='attempts differ'):
        verify_archive(seal(tmp_path,value))
