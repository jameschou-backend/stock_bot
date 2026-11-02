from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from urllib.parse import parse_qs, urlsplit

from fastapi.testclient import TestClient
import pyarrow.parquet as pq
import pytest

from us_market.alpaca import AlpacaClient, AlpacaError, AlpacaQuery, timestamp_ns
from us_market.alpaca_smoke import diagnose
from us_market.api import create_app
from us_market.config import alpaca_credentials
from us_market.fixtures import AlpacaFixtureClient
from us_market.pipeline import ingest
from us_market.query import Query
from us_market.store import Store


def query(dataset="trades", **kw):
    return AlpacaQuery(dataset,"AAPL","2026-10-29","2026-10-29",**kw)


def trade(ident=1, **kw):
    return dict(t="2026-10-29T14:00:00.123456789Z",p="100.123456789012",s="0.125",i=ident,x="V",z="C",**kw)


def page(rows, token=None):
    return dict(trades={"AAPL":rows},next_page_token=token)


class Pages:
    def __init__(self, pages):
        self.pages = list(pages)
        self.requests = 0

    def page(self, url):
        self.requests += 1
        value = self.pages.pop(0)
        if isinstance(value,Exception):
            raise value
        return value


class Response:
    def __init__(self,body,status=200): self.body=body; self.status_code=status
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def iter_content(self,size): yield self.body


class Session:
    def __init__(self,response): self.response=response; self.calls=[]
    def get(self,url,**kwargs): self.calls.append((url,kwargs)); return self.response


def client(session, **kw):
    return AlpacaClient("PRIVATE_KEY","PRIVATE_SECRET",session=session,
                        now=lambda:datetime(2026,11,2,tzinfo=timezone.utc),**kw)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path/"us",min_free_bytes=0)


def test_native_trade_schema_keeps_nanoseconds_fractional_shares_and_unknown_conditions(store):
    result,_ = ingest(query(),store,Pages([page([trade(),trade(2,c=["@","I"])])]))
    table = pq.read_table(store.root/result["pages"][0]["parquet"]["path"])
    rows = table.to_pylist()
    assert len(rows)==2 and rows[0]["timestamp_ns"] % 10**9 == 123456789
    assert rows[0]["shares"]==Decimal("0.125") and rows[0]["price"]==Decimal("100.123456789012")
    assert table["conditions"].to_pylist()==[None,["@","I"]]
    assert table["exchange"].to_pylist()==["V","V"]
    assert result["feed"]=="sip" and result["schema"]=="alpaca_us_snapshot_v1"
    assert not result["poc_ready"] and not result["live_qualified"]


def test_dst_date_boundary_uses_inclusive_last_nanosecond():
    q = AlpacaQuery("trades","BRK.B","2026-11-01","2026-11-01")
    params = parse_qs(urlsplit(q.url()).query)
    start,end = [timestamp_ns(params[k][0]) for k in ("start","end")]
    assert end-start+1==25*3600*10**9
    assert params["feed"]==["sip"] and params["asof"]==["-"]
    q.normalize_page(dict(trades={"BRK.B":[dict(trade(),t=params["end"][0])]}))
    with pytest.raises(ValueError,match="outside requested"):
        q.normalize_page(dict(trades={"BRK.B":[dict(trade(),t="2026-11-02T05:00:00Z")]}))
    assert timestamp_ns("2026-10-29T10:00:00.123456789-04:00")==timestamp_ns(trade()["t"])


def test_resume_empty_continuation_and_offline_replay_are_idempotent(store):
    q=query()
    with pytest.raises(AlpacaError,match="interrupted"):
        ingest(q,store,Pages([page([],"opaque+/token"),AlpacaError("interrupted")]))
    assert not store.snapshots()
    resumed = Pages([page([trade()])])
    result,ref = ingest(q,store,resumed)
    assert resumed.requests==1 and result["rows"]==1 and len(result["pages"])==2
    size=store.capacity()["used_bytes"]
    assert ingest(q,store)==(result,ref)
    assert size==store.capacity()["used_bytes"]


def test_repeated_identity_with_different_timestamp_is_a_conflict(store):
    with pytest.raises(ValueError,match="Duplicate"):
        ingest(query(),store,Pages([page([trade()],"two"),page([dict(trade(),t="2026-10-29T14:00:01Z")])]))
    assert not store.snapshots()


def test_page_budget_and_loop_never_publish_partial(store):
    with pytest.raises(AlpacaError,match="Page budget"):
        ingest(query(),store,Pages([page([trade()],"two")]),max_pages=1)
    assert not store.snapshots()
    with pytest.raises(AlpacaError,match="loop"):
        ingest(query(),store,Pages([page([trade(2)],"two")]))
    assert not store.snapshots()


@pytest.mark.parametrize("bad",[{}, {"AAPL":{}}, {"AAPL":""}, {"MSFT":[]}, []])
def test_invalid_symbol_map_fails_closed(bad):
    value={} if bad=={} else dict(trades=bad)
    with pytest.raises(ValueError): query().normalize_page(value)


@pytest.mark.parametrize("change",[dict(i=None),dict(x=None),dict(c=[1]),dict(p="NaN"),dict(s=-1)])
def test_malformed_trade_is_not_published(change,store):
    with pytest.raises(ValueError): ingest(query(),store,Pages([page([dict(trade(),**change)])]))
    assert not store.snapshots()


def test_fixture_source_cannot_be_mistaken_for_provider(store):
    fixture=AlpacaFixtureClient()
    with pytest.raises(ValueError,match="provenance"):
        query().normalize_page(fixture.page(query().url()))
    snapshots=[]
    for dataset in ("daily","trades"):
        q=query(dataset,source_kind="synthetic_fixture")
        snapshot,_=ingest(q,store,fixture)
        snapshots.append(snapshot)
        assert q.key!=query(dataset).key
        assert q.key!=Query(dataset,"AAPL","2026-10-29","2026-10-29",source_kind="synthetic_fixture").key
    report=diagnose(store,*snapshots)
    assert report["raw_volume_equal"] and report["source_kind"]=="synthetic_fixture"
    assert report["trade_rows"]==2 and not report["poc_ready"] and not report["strategy_ready"]
    api=TestClient(create_app(store))
    assert all(job["provider"]=="alpaca" and job["feed"]=="sip" for job in api.get("/jobs").json())


@pytest.mark.parametrize("url",[
    "https://paper-api.alpaca.markets/v2/orders", "https://evil.test/v2/stocks/trades",
    "https://data.alpaca.markets:443/v2/stocks/trades", "http://data.alpaca.markets/v2/stocks/trades",
    query().url().replace("feed=sip","feed=iex"),query().url().replace("symbols=AAPL","symbols=MSFT"),
    query().url().replace("2026-10-29","2026-10-28"),query().url()+"&feed=sip"])
def test_url_cannot_change_origin_feed_or_query(url):
    with pytest.raises((ValueError,AlpacaError)): query().validate_url(url)


def test_two_secrets_only_sent_in_headers_and_redacted_from_response():
    session=Session(Response(json.dumps(dict(page([]),echo=["PRIVATE_KEY","PRIVATE_SECRET"])).encode()))
    result=client(session).page(query().url())
    assert "PRIVATE_KEY" not in json.dumps(result) and "PRIVATE_SECRET" not in json.dumps(result)
    url,opts=session.calls[0]
    assert "PRIVATE" not in url and opts["headers"]=={"APCA-API-KEY-ID":"PRIVATE_KEY","APCA-API-SECRET-KEY":"PRIVATE_SECRET"}
    assert not opts["allow_redirects"] and not session.trust_env


@pytest.mark.parametrize("status",[301,401,403,429,500])
def test_http_failure_no_retry_fallback_or_sensitive_error(status):
    session=Session(Response(b"PRIVATE_KEY PRIVATE_SECRET",status))
    with pytest.raises(AlpacaError) as exc: client(session).page(query().url())
    assert str(status) in str(exc.value) and "PRIVATE" not in str(exc.value)
    assert len(session.calls)==1


def test_recent_requests_and_exhausted_budget_never_contact_provider():
    session=Session(Response(json.dumps(page([])).encode()))
    c=client(session,max_requests=1)
    with pytest.raises(AlpacaError,match="15 minutes"):
        c.page(AlpacaQuery("trades","AAPL","2026-11-02","2026-11-02").url())
    assert not session.calls
    c.page(query().url())
    with pytest.raises(AlpacaError,match="budget"): c.page(query().url())
    assert len(session.calls)==1


def test_response_size_and_json_errors_fail_closed():
    with pytest.raises(AlpacaError,match="memory budget"):
        client(Session(Response(b"x"*100)),max_response_bytes=10).page(query().url())
    with pytest.raises(AlpacaError,match="unsuccessful"):
        client(Session(Response(b'{"code":403,"message":"PRIVATE_SECRET"}'))).page(query().url())


def test_credentials_are_a_complete_private_pair_and_never_read_taiwan_env(tmp_path,monkeypatch):
    monkeypatch.setattr("us_market.config.REPO",tmp_path)
    for name in ("ALPACA_API_KEY","ALPACA_SECRET_KEY"): monkeypatch.delenv(name,raising=False)
    (tmp_path/".env").write_text("ALPACA_API_KEY=wrong\nALPACA_SECRET_KEY=wrong\n")
    with pytest.raises(ValueError,match="both ALPACA"): alpaca_credentials()
    path=tmp_path/".env.us"
    path.write_text("ALPACA_API_KEY=file-key\nALPACA_SECRET_KEY=literal-${DB_NAME}\nDB_NAME=wrong\n")
    path.chmod(0o600)
    before=dict(os.environ)
    assert alpaca_credentials()==("file-key","literal-${DB_NAME}")
    assert dict(os.environ)==before
    path.chmod(0o644)
    with pytest.raises(ValueError,match="600"): alpaca_credentials()
    path.chmod(0o600)
    monkeypatch.setenv("ALPACA_API_KEY","process-key")
    with pytest.raises(ValueError,match="both ALPACA"): alpaca_credentials()
    monkeypatch.setenv("ALPACA_SECRET_KEY","process-secret")
    assert alpaca_credentials()==("process-key","process-secret")


def test_cli_missing_credentials_leaves_no_data_directory(tmp_path,monkeypatch,capsys):
    from us_market.cli import main
    monkeypatch.setattr("us_market.config.REPO",tmp_path)
    for name in ("ALPACA_API_KEY","ALPACA_SECRET_KEY"): monkeypatch.delenv(name,raising=False)
    root=tmp_path/"landing"
    monkeypatch.setattr("sys.argv",["us","ingest","--provider","alpaca","--dataset","trades","--ticker","AAPL",
                                   "--start","2026-10-29","--end","2026-10-29","--fetch","--root",str(root)])
    assert main()==1 and not root.exists()
    assert "ALPACA_SECRET_KEY" in capsys.readouterr().out


def test_unsupported_alpaca_dataset_does_not_fall_back():
    with pytest.raises(ValueError,match="daily/trades"): query("splits")


def test_smoke_runner_uses_one_client_then_replays_without_fetching(tmp_path,monkeypatch,capsys):
    from us_market import alpaca_smoke
    class NativePages(AlpacaFixtureClient):
        def page(self,url):
            result=super().page(url)
            del result["synthetic_fixture"]
            return result
    transport=NativePages()
    monkeypatch.setattr(alpaca_smoke,"alpaca_credentials",lambda:("key","secret"))
    monkeypatch.setattr(alpaca_smoke,"AlpacaClient",lambda *args,**kw:transport)
    monkeypatch.setattr(alpaca_smoke,"Store",lambda root:Store(root,min_free_bytes=0))
    argv=["smoke","--ticker","AAPL","--date","2026-10-29","--root",str(tmp_path/"smoke")]
    monkeypatch.setattr("sys.argv",argv+["--fetch"])
    assert alpaca_smoke.main()==0
    result=json.loads(capsys.readouterr().out)
    assert result["report"]["requests"]==2 and result["report"]["offline_replay_verified"]
    assert not result["report"]["poc_ready"]
    monkeypatch.setattr("sys.argv",argv)
    assert alpaca_smoke.main()==0
    assert json.loads(capsys.readouterr().out)["report"]["requests"]==0
    assert transport.requests==2
    monkeypatch.setattr("sys.argv",argv+["--max-requests","21"])
    assert alpaca_smoke.main()==1
    assert "20 total" in capsys.readouterr().out
