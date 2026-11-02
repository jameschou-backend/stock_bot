import ast
from datetime import datetime
from decimal import Decimal
import gzip
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from fastapi.testclient import TestClient
import pyarrow.parquet as pq
import pytest

from us_market.api import create_app
from us_market.client import MassiveClient, MassiveError, safe_url
from us_market.config import REPO, api_key
from us_market.fixtures import FixtureClient
from us_market.normalize import normalize
from us_market.pipeline import ingest
from us_market.query import Query, NY, DATASETS
from us_market.store import Store


def query(dataset="trades", **kwargs):
    return Query(dataset,"AAPL","2026-10-29","2026-10-29",**kwargs)


def trade(id="one", **kwargs):
    return dict(id=id,sip_timestamp=int(datetime(2026,10,29,10,tzinfo=NY).timestamp())*10**9,
                exchange=12,price=Decimal("100.123456789012"),size=10,**kwargs)


def payload(rows, **kwargs):
    return dict(status="OK",results=rows,**kwargs)


class Pages:
    def __init__(self, pages):
        self.pages=list(pages); self.requests=0

    def page(self, url):
        self.requests += 1
        value=self.pages.pop(0)
        if isinstance(value,Exception): raise value
        return value


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path/"us",min_free_bytes=0)


def test_resumes_verified_pages_and_is_idempotent_offline(store):
    q=query()
    next_url="https://api.massive.com/v3/trades/AAPL?cursor=two"
    first=Pages([payload([trade()],next_url=next_url),MassiveError("budget")])
    with pytest.raises(MassiveError): ingest(q,store,first)
    assert store.snapshots()==[]
    second=Pages([payload([trade("two")])])
    result,ref=ingest(q,store,second)
    assert second.requests==1 and result["rows"]==2
    assert not result["live_qualified"] and not result["calendar_verified"]
    before=store.capacity()["used_bytes"]
    assert ingest(q,store)==(result,ref)
    assert store.capacity()["used_bytes"]==before


def test_equal_price_and_timestamp_with_distinct_ids_are_preserved(store):
    result,_=ingest(query(),store,Pages([payload([trade(),trade("two",decimal_size="0.125")])]))
    table=pq.read_table(store.root/result["pages"][0]["parquet"]["path"])
    assert table.num_rows==2
    assert table["shares"].to_pylist()==[Decimal(10),Decimal("0.125")]
    assert table["price"].to_pylist()==[Decimal("100.123456789012")]*2


def test_duplicate_page_identity_is_not_silently_dropped(store):
    with pytest.raises(ValueError,match="Duplicate"):
        ingest(query(),store,Pages([payload([trade(),trade()])]))
    assert store.snapshots()==[]


def test_missing_trade_conditions_remain_unknown_not_empty(store):
    result,_=ingest(query(),store,Pages([payload([trade(),trade("two",conditions=[])])]))
    table=pq.read_table(store.root/result["pages"][0]["parquet"]["path"])
    assert table["conditions"].to_pylist()==[None,[]]


def test_corrupt_cache_fails_without_fetching(store):
    q=query();result,_=ingest(q,store,Pages([payload([trade()])]))
    (store.root/result["pages"][0]["parquet"]["path"]).write_bytes(b"bad")
    client=Pages([])
    with pytest.raises(ValueError,match="checksum"):
        ingest(q,store,client)
    assert client.requests==0


def test_page_budget_never_publishes_partial_snapshot(store):
    with pytest.raises(MassiveError,match="Page budget"):
        ingest(query(),store,Pages([payload([trade()],next_url="https://api.massive.com/v3/trades/AAPL?cursor=two")]),max_pages=1)
    assert store.snapshots()==[]


def test_empty_success_is_not_a_certified_zero_trade_day(store):
    result,_=ingest(query(),store,Pages([payload([])]))
    assert result["rows"]==0 and result["data_status"]=="provider_landing_only"
    assert not result["strategy_ready"]


@pytest.mark.parametrize("url",["https://evil.test/v3/trades/AAPL","http://api.massive.com/v3/trades/AAPL",
                                "https://api.massive.com:443/v3/trades/AAPL","https://x@api.massive.com/v3/trades/AAPL"])
def test_untrusted_pagination_never_receives_credentials(url):
    with pytest.raises(MassiveError): safe_url(url)


def test_pagination_endpoint_must_not_change(store):
    with pytest.raises(MassiveError):
        ingest(query(),store,Pages([payload([trade()],next_url="https://api.massive.com/v3/trades/MSFT?cursor=x")]))
    assert store.snapshots()==[]


def test_us_root_refuses_taiwan_paths_and_symlinks(tmp_path):
    for root in (REPO,REPO/"artifacts",REPO/".cache",REPO/"storage",REPO.parent):
        with pytest.raises(ValueError): Store(root,min_free_bytes=0)
    data=tmp_path/"data";data.mkdir();(data/"old").write_text("TW")
    with pytest.raises(ValueError,match="unmarked"): Store(data,min_free_bytes=0)
    link=tmp_path/"link";link.symlink_to(data,target_is_directory=True)
    with pytest.raises(ValueError,match="symbolic"): Store(link,min_free_bytes=0)
    assert (data/"old").read_text()=="TW"


def test_storage_budget_never_deletes_old_files(tmp_path):
    store=Store(tmp_path/"us",max_bytes=100,min_free_bytes=0)
    with pytest.raises(ValueError,match="quota"):
        ingest(query(),store,Pages([payload([trade()])]))
    assert (store.root/"owner.json").exists()
    assert store.snapshots()==[]


def test_raw_daily_policy_and_ohlc_are_strict():
    q=query("daily")
    row=dict(t=int(datetime(2026,10,29,tzinfo=NY).timestamp())*1000,o=100,h=103,l=99,c=102,v=100)
    with pytest.raises(ValueError,match="unadjusted"): normalize(q,payload([row],adjusted=True))
    with pytest.raises(ValueError,match="OHLC"): normalize(q,payload([dict(row,h=90)],adjusted=False))


def test_trade_range_uses_new_york_calendar_day_across_dst():
    q=Query("trades","BRK.B","2026-11-01","2026-11-01")
    params=parse_qs(urlsplit(q.url()).query)
    assert int(params["timestamp.lt"][0])-int(params["timestamp.gte"][0])==25*3600*10**9
    with pytest.raises(ValueError): Query("trades","2330","2026-10-29","2026-10-29")
    with pytest.raises(ValueError): Query("trades","AAPL","2026-10-28","2026-10-29")


def test_synthetic_fixtures_are_explicit_and_separate_from_provider(store):
    for dataset in DATASETS:
        q=query(dataset,source_kind="synthetic_fixture",revision="fixture-v1")
        result,_=ingest(q,store,FixtureClient())
        assert result["data_status"]=="synthetic_fixture"
        assert q.key!=query(dataset).key
    with pytest.raises(ValueError,match="provenance"):
        normalize(query(),FixtureClient().page(query().url()))


def test_independent_api_never_claims_strategy_or_model_ready(store):
    ingest(query(source_kind="synthetic_fixture"),store,FixtureClient())
    client=TestClient(create_app(store))
    assert client.get("/health").json()["market"]=="US"
    assert client.get("/picks").json()["items"]==[]
    assert client.get("/models").json()["items"]==[]
    assert client.get("/jobs?limit=10").json()[0]["status"]=="synthetic_fixture"


class Response:
    def __init__(self,body,status=200): self.body=body;self.status_code=status
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def iter_content(self,size): yield self.body


class Session:
    def __init__(self,response): self.response=response;self.calls=[]
    def get(self,url,**kwargs): self.calls.append((url,kwargs));return self.response


def test_secret_stays_in_authorization_header_and_out_of_payload():
    secret="TEST_PRIVATE_TOKEN"
    body=json.dumps(payload([],next_url="https://api.massive.com/v3/trades/AAPL?apiKey="+secret+"&cursor=two",apiKey=secret,message=secret)).encode()
    session=Session(Response(body))
    client=MassiveClient(secret,session=session)
    result=client.page(query().url())
    assert secret not in json.dumps(result)
    url,options=session.calls[0]
    assert secret not in url and options["headers"]["Authorization"]=="Bearer "+secret
    assert options["allow_redirects"] is False


@pytest.mark.parametrize("status",[301,401,403,429,500])
def test_http_errors_do_not_leak_provider_body_or_retry(status):
    session=Session(Response(b"VERY_SECRET_PROVIDER_ERROR",status))
    client=MassiveClient("KEY",session=session)
    with pytest.raises(MassiveError) as err: client.page(query().url())
    assert "VERY_SECRET" not in str(err.value) and len(session.calls)==1


def test_response_budget_and_unknown_status_fail_closed():
    with pytest.raises(MassiveError,match="budget"):
        MassiveClient("KEY",session=Session(Response(b"x"*100)),max_response_bytes=10).page(query().url())
    with pytest.raises(MassiveError,match="status"):
        MassiveClient("KEY",session=Session(Response(b'{"status":"ERROR","results":[]}'))).page(query().url())


def test_api_key_is_never_loaded_from_taiwan_env(monkeypatch,tmp_path):
    monkeypatch.setattr("us_market.config.REPO",tmp_path/"repo")
    (tmp_path/"repo").mkdir()
    (tmp_path/"repo/.env").write_text("MASSIVE_API_KEY=taiwan-must-not-be-read\n")
    monkeypatch.delenv("MASSIVE_API_KEY",raising=False)
    with pytest.raises(ValueError,match="Set MASSIVE_API_KEY"): api_key()
    key=tmp_path/"key";key.write_text("example_private_key");key.chmod(0o600)
    assert api_key(key)=="example_private_key"
    key.chmod(0o644)
    with pytest.raises(ValueError,match="600"): api_key(key)


def test_us_dotenv_uses_repo_path_without_environment_mutation_or_interpolation(monkeypatch,tmp_path):
    import os
    repo=tmp_path/"repo";repo.mkdir()
    path=repo/".env.us"
    path.write_text('MASSIVE_API_KEY="literal-${DB_NAME}"\nDB_NAME=must-not-replace-taiwan\n')
    path.chmod(0o600)
    monkeypatch.setattr("us_market.config.REPO",repo)
    monkeypatch.delenv("MASSIVE_API_KEY",raising=False)
    monkeypatch.setenv("DB_NAME","stock_bot")
    # A different working directory must not choose a different .env.us.
    monkeypatch.chdir(tmp_path)
    (tmp_path/".env.us").write_text("MASSIVE_API_KEY=wrong-directory\n")
    before=dict(os.environ)
    assert api_key()=="literal-${DB_NAME}"
    assert dict(os.environ)==before


def test_us_key_source_precedence_is_explicit(monkeypatch,tmp_path):
    repo=tmp_path/"repo";repo.mkdir()
    path=repo/".env.us";path.write_text("MASSIVE_API_KEY=dotenv-key\n");path.chmod(0o600)
    key=tmp_path/"key";key.write_text("explicit-file-key");key.chmod(0o600)
    monkeypatch.setattr("us_market.config.REPO",repo)
    monkeypatch.delenv("MASSIVE_API_KEY",raising=False)
    assert api_key()=="dotenv-key"
    monkeypatch.setenv("MASSIVE_API_KEY","process-key")
    assert api_key()=="process-key"
    assert api_key(key)=="explicit-file-key"
    # An explicitly empty process setting is an error, not a different key.
    monkeypatch.setenv("MASSIVE_API_KEY","")
    with pytest.raises(ValueError,match="Set MASSIVE_API_KEY"): api_key()


@pytest.mark.parametrize("content",["", "MASSIVE_API_KEY=\n", 'MASSIVE_API_KEY="private invalid key"\n'])
def test_invalid_us_dotenv_does_not_expose_key(monkeypatch,tmp_path,content):
    path=tmp_path/".env.us";path.write_text(content);path.chmod(0o600)
    monkeypatch.setattr("us_market.config.REPO",tmp_path)
    monkeypatch.delenv("MASSIVE_API_KEY",raising=False)
    with pytest.raises(ValueError,match="Set MASSIVE_API_KEY") as error: api_key()
    assert "private invalid key" not in str(error.value)


def test_us_dotenv_rejects_public_permissions_and_symlinks(monkeypatch,tmp_path):
    path=tmp_path/".env.us";path.write_text("MASSIVE_API_KEY=example-key\n");path.chmod(0o644)
    monkeypatch.setattr("us_market.config.REPO",tmp_path)
    monkeypatch.delenv("MASSIVE_API_KEY",raising=False)
    with pytest.raises(ValueError,match="600"): api_key()
    path.unlink();path.symlink_to(tmp_path/".env")
    with pytest.raises(ValueError,match="symbolic"): api_key()


def test_us_modules_have_no_taiwan_runtime_dependency():
    for path in (REPO/"us_market").glob("*.py"):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module:
                assert not node.module.startswith(("app.","skills.","pipelines.")),path
            if isinstance(node,ast.Import):
                assert all(not alias.name.startswith(("app.","skills.","pipelines.")) for alias in node.names),path


def test_us_mysql_configuration_rejects_taiwan_before_connection(tmp_path):
    from us_market.catalog import engine_from_file
    path=tmp_path/"db.json"
    path.write_text(json.dumps(dict(host="127.0.0.1",port=3307,database="stock_bot",user="us_market",password="fake")))
    path.chmod(0o600)
    with pytest.raises(ValueError,match="dedicated"): engine_from_file(path)
