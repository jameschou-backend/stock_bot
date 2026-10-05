"""Offline scanner presentation contracts and actual JavaScript interactions."""
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from skills.strategy_scanner.presentation import render_html


TEMPLATE = Path(__file__).resolve().parents[1] / "ui/multi_strategy_scanner.html"


def fixture():
    def result(status, **extra):
        return dict(status=status, reasons=["原因：" + status], metrics={"return20": .1234},
                    first_signal=None, regime_fit=None, **extra)
    return dict(schema="multi_strategy_scan_v1", start="2026-09-30", end="2026-10-02",
        source_end="2026-10-02", live_qualified=False,
        strategies=[dict(id="momentum", name="動能", family="趨勢", kind="ranking", status="active",
                         version="v1", description="中期動能", required_data=["日線"], preferred_regimes=["trend_up"]),
                    dict(id="breakout", name="突破", family="趨勢", kind="entry", status="active",
                         version="v2", description="突破高點", required_data=["日線"], preferred_regimes=[]),
                    dict(id="news", name="題材事件", family="事件", kind="entry", status="catalog_only",
                         version="v1", description="等待事件資料", required_data=["新聞"], preferred_regimes=[])],
        days=[dict(date="2026-09-30", stocks=[], counts={}),
              dict(date="2026-10-02", market_regime="trend_up", counts={"matched": 999}, stocks=[
                  dict(stock_id="2330", name="台積電", regime="trend_up", results={"momentum": result("matched"), "breakout": result("matched")}),
                  dict(stock_id="2408", name="南亞科", regime="unknown", results={"momentum": result("unknown"), "breakout": result("not_matched")}),
                  dict(stock_id="2317", name="鴻海", regime="range", results={"momentum": result("not_matched"), "breakout": result("matched")}),
                  dict(stock_id="9999", name="範圍外", regime="unknown", results={"momentum": result("ineligible"), "breakout": result("ineligible")})])],
        provenance={"source": "封存研究資料", "history_is_reconstructed": True})


HARNESS = r"""
const assert=require('node:assert/strict'),vm=require('node:vm');
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element{
 constructor(tag='div'){this.tagName=tag;this.children=[];this._text='';this.listeners={};this.attrs={};this.value='';this.hidden=false;this.disabled=false;this.className='';}
 get textContent(){return this._text+this.children.map(x=>x.textContent).join('');}
 set textContent(v){this._text=String(v);this.children=[];}
 get innerHTML(){throw new Error('innerHTML is forbidden for scanner data');}
 set innerHTML(v){throw new Error('innerHTML is forbidden for scanner data');}
 get childNodes(){return this.children;}
 append(...rows){this.children.push(...rows);}
 replaceChildren(...rows){this._text='';this.children=[...rows];}
 setAttribute(k,v){this.attrs[k]=v;}
 addEventListener(k,fn){this.listeners[k]=fn;}
 fire(k){if(this.listeners[k])this.listeners[k]({target:this});}
}
function mount(payload=input.payload){
 const map=new Map(),element=id=>{if(!map.has(id))map.set(id,new Element());return map.get(id);};
 element('scan-data').textContent=JSON.stringify(payload);
 const context={document:{getElementById:element,createElement:tag=>new Element(tag)},console};
 vm.runInNewContext(input.script,context);
 return {ui:context.scannerUI,element,context};
}
"""


def run_js(body, payload=None):
    node = shutil.which("node")
    assert node, "Install Node.js to validate the standalone scanner UI"
    script = re.findall(r"<script>([\s\S]*?)</script>", TEMPLATE.read_text())[0]
    script = script.replace("})();", "globalThis.scannerUI={data,state,active,selectDay,setFilters,visibleRows,resultFor,totals,render};})();")
    result = subprocess.run([node, "-e", HARNESS + "\n" + body],
        input=json.dumps(dict(script=script, payload=fixture() if payload is None else payload)),
        text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_renderer_script_embedding_roundtrips_hostile_strings_without_execution():
    payload = fixture()
    malicious = '</script><script>alert("unsafe")</script><img src=x onerror=alert(1)>&\u2028\u2029'
    payload["days"][1]["stocks"][0]["name"] = malicious
    payload["provenance"] = {"unsafe": malicious}
    html = render_html(payload)
    assert html.count("<script") == 2
    assert malicious not in html
    blob = re.search(r'<script id="scan-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert "<" not in blob and ">" not in blob and "&" not in blob
    assert json.loads(blob) == payload
    assert "__SCAN_DATA__" not in html


@pytest.mark.parametrize("change", [{"schema": "other"}, {"live_qualified": True},
    {"live_qualified": None}, {"days": {}}, {"strategies": {}}, {"source_end": ""}])
def test_renderer_rejects_ambiguous_contract(change):
    payload = fixture() | change
    with pytest.raises(ValueError):
        render_html(payload)


def test_renderer_rejects_nonfinite_json_values():
    payload = fixture()
    payload["provenance"]["bad"] = float("nan")
    with pytest.raises(ValueError):
        render_html(payload)


@pytest.mark.parametrize("change", [{"source_end": "2026-09-30"},
    {"start": "2026-10-05"}, {"end": "2026-02-30"}, {"start": "20260930"}])
def test_renderer_rejects_date_scope_overclaims(change):
    with pytest.raises(ValueError):
        render_html(fixture() | change)


def test_daily_counts_are_recomputed_multihits_and_default_filter_is_matched():
    run_js(r"""
const {ui,element}=mount();assert.equal(ui.state.date,'2026-10-02');assert.equal(ui.state.status,'matched');
assert.equal(element('stat-stocks').textContent,'4');assert.equal(element('stat-matched').textContent,'2');
assert.equal(element('stat-pairs').textContent,'3');assert.equal(element('stat-unknown').textContent,'1');
assert.deepEqual(Array.from(ui.visibleRows(),x=>x.stock_id),['2330','2317']);
assert.match(element('stock-rows').textContent,/動能/);assert.match(element('stock-rows').textContent,/突破/);
assert.match(element('strategy-details').textContent,/排序/);assert.match(element('strategy-details').textContent,/僅收錄/);
assert.match(element('scope-note').textContent,/市場情境：上升趨勢/);
""")


def test_filters_unknown_ineligible_and_missing_results_do_not_become_false():
    run_js(r"""
const {ui,element}=mount();ui.setFilters({status:'unknown'});
assert.deepEqual(Array.from(ui.visibleRows(),x=>x.stock_id),['2408']);
assert.match(element('strategy-details').textContent,/資料不足/);
ui.setFilters({status:'ineligible'});assert.deepEqual(Array.from(ui.visibleRows(),x=>x.stock_id),['9999']);
delete ui.data.days[1].stocks[0].results.momentum;ui.setFilters({status:'unknown'});
assert.equal(element('stat-unknown').textContent,'2');assert.match(element('strategy-details').textContent,/缺少這套策略的評估紀錄/);
ui.data.days[1].stocks[0].results.momentum={status:'matched',reasons:[],metrics:null};ui.render();
assert.equal(element('stat-unknown').textContent,'2');assert.match(element('strategy-details').textContent,/紀錄不完整/);
""")


def test_strategy_and_search_filters_preserve_all_strategy_detail_and_global_counts():
    run_js(r"""
const {ui,element}=mount();ui.setFilters({strategy:'breakout',query:'鴻'});
assert.deepEqual(Array.from(ui.visibleRows(),x=>x.stock_id),['2317']);
assert.match(element('strategy-details').textContent,/動能/);assert.match(element('strategy-details').textContent,/突破/);
assert.equal(element('stat-stocks').textContent,'4');assert.equal(element('stat-pairs').textContent,'3');
ui.setFilters({query:'2330'});assert.deepEqual(Array.from(ui.visibleRows(),x=>x.stock_id),['2330']);
ui.setFilters({strategy:'news',query:''});assert.equal(ui.state.status,'all');assert.equal(ui.visibleRows().length,4);
assert.match(element('stock-rows').textContent,/僅收錄/);assert.equal(element('stat-unknown').textContent,'1');
""")


def test_explicit_subset_distinguishes_not_evaluated_from_unknown():
    payload = fixture()
    payload["evaluated_strategy_ids"] = ["breakout"]
    for stock in payload["days"][1]["stocks"]:
        del stock["results"]["momentum"]
    run_js(r"""
const {ui,element}=mount();assert.equal(element('stat-unknown').textContent,'0');assert.equal(element('stat-pairs').textContent,'2');
assert.match(element('strategy-details').textContent,/本次未掃描/);
ui.setFilters({strategy:'momentum'});assert.equal(ui.state.status,'all');assert.equal(ui.visibleRows().length,4);
assert.match(element('stock-rows').textContent,/本次未掃描/);
""", payload)


def test_zero_day_clears_stale_details_and_date_arrows_never_invent_dates():
    run_js(r"""
const {ui,element}=mount();assert.equal(element('next-day').disabled,true);element('previous-day').fire('click');
assert.equal(ui.state.date,'2026-09-30');assert.equal(element('stat-stocks').textContent,'0');
assert.equal(ui.state.selected,null);assert.equal(element('detail').hidden,true);assert.equal(element('strategy-details').textContent,'');
assert.match(element('list-empty').textContent,/沒有股票評估紀錄/);assert.equal(element('previous-day').disabled,true);
ui.selectDay('2026-10-05');assert.equal(ui.state.date,'2026-09-30');element('next-day').fire('click');
assert.equal(ui.state.date,'2026-10-02');assert.equal(element('stat-stocks').textContent,'4');
""")


def test_pagination_keeps_full_population_searchable_and_click_updates_detail():
    payload = fixture()
    sample = payload["days"][1]["stocks"][0]
    payload["days"][1]["stocks"] = [dict(deepcopy(sample), stock_id=str(1000+i), name="股票"+str(i)) for i in range(215)]
    run_js(r"""
const {ui,element}=mount();assert.equal(element('stat-stocks').textContent,'215');
assert.equal(ui.visibleRows().length,215);assert.equal(element('stock-rows').children.length,100);
element('next-page').fire('click');assert.equal(ui.state.page,1);assert.equal(element('stock-rows').children.length,100);
assert.match(element('detail-title').textContent,/1100/);
const row=element('stock-rows').children[0];row.children[0].children[0].fire('click');
assert.match(element('detail-title').textContent,/1100/);
element('next-page').fire('click');assert.equal(element('stock-rows').children.length,15);assert.equal(element('next-page').disabled,true);
ui.setFilters({query:'1214'});assert.equal(ui.state.page,0);assert.equal(element('stock-rows').children.length,1);
assert.match(element('detail-title').textContent,/1214/);
""", payload)


def test_untrusted_values_only_reach_text_nodes_and_payload_is_never_mutated():
    payload = fixture()
    stock = payload["days"][1]["stocks"][0]
    stock["name"] = '<img src=x onerror="bad()">'
    stock["results"]["momentum"]["reasons"] = ['<script>bad()</script>']
    stock["results"]["momentum"]["metrics"] = {"<svg onload=bad()>": '<a href="javascript:bad()">bad</a>'}
    run_js(r"""
const before=JSON.stringify(input.payload),{ui,element}=mount();
assert.match(element('detail-title').textContent,/<img src=x/);
assert.match(element('strategy-details').textContent,/<script>bad\(\)<\/script>/);
assert.equal(JSON.stringify(ui.data),before);ui.setFilters({status:'all'});ui.selectDay('2026-09-30');
assert.equal(JSON.stringify(ui.data),before);assert.equal(typeof ui.data.orders,'undefined');
""", payload)


def test_duplicate_dates_or_stocks_and_invalid_strategy_registry_fail_closed():
    run_js(r"""
for(const mutation of [p=>p.days.push(p.days[0]),p=>p.days[1].stocks.push(p.days[1].stocks[0]),
 p=>p.strategies.push(p.strategies[0]),p=>p.evaluated_strategy_ids=['absent'],p=>p.days.push(null),
 p=>p.days[1].date='2026-10-05',p=>p.source_end='2026-09-30']){
 const p=JSON.parse(JSON.stringify(input.payload));mutation(p);const x=mount(p);
 assert.equal(x.ui,undefined);assert.equal(x.element('scanner').hidden,true);assert(x.element('error').textContent.length>0);
}
""")


def test_standalone_copy_keeps_research_and_context_boundaries_clear():
    text = TEMPLATE.read_text()
    assert text.count("__SCAN_DATA__") == 1
    assert "研究掃描 · 非買單" in text
    assert "不是報酬預測或建議配置" in text
    assert "命中數不是勝率；多套策略可能使用相同價量條件" in text
    assert "沒有重建原三檔帳戶的最終排序" in text
    assert "尚未驗證為最適策略" in text
    assert "本頁沒有重新計算帳戶報酬" in text
    assert "innerHTML" not in text
    assert not re.search(r'<(?:script|link)[^>]+(?:src|href)=["\']https?://', text)


def test_actual_catalog_renders_all_families_without_claiming_live_qualification():
    from skills.strategy_scanner.catalog import get_catalog
    payload = fixture()
    payload["strategies"] = get_catalog()
    for stock in payload["days"][1]["stocks"]:
        stock["results"] = {s["id"]: dict(status="not_matched", reasons=["本次未符合"], metrics={},
            first_signal=None, regime_fit=None) for s in payload["strategies"] if s["status"] == "active"}
    run_js(r"""
const {ui,element}=mount();ui.setFilters({status:'all'});
assert.equal(ui.data.strategies.length,60);assert.equal(ui.active.length,12);
assert.equal(element('strategy-details').children.length,60);
assert.match(element('strategy-details').textContent,/沒有重建原三檔帳戶的最終排序/);
assert.match(element('strategy-details').textContent,/策略來源/);
""", payload)
