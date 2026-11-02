"""Causal chart alignment, frozen membership and client selection regression tests."""
import copy
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from app import holder_stability_charts as charts
from app import holder_stability_web as web
from tests.test_holder_stability_web import client


@pytest.mark.parametrize('sessions', [60, 120, 240])
@pytest.mark.parametrize('lag', [8, 15])
def test_requested_calendar_and_latest_holder_match_the_frozen_study(sessions, lag):
    result = charts.chart_view('6442', sessions, lag)
    expected = next(r for r in web.read_report()['latest'] if r['stock_id'] == '6442' and r['holder_lag'] == lag)
    assert len(result['candles']) == len(result['holdings']) == sessions
    assert [r['date'] for r in result['candles']] == [r['date'] for r in result['holdings']]
    assert result['candles'][-1]['close'] == expected['adjusted_close']
    h = result['latest_holding']
    assert h['ratio'] == expected['large_pct'] and h['delta4'] == expected['large_pct_delta4']
    assert h['observed_date'] == expected['observed_date'] and h['available_date'] == expected['available_date']
    assert all(r['available_date'] is None or r['available_date'] <= r['date'] for r in result['holdings'])
    assert all(r['observed_date'] != '2026-10-30' for r in result['holdings'])
    assert result['price_basis'] == 'adjusted' and result['volume_unit'] == 'shares'
    assert not result['historical_publication_verified'] and not result['live_qualified']
    assert all(value not in json.dumps(result) for value in ['/Users/', '.cache/', 'source_sha256'])


def test_all_112_stocks_keep_frozen_price_and_holder_values_in_both_lags():
    members = web.study_view()['candidates']
    expected = {(r['stock_id'], r['holder_lag']): r for r in web.read_report()['latest']}
    for member in members:
        for lag in (8, 15):
            result = charts.chart_view(member['stock_id'], 240, lag)
            row = expected[(member['stock_id'], lag)]
            assert result['candles'][-1]['close'] == row['adjusted_close']
            # A valid level can be plotted even when a prior week needed for delta4 is unknown.
            assert result['latest_holding']['ratio'] == (row['large_pct'] if row['valid'] else None)
            assert result['latest_holding']['delta4'] == row['large_pct_delta4']
            assert result['latest_holding']['observed_date'] == row['observed_date']
            assert len(result['candles']) == 240
            assert all(r['available_date'] is None or r['available_date'] <= r['date'] for r in result['holdings'])


def observation(day, ratio=.5, valid=True):
    return dict(observed_date=day, ratio=ratio if valid else None, delta4=.01 if valid else None, valid=valid)


def test_alignment_has_no_future_leak_and_retains_stale_invalid_and_missing_days():
    candles = [dict(date=d, valid=False) for d in ['2026-10-08','2026-10-09','2026-10-12','2026-10-16','2026-10-30','2026-11-02']]
    # Availability on a weekend becomes visible at the next represented trading day.
    weeks = [observation('2026-10-01'), observation('2026-10-04', .6), observation('2026-10-08', valid=False)]
    actual = charts.align_holdings(candles, weeks, 8)
    assert actual[0]['status'] == 'unavailable'
    assert actual[1]['ratio'] == .5 and actual[1]['available_date'] == '2026-10-09'
    assert actual[2]['ratio'] == .6 and actual[2]['available_date'] == '2026-10-12'
    assert actual[3]['ratio'] is None and actual[3]['status'] == 'invalid_snapshot'
    assert actual[4]['status'] == 'invalid_snapshot'  # Exactly 14 days old, still not usable.
    assert actual[5]['status'] == 'stale'  # Do not backfill an older valid snapshot.
    future = weeks + [observation('2026-10-30', .99)]
    assert charts.align_holdings(candles, future, 8) == actual
    assert len(actual) == len(candles)  # Missing prices do not compress the holding timeline.
    with pytest.raises(ValueError, match='Duplicate'):
        charts.align_holdings(candles, weeks + [weeks[0]], 8)


def test_chart_endpoints_validate_membership_options_and_local_read_only_access():
    base = '/research-terminal/api/holder-stability/stocks/'
    with client() as c:
        assert c.get(base+'6223?sessions=60&holder_lag=15').status_code == 200
        assert c.get(base+'9999').status_code == 404
        for path in ['0050','6442x','6442?sessions=61','6442?holder_lag=9','6442?sessions=0','6442?sessions=x']:
            assert c.get(base+path).status_code == 422
        assert c.post(base+'6442').status_code == 405
    with client(local=False) as c:
        assert c.get(base+'6442').status_code == 403
        assert c.get('/terminal/holder-stability/charts.js').status_code == 403


def test_tampered_or_missing_chart_after_cache_warmup_fails_closed(tmp_path, monkeypatch):
    path = tmp_path / 'charts.json.gz'
    path.write_bytes(charts.CHARTS.read_bytes())
    monkeypatch.setattr(charts, 'CHARTS', path)
    with client() as c:
        url = '/research-terminal/api/holder-stability/stocks/6442'
        assert c.get(url).status_code == 200
        path.write_bytes(path.read_bytes() + b'changed')
        response = c.get(url)
        assert response.status_code == 503 and str(tmp_path) not in response.text
        path.unlink()
        assert c.get(url).status_code == 503


HARNESS = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element {
 constructor(tag='div'){this.tagName=tag;this.children=[];this._text='';this.listeners={};this.attrs={};this.value='';this.hidden=false;this.clientWidth=620;}
 get textContent(){return this._text+this.children.map(x=>x.textContent).join('');}
 set textContent(v){this._text=String(v);this.children=[];}
 get innerHTML(){throw Error('HTML injection');} set innerHTML(v){throw Error('HTML injection');}
 append(...rows){this.children.push(...rows);} replaceChildren(...rows){this._text='';this.children=[...rows];}
 setAttribute(k,v){this.attrs[k]=String(v);} addEventListener(k,fn){this.listeners[k]=fn;}
 fire(k,props={}){return this.listeners[k]?.({target:this,preventDefault(){},...props});}
 getBoundingClientRect(){return {left:0,width:620};} scrollIntoView(){this.scrolled=true;}
}
const map=new Map(),element=id=>{if(!map.has(id))map.set(id,new Element());return map.get(id);};
const urls=[],timers=new Map();let timerId=0;
const context={document:{getElementById:element,createElement:tag=>new Element(tag),createElementNS:(ns,tag)=>new Element(tag)},
 Intl,AbortController,URL,location:{href:'http://localhost/terminal/holder-stability/'},history:{replaceState:(a,b,url)=>{context.location.href=String(url);}},
 setTimeout:fn=>{timers.set(++timerId,fn);return timerId;},clearTimeout:id=>timers.delete(id),ResizeObserver:class{observe(){}},
 fetch:async url=>{urls.push(url);return {ok:true,json:async()=>input.payloads[url.match(/stocks\/(\d+)/)[1]]};}};
vm.runInNewContext(input.script,context);const ui=context.HolderChart;
element('chart-range').value='120';element('chart-lag').value='8';ui.bind();
'''


def run_js(body, payloads=None):
    node = shutil.which('node')
    assert node, 'Install Node.js to validate chart interactions'
    source = Path(__file__).resolve().parents[1] / 'ui/research_terminal/holder-stability/charts.js'
    result = subprocess.run([node, '-e', HARNESS + '\n' + body], input=json.dumps(dict(
        script=source.read_text(), payloads=payloads or {sid: charts.chart_view(sid) for sid in ['6442','6223']})),
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_chart_controls_cursor_selection_and_untrusted_text():
    run_js(r'''
(async()=>{
 input.payloads['6442'].name='<img src=x onerror=alert(1)>';
 await ui.open('6442',true);assert.match(element('chart-title').textContent,/<img src=x onerror=alert\(1\)>/);
 assert.equal(element('chart-content').hidden,false);assert.equal(element('stock-chart').scrolled,true);
 assert.match(context.location.href,/stock=6442#stock-chart/);
 assert.match(element('chart-readout').textContent,/2026-10-30.*2026-10-22.*2026-10-30/);
 assert.equal(element('stock-chart-plot').children[0].tagName,'svg');
 element('stock-chart-plot').fire('keydown',{key:'Home'});
 assert.match(element('chart-readout').textContent,new RegExp(input.payloads['6442'].start));
 element('stock-chart-plot').fire('keydown',{key:'End'});assert.match(element('chart-readout').textContent,/2026-10-30/);
 element('stock-chart-plot').children[0].fire('pointermove',{clientX:12});
 assert.match(element('chart-readout').textContent,new RegExp(input.payloads['6442'].start));
 element('chart-range').value='60';element('chart-lag').value='15';await element('chart-lag').fire('change');
 assert.match(urls.at(-1),/stocks\/6442\?sessions=60&holder_lag=15/);
 await ui.open('6223');assert.match(element('chart-title').textContent,/旺矽 6223/);
 ui.clear();assert.equal(element('chart-content').hidden,true);assert.equal(element('stock-chart-plot').children.length,0);
 assert.equal(timers.size,0);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_late_response_and_old_timeout_never_replace_or_abort_new_selection():
    run_js(r'''
(async()=>{
 const pending=[];context.fetch=(url,opts)=>new Promise(resolve=>pending.push({url,opts,resolve}));
 const first=ui.open('6442');const oldTimeout=[...timers.values()][0];const second=ui.open('6223');
 assert.equal(pending[0].opts.signal.aborted,true);oldTimeout();assert.equal(pending[1].opts.signal.aborted,false);
 pending[1].resolve({ok:true,json:async()=>input.payloads['6223']});await second;
 pending[0].resolve({ok:true,json:async()=>input.payloads['6442']});await first;
 assert.match(element('chart-title').textContent,/旺矽 6223/);assert.match(context.location.href,/stock=6223/);
 context.fetch=async()=>({ok:false});await ui.open('6442');
 assert.equal(element('chart-content').hidden,true);assert.equal(element('chart-error').hidden,false);
 assert.equal(element('stock-chart-plot').children.length,0);assert.match(element('chart-title').textContent,/6442/);
 context.fetch=async()=>({ok:true,json:async()=>input.payloads['6442']});await element('chart-retry').fire('click');
 assert.equal(element('chart-content').hidden,false);assert.equal(element('chart-error').hidden,true);
 assert.equal(timers.size,0);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_holding_path_breaks_at_unknown_weeks_and_missing_candles_remain_visible():
    data = copy.deepcopy(charts.chart_view('6442', 60))
    data['holdings'][30].update(ratio=None, status='invalid_snapshot')
    data['candles'][30].update(valid=False, open=None, high=None, low=None, close=None, volume=None)
    run_js(r'''
(async()=>{
 await ui.open('6442');const svg=element('stock-chart-plot').children[0];
 const path=svg.children.find(e=>e.tagName==='path').attrs.d;assert.equal((path.match(/ M/g)||[]).length,2);
 element('stock-chart-plot').fire('keydown',{key:'Home'});
 for(let i=0;i<30;i++)element('stock-chart-plot').fire('keydown',{key:'ArrowRight'});
 assert.match(element('chart-readout').textContent,/當日行情未知.*持股|當日行情未知.*600 張級距占比 未知/);
 assert.match(element('chart-readout').textContent,/當期級距驗證未通過/);
})().catch(e=>{console.error(e);process.exitCode=1;});
''', {'6442': data})
