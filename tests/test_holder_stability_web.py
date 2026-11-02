"""Frozen membership, safe publication and actual UI interaction contracts."""
import json
from pathlib import Path
import shutil
import subprocess

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import holder_stability_web as web
from app import research_terminal_web
from app.workbench_api import require_local


def client(local=True):
    app = FastAPI()
    app.include_router(web.router)
    app.include_router(research_terminal_web.router)
    if local:
        app.dependency_overrides[require_local] = lambda: None
    return TestClient(app)


def test_view_preserves_frozen_membership_statistics_and_qualification():
    original = json.loads(web.REPORT.read_text())
    actual = web.study_view()
    expected = {row['stock_id'] for row in original['latest'] if row['holder_lag'] == 8 and row['stable']}
    assert {row['stock_id'] for row in actual['candidates']} == expected
    assert len(actual['candidates']) == 112 and actual['confirmed_count'] == 0
    assert actual['statistics'] == original['statistics']
    assert actual['paired_statistics'] == original['paired_statistics']
    assert not actual['live_qualified'] and not actual['adopted']
    assert actual['latest_counts']['15']['stable'] == 126  # Must not replace the 112-stock pool.
    assert {row['stock_id'] for row in actual['candidates'] if row['named_case']} == {'6442','6223'}
    assert {row['observed_date'] for row in actual['candidates']} == {'2026-10-22'}
    assert {row['delayed_observed_date'] for row in actual['candidates']} == {'2026-10-08'}
    encoded = json.dumps(actual)
    assert all(secret not in encoded for secret in ['/Users/', '.cache/', 'source_sha256', 'research_outputs'])


def test_local_routes_are_read_only_and_assets_use_existing_security_headers():
    with client() as c:
        assert c.get('/research-terminal/api/holder-stability').status_code == 200
        assert c.post('/research-terminal/api/holder-stability').status_code == 405
        redirect = c.get('/terminal/holder-stability', follow_redirects=False)
        assert redirect.status_code == 307 and redirect.headers['location'] == '/terminal/holder-stability/'
        page = c.get('/terminal/holder-stability/')
        assert page.status_code == 200 and '112 檔觀察名單' in page.text
        assert "script-src 'self'" in page.headers['content-security-policy']
        assert page.headers['cache-control'] == 'no-store'
        for name in ['app.css','app.js']:
            assert c.get('/terminal/holder-stability/'+name).status_code == 200
        assert c.get('/terminal/holder-stability/.env').status_code == 404
        assert c.get('/terminal/holder-stability/private/report.json').status_code == 404
        assert 'href="/terminal/holder-stability/"' in c.get('/terminal/').text
    with client(local=False) as c:
        for path in ['/terminal/holder-stability/','/terminal/holder-stability/app.js','/research-terminal/api/holder-stability']:
            assert c.get(path).status_code == 403


def test_changed_or_missing_report_is_explicitly_unavailable(tmp_path, monkeypatch):
    report = tmp_path / 'study.json'
    report.write_bytes(web.REPORT.read_bytes() + b' ')
    monkeypatch.setattr(web, 'REPORT', report)
    with client() as c:
        response = c.get('/research-terminal/api/holder-stability')
        assert response.status_code == 503 and '核對' in response.json()['detail']
        assert str(tmp_path) not in response.text
        report.unlink()
        assert c.get('/research-terminal/api/holder-stability').status_code == 503


SCRIPT = Path(__file__).resolve().parents[1] / 'ui/research_terminal/holder-stability/app.js'
HARNESS = r'''
const assert=require('node:assert/strict'),vm=require('node:vm');
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
class Element {
 constructor(tag='div'){this.tagName=tag;this.children=[];this._text='';this.listeners={};this.value='';this.hidden=false;this.className='';}
 get textContent(){return this._text+this.children.map(x=>x.textContent).join('');}
 set textContent(v){this._text=String(v);this.children=[];}
 get innerHTML(){throw Error('HTML injection');} set innerHTML(v){throw Error('HTML injection');}
 get selectedOptions(){return [{textContent:this.value}];}
 append(...rows){this.children.push(...rows);}
 replaceChildren(...rows){this._text='';this.children=[...rows];}
 addEventListener(k,fn){this.listeners[k]=fn;}
 fire(k){return this.listeners[k]?.({target:this});}
}
const map=new Map(),element=id=>{if(!map.has(id))map.set(id,new Element());return map.get(id);};
const context={document:{getElementById:element,createElement:tag=>new Element(tag)},Intl,AbortController,setTimeout,clearTimeout,
 fetch:async()=>({ok:true,json:async()=>input.payload})};
vm.runInNewContext(input.script,context);
const ui=context.holderUI;
for(const [key,value] of Object.entries({period:'2025_2026',exit:'20',lag:'8',cost:'main',scope:'all',sort:'liquidity',search:''}))element(key).value=value;
ui.bind();
'''


def run_js(body):
    node = shutil.which('node')
    assert node, 'Install Node.js to validate the research page interactions'
    script = SCRIPT.read_text().replace('bind(); load();', 'globalThis.holderUI={bind,load};')
    result = subprocess.run([node, '-e', HARNESS + '\n' + body],
                            input=json.dumps(dict(script=script, payload=web.study_view())),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_watchlist_search_scopes_sort_empty_and_reset_use_the_actual_script():
    run_js(r'''
(async()=>{
 await ui.load();assert.equal(element('study').hidden,false);
 assert.equal(element('candidate-rows').children.length,112);
 element('search').value='6442';element('search').fire('input');
 assert.equal(element('candidate-rows').children.length,1);assert.match(element('candidate-rows').textContent,/光聖/);
 element('search').value='沒有這個名字';element('search').fire('input');assert.equal(element('empty').hidden,false);
 element('clear').fire('click');assert.equal(element('candidate-rows').children.length,112);
 element('scope').value='named';element('scope').fire('change');assert.equal(element('candidate-rows').children.length,2);
 element('scope').value='both';element('scope').fire('change');assert.equal(element('candidate-rows').children.length,81);
 element('scope').value='different';element('scope').fire('change');assert.equal(element('candidate-rows').children.length,31);
 element('clear').fire('click');element('sort').value='stock';element('sort').fire('change');
 assert.match(element('candidate-rows').children[0].textContent,/1590/);
 assert.equal(input.payload.candidates.length,112);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_result_controls_preserve_watchlist_and_display_correct_period_costs():
    run_js(r'''
(async()=>{
 await ui.load();assert.match(element('headline-return').textContent,/2.47%/);
 assert.match(element('headline-excess').textContent,/-0.46/);
 assert.equal(element('comparison-rows').children.length,6);
 element('lag').value='15';element('cost').value='stress';element('period').value='2022_2024';element('exit').value='60';element('exit').fire('change');
 const expected=input.payload.statistics.find(x=>x.lag===15&&x.cost==='stress'&&x.phase==='2022_2024'&&x.rule==='60'&&x.arm==='stable');
 assert.match(element('comparison-rows').children[1].textContent,new RegExp((expected.mean_net*100).toFixed(2).replace('.','\\.')));
 assert.match(element('comparison-context').textContent,/15 日.*0.9%/);
 assert.equal(element('candidate-rows').children.length,112);
 assert.match(element('coverage').textContent,/有效/);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')


def test_failed_reload_hides_stale_data_and_untrusted_names_stay_text_only():
    run_js(r'''
(async()=>{
 input.payload.candidates[0].name='<img src=x onerror=alert(1)>';
 await ui.load();assert.match(element('candidate-rows').textContent,/<img src=x onerror=alert\(1\)>/);
 context.fetch=async()=>({ok:false});await ui.load();
 assert.equal(element('study').hidden,true);assert.equal(element('error').hidden,false);assert.equal(element('loading').hidden,true);
 context.fetch=async()=>({ok:true,json:async()=>({...input.payload,candidates:[]})});await ui.load();
 assert.equal(element('study').hidden,true);assert.equal(element('error').hidden,false);
 context.fetch=async()=>({ok:true,json:async()=>input.payload});await element('retry').fire('click');
 assert.equal(element('study').hidden,false);assert.equal(element('error').hidden,true);
})().catch(e=>{console.error(e);process.exitCode=1;});
''')
