(() => {
  'use strict';
  const $ = id => document.getElementById(id), ns = 'http://www.w3.org/2000/svg';
  const colors = {up:'#cd6256', down:'#168c78', holder:'#9b7139', grid:'#e1e8e2', ink:'#526e63'};
  const number = new Intl.NumberFormat('zh-TW',{maximumFractionDigits:2});
  const pct = value => Number.isFinite(value) ? `${(value*100).toFixed(2)}%` : '未知';
  const delta = value => Number.isFinite(value) ? `${value>0?'+':''}${(value*100).toFixed(2)} 百分點` : '未知';
  const label = (tag,value,className='') => {const e=document.createElement(tag); e.textContent=value; e.className=className; return e;};
  const svgElement = (tag,attrs={},value) => {
    const e=document.createElementNS(ns,tag);
    for(const [key,v] of Object.entries(attrs)) e.setAttribute(key,String(v));
    if(value!==undefined) e.textContent=value;
    return e;
  };
  let source=null, selected=null, request=0, controller=null, bound=false, active=-1, cursor=null, geometry=null;
  function clear() {
    request++; controller?.abort(); source=null; selected=null; active=-1; geometry=null;
    $('chart-content').hidden=true; $('chart-loading').hidden=true; $('chart-error').hidden=true; $('chart-empty').hidden=false;
    $('stock-chart-plot').replaceChildren(); $('chart-readout').replaceChildren();
    $('chart-title').textContent='個股 K 線與籌碼';
    $('chart-subtitle').textContent='從觀察名單點選股票，或點選上方光聖、旺矽案例。';
  }
  function inspect(index) {
    if(!source || !geometry) return;
    active=Math.max(0,Math.min(index,source.candles.length-1));
    const bar=source.candles[active], holding=source.holdings[active];
    const x=geometry.x(active); cursor.setAttribute('x1',x); cursor.setAttribute('x2',x);
    const first=label('div',''); first.append(label('strong',bar.date));
    first.append(label('p',bar.valid ? `開 ${number.format(bar.open)}　高 ${number.format(bar.high)}　低 ${number.format(bar.low)}　收 ${number.format(bar.close)}` : '當日行情未知，保留交易日位置'));
    first.append(label('p',bar.valid ? `成交量 ${number.format(bar.volume/1000)} 張` : '成交量未知'));
    const second=label('div',''); second.append(label('strong',`600 張級距占比 ${pct(holding.ratio)}`));
    const state={unavailable:'尚無可用資料',stale:'週資料已過期',invalid_snapshot:'當期級距驗證未通過',known:'依假設可用時間對齊'};
    second.append(label('p',holding.observed_date ? `觀察 ${holding.observed_date} · 假設可用 ${holding.available_date}` : '觀察日期與可用日期未知'));
    second.append(label('p',`${state[holding.status] || '資料未知'} · 四週變動 ${delta(holding.delta4)}`));
    $('chart-readout').replaceChildren(first,second);
  }
  function draw() {
    if(!source || $('chart-content').hidden) return;
    const w=$('stock-chart-plot').clientWidth;
    if(!w) return;
    const left=12,right=66,span=w-left-right,n=source.candles.length,step=span/n;
    const x=i=>left+(i+.5)*step;
    const svg=svgElement('svg',{viewBox:`0 0 ${w} 540`,width:'100%',height:540,role:'img',
      'aria-label':`${source.name} ${source.stock_id} 還原 K 線、成交量與 600 張級距持股比例，${source.start} 至 ${source.end}`});
    const prices=source.candles.filter(b=>b.valid), ratios=source.holdings.filter(h=>Number.isFinite(h.ratio)).map(h=>h.ratio);
    const minPrice=prices.length?Math.min(...prices.map(b=>b.low)):0,maxPrice=prices.length?Math.max(...prices.map(b=>b.high)):1;
    const pricePad=(maxPrice-minPrice || Math.max(maxPrice*.1,1))*.08;
    const priceY=v=>35+(maxPrice+pricePad-v)/(maxPrice-minPrice+2*pricePad)*210;
    const minRatio=ratios.length?Math.min(...ratios):0,maxRatio=ratios.length?Math.max(...ratios):1;
    const ratioPad=Math.max((maxRatio-minRatio)*.15,.002);
    const rlo=Math.max(0,minRatio-ratioPad),rhi=Math.min(1,maxRatio+ratioPad);
    const holderY=v=>385+(rhi-v)/(rhi-rlo || .01)*105;
    const maxVolume=Math.max(1,...prices.map(b=>b.volume));
    const line=(x1,y1,x2,y2,stroke,width=1)=>svgElement('line',{x1,y1,x2,y2,stroke,'stroke-width':width});
    const caption=(xx,yy,value,anchor='start')=>svgElement('text',{x:xx,y:yy,fill:colors.ink,'font-size':11,'text-anchor':anchor},value);
    svg.append(caption(left,16,'還原 K 線'),caption(left,278,'成交量（張）'),caption(left,366,'600 張級距持股比例（%）'));
    for(let i=0;i<=4;i++) {
      const v=minPrice-pricePad+(maxPrice-minPrice+2*pricePad)*i/4,y=priceY(v);
      svg.append(line(left,y,w-right,y,colors.grid),caption(w-right+9,y+4,number.format(v)));
    }
    for(let i=0;i<=2;i++) {
      const v=rlo+(rhi-rlo)*i/2,y=holderY(v);
      svg.append(line(left,y,w-right,y,colors.grid),caption(w-right+9,y+4,(v*100).toFixed(2)));
    }
    svg.append(caption(w-right+9,295,number.format(maxVolume/1000)));
    const bodyWidth=Math.max(.6,Math.min(8,step*.66));
    source.candles.forEach((bar,i)=>{
      if(!bar.valid) {svg.append(line(x(i),238,x(i),245,'#c5b49b'));return;}
      const color=bar.close>=bar.open?colors.up:colors.down;
      svg.append(line(x(i),priceY(bar.high),x(i),priceY(bar.low),color));
      svg.append(svgElement('rect',{x:x(i)-bodyWidth/2,y:Math.min(priceY(bar.open),priceY(bar.close)),width:bodyWidth,
        height:Math.max(1,Math.abs(priceY(bar.open)-priceY(bar.close))),fill:color}));
      const vh=bar.volume/maxVolume*46;
      svg.append(svgElement('rect',{x:x(i)-bodyWidth/2,y:334-vh,width:bodyWidth,height:vh,fill:color,opacity:.45}));
    });
    let path='',previous=null;
    source.holdings.forEach((holding,i)=>{
      if(!Number.isFinite(holding.ratio)) {previous=null;return;}
      const xx=x(i),yy=holderY(holding.ratio);
      path+=previous ? ` H${xx} V${yy}` : ` M${xx},${yy}`;
      if(!previous || previous.observed_date!==holding.observed_date) svg.append(svgElement('circle',{cx:xx,cy:yy,r:2.5,fill:colors.holder}));
      previous=holding;
    });
    svg.append(svgElement('path',{d:path,fill:'none',stroke:colors.holder,'stroke-width':2}));
    if(!prices.length) svg.append(caption(left+12,135,'此區間沒有有效行情'));
    if(!ratios.length) svg.append(caption(left+12,435,'此區間沒有可用持股資料'));
    const ticks=Math.max(2,Math.floor(span/105));
    for(let j=0;j<ticks;j++) {
      const i=Math.round(j*(n-1)/(ticks-1));
      svg.append(caption(x(i),518,source.candles[i].date.slice(2),j===0?'start':j===ticks-1?'end':'middle'));
    }
    cursor=line(0,26,0,496,'#7b9585'); cursor.setAttribute('stroke-dasharray','4 4'); svg.append(cursor);
    geometry={x};
    const point=event=>{
      const rect=svg.getBoundingClientRect(),xx=(event.clientX-rect.left)*w/rect.width;
      const index=Math.max(0,Math.min(n-1,Math.floor((xx-left)/step)));
      if(index!==active) inspect(index);
    };
    svg.addEventListener('pointermove',point); svg.addEventListener('click',point);
    $('stock-chart-plot').replaceChildren(svg); inspect(active<0?n-1:active);
  }
  function render() {
    const last=source.candles.at(-1),holding=source.latest_holding;
    $('chart-title').textContent=`${source.name} ${source.stock_id} · K 線與籌碼`;
    $('chart-subtitle').textContent=`${source.start} → ${source.end} · ${source.sessions} 個交易日 · 持股觀察日後 ${source.holder_lag} 個日曆日假設可用`;
    $('chart-close').textContent=last.valid?number.format(last.close):'未知';
    $('chart-ratio').textContent=pct(holding.ratio); $('chart-delta').textContent=delta(holding.delta4);
    $('chart-coverage').textContent=`行情未知 ${source.missing_price_sessions} 日；持股未知 ${source.unknown_holding_sessions} 日。最新可用持股觀察日 ${holding.observed_date || '未知'}，假設可用日 ${holding.available_date || '未知'}。`;
    $('chart-weeks').replaceChildren();
    const seen=new Set();
    for(const h of source.holdings) {
      if(!h.observed_date || seen.has(h.observed_date)) continue;
      seen.add(h.observed_date); const row=label('tr','');
      for(const v of [h.observed_date,h.available_date,pct(h.ratio),delta(h.delta4)]) row.append(label('td',v));
      $('chart-weeks').append(row);
    }
    $('chart-content').hidden=false; active=-1; draw();
  }
  async function open(stock_id,scroll=false) {
    const current=++request; controller?.abort();
    const currentController=new AbortController(); controller=currentController; selected=stock_id; source=null;
    $('chart-content').hidden=true; $('chart-empty').hidden=true; $('chart-error').hidden=true; $('chart-loading').hidden=false;
    $('stock-chart-plot').replaceChildren(); $('chart-title').textContent=`${stock_id} · 讀取 K 線與籌碼`;
    if(scroll) $('stock-chart').scrollIntoView({block:'start',behavior:'auto'});
    const timer=setTimeout(()=>currentController.abort(),30000);
    try {
      const response=await fetch(`/research-terminal/api/holder-stability/stocks/${encodeURIComponent(stock_id)}?sessions=${$('chart-range').value}&holder_lag=${$('chart-lag').value}`,{cache:'no-store',signal:currentController.signal});
      if(!response.ok) throw new Error('Chart unavailable');
      const body=await response.json(); if(current!==request) return;
      if(body.stock_id!==stock_id || body.price_basis!=='adjusted' || body.end!=='2026-10-30' || body.candles.length!==body.holdings.length) throw new Error('Unexpected chart source');
      source=body; render();
      const url=new URL(location.href);url.searchParams.set('stock',stock_id);url.hash='stock-chart';history.replaceState(null,'',url);
      if(scroll) $('stock-chart').scrollIntoView({block:'start',behavior:'auto'});
    } catch(error) {
      if(current!==request) return;
      source=null; $('chart-content').hidden=true; $('chart-error').hidden=false;
      $('chart-title').textContent=`${stock_id} · 圖表暫時無法顯示`;
    } finally {clearTimeout(timer);if(current===request) $('chart-loading').hidden=true;}
  }
  function bind() {
    if(bound) return;bound=true;
    for(const id of ['chart-range','chart-lag']) $(id).addEventListener('change',()=>{if(selected) return open(selected);});
    $('chart-retry').addEventListener('click',()=>{if(selected) return open(selected);});
    $('stock-chart-plot').addEventListener('keydown',event=>{
      if(!source || !['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
      event.preventDefault(); inspect(event.key==='Home'?0:event.key==='End'?source.candles.length-1:active+(event.key==='ArrowLeft'?-1:1));
    });
    new ResizeObserver(draw).observe($('stock-chart-plot'));
  }
  globalThis.HolderChart={bind,open,clear};
})();
