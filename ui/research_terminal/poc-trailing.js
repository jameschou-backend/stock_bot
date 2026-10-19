'use strict';
window.renderPocTrailing = function(item, host, h) {
  const {el,number,pct,api,renderAccountDetail}=h;
  host.replaceChildren(el('div','TRAILING EXIT · 20% / 12%','eyebrow'),el('h2','POC硬篩＋紅K：加上移動停利'));
  if(item?.status!=='verified_publication'||!item.trailing){
    host.append(el('p','四種成本情境全部核對完成後，在此公布與原版、同成本0050的比較。','alert'));return;
  }
  const r=item.trailing, modes={baseline:'原成本',fee2:'佣金加倍',slip2:'滑價加倍',combined:'佣金與滑價加倍'};
  const variants=[['poc_up_red:trail20_12','移動停利20/12'],['poc_up_red:original','POC硬篩＋紅K原版'],['benchmark:original','同成本0050']];
  const pp=x=>(x>0?'+':'')+number(x*100,2);
  function table(parent,heads,rows){
    const wrap=el('div',undefined,'compact-comparison'),t=el('table'),head=el('thead'),tr=el('tr'),body=el('tbody');
    heads.forEach(x=>tr.append(el('th',x)));head.append(tr);
    rows.forEach(cells=>{const row=el('tr');cells.forEach((x,i)=>{const td=el('td',String(x));td.dataset.label=heads[i];row.append(td);});body.append(row);});
    t.append(head,body);wrap.append(t);parent.append(wrap);
  }
  host.append(el('p',`${r.start} → ${r.end} · 本金100萬元 · 3個活躍名額 · 4/4新帳戶完成 · 重用8個原版與0050對照`,'caption'),
    el('p','相對實際進場日的還原收盤價上漲20%後啟動；從持有期最高還原收盤回落12%時，次一交易日起嘗試賣出，未成交部分保留賣出指令。原本12%停損、三黑K、63日期限優先保留，其餘條件不變。','result-note'));
  const mode=el('select');mode.setAttribute('aria-label','移動停利成本情境');
  Object.entries(modes).forEach(([key,name])=>{const o=el('option',name);o.value=key;mode.append(o);});
  const result=el('section'),delta=el('section'),stress=el('details',undefined,'comparison-method'),annual=el('details',undefined,'comparison-method'),actions=el('div',undefined,'study-actions'),detail=el('div',undefined,'comparison-detail');
  host.append(mode,result,delta,stress,annual,actions,detail);let generation=0;
  function render(){
    ++generation;[result,delta,stress,annual,actions,detail].forEach(x=>x.replaceChildren());
    const m=mode.value,newCase=r.cases[`poc_up_red:trail20_12:${m}`];
    result.append(el('h3',`${modes[m]}：移動停利淨報酬 ${pct(newCase.metrics.total_return)}，最大回撤 ${pct(newCase.metrics.max_drawdown)}`));
    table(result,['策略','累積淨報酬','最大回撤','期末資產','總費稅與滑價'],variants.map(([key,name])=>{const x=r.cases[`${key}:${m}`].metrics;return[name,pct(x.total_return),pct(x.max_drawdown),number(x.final_nav,2),number(x.costs.total_cost,2)];}));
    table(delta,['比較對象','回撤改善百分點','報酬變化百分點','淨利保留比例','損益差額'],[['original','原版'],['benchmark','同成本0050']].map(([key,name])=>{const x=r.comparisons[m][key];return[name,pp(x.drawdown_improvement),pp(x.return_change),x.net_profit_retained==null?'—':pct(x.net_profit_retained),number(x.profit_change,2)+' 元'];}));
    delta.append(el('p','回撤是整個帳戶淨值從高點下跌的幅度。改善為正代表回撤減少；報酬變化為負代表少賺。淨利保留比例不含本金；報酬為整段累積值，非年化。','caption'),
      el('p',`新增停利觸發 ${newCase.trailing_audit.triggers} 次；成交資料品質排除 ${newCase.data_gap_exclusions} 筆、POC資料排除 ${newCase.profile_data_exclusions} 筆。觸發前的帳戶與原版一致。`,'caption'));
    stress.append(el('summary','四種成本情境完整比較'));stress.open=true;
    table(stress,['成本','停利報酬／回撤','原版報酬／回撤','0050報酬／回撤','相對原版回撤改善','相對原版報酬變化'],Object.entries(modes).map(([key,name])=>[name,...variants.map(([v])=>{const x=r.cases[`${v}:${key}`].metrics;return pct(x.total_return)+'／'+pct(x.max_drawdown);}),pp(r.comparisons[key].original.drawdown_improvement),pp(r.comparisons[key].original.return_change)]));
    annual.append(el('summary','逐年報酬與回撤'));
    table(annual,['年度',...variants.map(x=>x[1]+'報酬／回撤')],newCase.metrics.annual.map((x,i)=>[x.year+(x.partial_year?'（至10/2）':''),...variants.map(([v])=>{const y=r.cases[`${v}:${m}`].metrics.annual[i];return pct(y.total_return)+'／'+pct(y.max_drawdown);})]));
    for(const [key,name] of variants){
      const button=el('button',name+'曲線','primary-button');
      button.onclick=async()=>{const request=++generation,selected=mode.value,prefix=`/archive/${item.id}/cases/`;button.disabled=true;detail.replaceChildren(el('p','正在核對帳戶…'));
        try{const [x,b]=await Promise.all([api(prefix+key.replaceAll(':','__')+'__'+selected),key.startsWith('benchmark')?Promise.resolve(null):api(prefix+'benchmark__original__'+selected)]);if(request!==generation)return;detail.replaceChildren();renderAccountDetail(x,detail,b);}
        catch(e){if(request===generation)detail.replaceChildren(el('p',e.message,'alert error'));}finally{button.disabled=false;}};
      actions.append(button);
    }
  }
  mode.addEventListener('change',render);render();
  host.append(el('p','這是已見歷史的研究，買HL2／賣30%日區間價仍屬成交估算，未證明可實際成交。結果不會自動取代原版訊號；未建立委託或成交。','result-note'));
  const link=el('a','查看完整結果與來源紀錄 ↗');link.href=item.download_url;link.target='_blank';link.rel='noopener';host.append(link);
};
