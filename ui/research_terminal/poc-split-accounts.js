'use strict';
window.renderPocSplitAccounts = function(item, host, h) {
  const {el, number, pct, api, renderAccountDetail} = h;
  host.replaceChildren(el('div', 'TWO ACCOUNTS · NO REBALANCING', 'eyebrow'), el('h2', '兩套原版各 50 萬，不再平衡'));
  if (item?.status !== 'verified_publication' || !item.split_accounts) {
    host.append(el('p', '8 個新分帳正在核對；完成後在此與原版、同成本0050比較。已完成原版不重跑。', 'alert'));
    return;
  }
  const r = item.split_accounts;
  const modes = {baseline:'原成本', fee2:'佣金加倍', slip2:'滑價加倍', combined:'佣金與滑價加倍'};
  const names = {poc_up_red:'硬篩原版', poc_priority:'優先原版', benchmark:'0050'};
  const pp = x => `${x > 0 ? '+' : ''}${number(x * 100, 2)}`;
  function table(parent, heads, rows) {
    const wrap=el('div', undefined, 'compact-comparison'), t=el('table'), head=el('thead'), tr=el('tr'), body=el('tbody');
    heads.forEach(x=>tr.append(el('th', x))); head.append(tr);
    rows.forEach(cells=>{const row=el('tr'); cells.forEach((x,i)=>{const c=el('td', String(x)); c.dataset.label=heads[i]; row.append(c);}); body.append(row);});
    t.append(head,body); wrap.append(t); parent.append(wrap);
  }
  host.append(el('p', `${r.start} → ${r.end} · 666 交易日 · 總本金 1,000,000 元 · 8/8 新分帳完成 · 重用 12 個原版與0050對照`, 'caption'),
    el('p', '硬篩原版與優先原版各投入50萬元，各自保留3個持股名額。獲利留在原帳戶，現金不互借，不轉帳、不再平衡；合併淨值等於兩帳戶每日資產總和。', 'result-note'));
  const controls=el('div', undefined, 'study-controls'), mode=el('select'); mode.setAttribute('aria-label','分帳成本情境');
  Object.entries(modes).forEach(([key,label])=>{const o=el('option',label);o.value=key;mode.append(o);}); controls.append(mode);host.append(controls);
  const result=el('section'), delta=el('section'), stress=el('details',undefined,'comparison-method'), sleeves=el('details',undefined,'comparison-method'), actions=el('div',undefined,'study-actions'), detail=el('div',undefined,'comparison-detail');
  host.append(result,delta,stress,sleeves,actions,detail); let generation=0;
  function render() {
    ++generation; [result,delta,stress,sleeves,actions,detail].forEach(x=>x.replaceChildren());
    const m=mode.value, combined=r.cases[`combined:combined:${m}`], metrics=combined.metrics;
    result.append(el('h3',`${modes[m]}：合併淨報酬 ${pct(metrics.total_return)}，最大回撤 ${pct(metrics.max_drawdown)}`));
    const entries=[[combined,'兩套各50萬'],...Object.entries(names).map(([a,n])=>[r.cases[`${a}:original:${m}`],n+'・100萬'])];
    table(result,['配置','淨報酬','最大回撤','期末資產','總費稅與滑價'],entries.map(([x,n])=>[n,pct(x.metrics.total_return),pct(x.metrics.max_drawdown),number(x.metrics.final_nav,2),number(x.metrics.costs.total_cost,2)]));
    delta.append(el('h3','分帳後改善多少、少賺多少'));
    table(delta,['相對配置','回撤改善百分點','回撤相對減少','報酬變化百分點','損益差額'],Object.entries(names).map(([a,n])=>{const d=r.comparisons[m][a];return[n,pp(d.drawdown_improvement),pct(d.drawdown_relative_reduction),pp(d.return_change),`${d.profit_change>0?'+':''}${number(d.profit_change,2)} 元`];}));
    delta.append(el('p','回撤改善為正代表回撤減少；報酬變化為負代表少賺。數字是整段期間的累積淨報酬，並非年化報酬。','caption'));
    const cap=combined.shared_capacity;
    delta.append(el('p',`兩帳戶有 ${cap.overlapping_security_channel_days} 組同日、同股、同成交管道的重疊成交；其中 ${cap.capacity_collision_count} 組合計超過單一市場容量假設。${cap.capacity_collision_count?'此合併結果保留各帳戶獨立成交估算，尚不能當作共用成交佇列可實現的績效。':'容量合計核對通過，實際成交時序仍未驗證。'}`, 'result-note'));
    stress.append(el('summary','四種成本情境，一起比較')); stress.open=true;
    table(stress,['成本','分帳報酬','分帳回撤','硬篩原版報酬／回撤','優先原版報酬／回撤','0050報酬／回撤'],Object.entries(modes).map(([c,n])=>{const x=r.cases[`combined:combined:${c}`];return[n,pct(x.metrics.total_return),pct(x.metrics.max_drawdown),...Object.keys(names).map(a=>{const y=r.cases[`${a}:original:${c}`].metrics;return `${pct(y.total_return)}／${pct(y.max_drawdown)}`;})];}));
    sleeves.append(el('summary','各50萬帳戶與年度結果'));
    table(sleeves,['分帳','期末資產','期末占比','淨報酬','最大回撤','成本'],['poc_up_red','poc_priority'].map(a=>{const x=r.cases[`${a}:split50:${m}`].metrics;return[names[a],number(x.final_nav,2),pct(x.final_nav/metrics.final_nav),pct(x.total_return),pct(x.max_drawdown),number(x.costs.total_cost,2)];}));
    sleeves.append(el('p',['poc_up_red','poc_priority'].map(a=>{const x=r.cases[`${a}:split50:${m}`],p=x.funded_path_comparison;return `${names[a]}：成交資料品質排除 ${x.data_gap_exclusions} 筆、POC資料排除 ${x.profile_data_exclusions} 筆${p?`；與100萬原版共同進場 ${p.common_cohorts} 批，分帳另有 ${p.only_sleeve} 批不同進場`:''}`;}).join('。')+'。股數、整股與零股容量、名額釋放和資料排除會改變後續交易，不能直接將原版淨值除以二。','caption'));
    table(sleeves,['年度','合併淨報酬','年度最大回撤','0050淨報酬'],metrics.annual.map((x,i)=>[x.year+(x.partial_year?'（至10/2）':''),pct(x.total_return),pct(x.max_drawdown),pct(r.cases[`benchmark:original:${m}`].metrics.annual[i].total_return)]));
    for (const [id,label] of [['combined:combined','合併100萬'],['poc_up_red:original','硬篩原版100萬'],['poc_priority:original','優先原版100萬'],['poc_up_red:split50','硬篩分帳50萬'],['poc_priority:split50','優先分帳50萬']]) {
      const button=el('button',label+'曲線','primary-button');
      button.onclick=async()=>{const request=++generation,selected=mode.value,prefix=`/archive/${item.id}/cases/`;button.disabled=true;detail.replaceChildren(el('p','正在核對帳戶…'));
        try {const [x,z]=await Promise.all([api(prefix+`${id.replaceAll(':','__')}__${selected}`),id.endsWith(':split50')?Promise.resolve(null):api(prefix+`benchmark__original__${selected}`)]);if(request!==generation)return;detail.replaceChildren();renderAccountDetail(x,detail,z);}
        catch(e){if(request===generation)detail.replaceChildren(el('p',e.message,'alert error'));}finally{button.disabled=false;}};
      actions.append(button);
    }
  }
  mode.addEventListener('change',render);render();
  host.append(el('p','原版的選股、出場與成交模型保持不變。這是已見歷史的日區間成交價估算；盤中零股時序、實際成交及鎖版後未見樣本仍未驗證，紙上交易未啟用。','result-note'));
  const link=el('a','查看完整結果與來源紀錄 ↗');link.href=item.download_url;link.target='_blank';link.rel='noopener';host.append(link);
};
