'use strict';

// Every number belongs to one publication, one account and one cost scenario.
window.renderStudyOverview = function(item, host, helpers, family) {
  const {el,tag,number,pct,api,renderAccountDetail}=helpers;
  const report=family==='legacy'?item.legacy_comparison:item.validation;
  const modeNames={baseline:'原成本',fee2:'佣金加倍',slip2:'滑價加倍',combined:'佣金與滑價同時加倍'};
  const arms=family==='legacy'?['poc_priority','liquid_universe','benchmark']:['poc_up_red','obv20_breakout','entry_peer_narrow','benchmark'];
  const shortName=row=>row.name.replace(/（原\d+%）/g,'');
  const pp=n=>n==null?'—':`${n>0?'+':''}${number(n*100,2)} 個百分點`;
  const metrics=key=>item.analytics?.status==='verified_supplement'?item.analytics.cases[key]:null;
  const win=m=>m?.not_applicable?'不適用':m?.win_rate==null?'—':`${pct(m.win_rate,2)}（${m.wins}／${m.settled}）`;
  host.append(el('p',`${report.start} → ${report.end} · 各 ${number(report.initial_cash)} 元 · 個股 3 名額`,'caption'),
    el('p','訊號隔日買 (最低＋最高) ÷ 2；賣 L＋30%×(H−L)。日區間價格為事後估算。','scope-line'));
  const controls=el('div',undefined,'study-controls'),strategy=el('select'),mode=el('select');
  strategy.setAttribute('aria-label',family==='legacy'?'重跑比較策略':'前三名比較策略');
  mode.setAttribute('aria-label',family==='legacy'?'舊策略成本情境':'前三名成本情境');
  for(const arm of arms){const o=el('option',shortName(report.cases[`${arm}:baseline`]));o.value=arm;strategy.append(o);}
  for(const [id,name] of Object.entries(modeNames)){const o=el('option',name);o.value=id;mode.append(o);}
  controls.append(strategy,mode);host.append(controls);
  const summary=el('div',undefined,'decision-metrics'),actions=el('div',undefined,'study-actions'),
    qualification=el('section',undefined,'qualification-card'),comparison=el('details',undefined,'comparison-method'),
    stress=el('details',undefined,'comparison-method'),details=el('div',undefined,'comparison-detail');
  details.setAttribute('aria-live','polite');summary.setAttribute('aria-live','polite');
  host.append(summary,actions,qualification,stress,comparison,details);
  let generation=0;
  function metric(label,value,note){const box=el('div',undefined,'decision-metric');box.append(el('span',label),el('strong',value));if(note)box.append(el('small',note));summary.append(box);}
  function table(parent,labels,rows){const wrap=el('div',undefined,'compact-comparison'),t=el('table'),head=el('thead'),h=el('tr'),body=el('tbody');labels.forEach(v=>h.append(el('th',v)));head.append(h);
    for(const values of rows){const row=el('tr');values.forEach((v,i)=>{const cell=el('td',v);cell.setAttribute('data-label',labels[i]);row.append(cell);});body.append(row);}t.append(head,body);wrap.append(t);parent.append(wrap);}
  function render(){
    ++generation;summary.replaceChildren();actions.replaceChildren();qualification.replaceChildren();details.replaceChildren();
    const arm=strategy.value||arms[0],cost=mode.value||'baseline',row=report.cases[`${arm}:${cost}`],base=report.cases[`benchmark:${cost}`],m=metrics(`${arm}:${cost}`),cfg=report.modes[cost];
    if(row.completed){
      metric('累積淨報酬',pct(row.total_return),`期末 ${number(row.metrics.final_nav)} 元`);
      metric('相對 0050（百分點）',row.excess_return==null?'—':`${row.excess_return>0?'+':''}${number(row.excess_return*100,2)}`,`0050 淨報酬 ${pct(base.total_return)}`);
      metric('最大回撤',pct(row.max_drawdown),'帳戶從高點的最大跌幅');
      metric('已結清勝率',m?.not_applicable?'不適用':pct(m?.win_rate,2),m?.not_applicable||(m?`${m.wins} 筆獲利／${m.settled} 批結清`:'獲利筆數／已結清批次'));
      metric('平均賺賠比',m?.payoff_ratio==null?'—':`${number(m.payoff_ratio,2)} 倍`,'平均獲利 ÷ 平均虧損金額');
      metric('獲利因子',m?.profit_factor==null?'—':number(m.profit_factor,2),'獲利總額 ÷ 虧損總額絕對值');
      const button=el('button','資產曲線與交易明細','primary-button');button.setAttribute('aria-label',`${row.name} ${modeNames[cost]} ${family==='legacy'?'比較':'驗證'}明細`);
      button.addEventListener('click',async()=>{const request=++generation;button.disabled=true;details.replaceChildren(el('p','讀取同成本帳戶與0050…'));
        try{const path=`/archive/${item.id}/cases/`,[data,benchmark]=await Promise.all([api(path+`${arm}_${cost}`),arm==='benchmark'?Promise.resolve(null):api(path+`benchmark_${cost}`)]);
          if(request!==generation)return;details.replaceChildren();renderAccountDetail(data,details,benchmark);
          const disclosure=el('p',`已結清 ${m?.settled??'—'} 批，未平倉 ${m?.open??'—'} 批，已賣出但應收未結 ${m?.pending_only??'—'} 批。平均獲利 ${number(m?.average_win)} 元、平均虧損 ${number(m?.average_loss)} 元；前5筆占正損益總額 ${pct(m?.top5_profit_share)}。金額受部位及複利影響。`,'result-note');details.append(disclosure);
          details.append(el('p',`資料排除 ${row.data_gap_exclusions??0} 筆；${row.selection_fallback_days==null?'':`POC恢復原排序 ${row.selection_fallback_days} 日；`}分年與滾動結果皆為已見資料。`,'caption'));
          for(const note of row.settlement_evidence_notes||[])details.append(el('p',note,'result-note'));
          details.scrollIntoView({behavior:'smooth',block:'start'});
        }catch(error){if(request===generation)details.replaceChildren(el('p',error.message,'alert error'));}finally{button.disabled=false;}});
      actions.append(button,el('span',m?`未平倉 ${m.open} 批 · 勝率已含費稅、滑價與實收分配`:`績效補充未就緒：${item.analytics?.reason||'缺少已核對補充資料'}`,'caption'));
    }else summary.append(el('p',`未完成：${row.reason||'資料缺件'}`,'alert error'));
    qualification.append(el('h3',arm==='benchmark'?'0050 是同期買進持有基準':'驗證進度與下一步'));
    if(arm!=='benchmark'){
      const rows=Object.keys(modeNames).map(k=>report.cases[`${arm}:${k}`]);
      const checks=report.qualification?.[arm]?.checks;
      const stressPass=checks?checks.historical_stress_pass:rows.every(r=>r.completed&&r.total_return>0&&r.excess_return>0&&r.max_drawdown>=report.cases[`benchmark:${r.mode}`]?.max_drawdown);
      const badges=el('div',undefined,'qualification-badges');
      for(const [label,ok] of [['歷史帳戶完成',rows.every(r=>r.completed)],['歷史壓力門檻通過',stressPass],['真正樣本外通過',false],['可執行成交驗證',false],['紙上交易啟用',false],['實盤合格',false]])badges.append(tag(`${ok?'✓':'待'} ${label}`,ok?'green':'neutral'));
      qualification.append(badges,el('p','下一步：建立事前可執行的委託規則，封存版本後累積真正前瞻紀錄。HL2事後價格尚不能認證為實際成交。','caption'));
      qualification.append(el('p',`歷史壓力門檻：四組成本都須完成、獲利、贏0050且回撤不差於0050。目前完成 ${rows.filter(r=>r.completed).length}/4、獲利 ${rows.filter(r=>r.completed&&r.total_return>0).length}/4、贏0050 ${rows.filter(r=>r.completed&&r.excess_return>0).length}/4、回撤不差 ${rows.filter(r=>r.completed&&r.max_drawdown>=report.cases[`benchmark:${r.mode}`]?.max_drawdown).length}/4。`,'caption'));
      if(family==='top3')qualification.append(el('p',`真正未見樣本 ${report.out_of_sample.sessions} 日；已啟用紙上交易 ${report.paper.enabled_strategies.length} 套。門檻另需 ${report.out_of_sample.minimum_sessions} 個市場日與 ${report.out_of_sample.minimum_settled_cohorts} 個結清批次。`,'caption'));
      else qualification.append(el('p','此封存研究沒有合格未見樣本，紙上未啟用。','caption'));
    }
    stress.replaceChildren(el('summary','四種成本情境，一起比較'));
    table(stress,['成本情境','淨報酬','相對0050（百分點）','最大回撤','已結清勝率'],Object.entries(modeNames).map(([k,n])=>{const r=report.cases[`${arm}:${k}`];return [n,r.completed?pct(r.total_return):'未完成',r.completed?pp(r.excess_return):'—',r.completed?pct(r.max_drawdown):'—',win(metrics(`${arm}:${k}`))];}));
    stress.append(el('p','成本會改變股數、名額與後續資金路徑；各情境為完整帳戶重播。','caption'));
    comparison.replaceChildren(el('summary','本情境所有策略與分年結果'));
    table(comparison,['策略','淨報酬','相對0050（百分點）','最大回撤','已結清勝率','2024超額（百分點）','2025超額（百分點）','2026至10/2超額（百分點）'],arms.map(a=>{const r=report.cases[`${a}:${cost}`];return [shortName(r),r.completed?pct(r.total_return):'未完成',pp(r.excess_return),pct(r.max_drawdown),win(metrics(`${a}:${cost}`)),...(r.annual_excess||[]).map(y=>pp(y.excess))];}));
    comparison.append(el('p',`每邊佣金 ${pct(cfg.commission,4)}、滑價 ${pct(cfg.slippage,2)}；最低每通路20元，賣出稅個股0.3%／0050 0.1%。本策略總成本 ${number(row.metrics?.costs.total_cost)} 元。`,'caption'));
    if(row.rolling252)comparison.append(el('p',`本策略滾動252日：贏0050比例 ${pct(row.rolling252.win_fraction)}，最差超額 ${pp(row.rolling252.worst_excess)}（${row.rolling252.windows} 個重疊已見窗口）。此為歷史穩定度，不是獨立樣本或未來勝率。`,'caption'));
  }
  strategy.addEventListener('change',render);mode.addEventListener('change',render);render();
};

window.renderAccountCharts = function(result,host,benchmark,helpers){
  const {el,number,pct}=helpers,daily=result.daily||[],base=benchmark?.daily||[];
  if(base.length&&(result.initial_cash!==benchmark.initial_cash||base.length!==daily.length||base.some((v,i)=>v.date!==daily[i].date)))throw Error('0050期間或本金不同，停止疊圖');
  if(!daily.length)return;
  if([...daily,...base].some(r=>!Number.isFinite(r.nav)||!Number.isFinite(r.drawdown)))throw Error('曲線資料缺少資產或回撤數值');
  const legend=el('p',benchmark?'綠線：策略　橘線：同期0050（相同成本）':'綠線：帳戶','chart-legend'),readout=el('p',undefined,'chart-readout');
  const slider=el('input');slider.type='range';slider.min=0;slider.max=daily.length-1;slider.value=daily.length-1;slider.setAttribute('aria-label','資產曲線日期');
  let index=daily.length-1;const charts=[];
  host.append(legend);
  for(const [key,label] of [['nav',benchmark?'每日資產與0050':'每日資產'],['drawdown','從高點回撤']]){
    host.append(el('h3',label));const wrap=el('div',undefined,'nav-chart'),canvas=el('canvas');canvas.setAttribute('role','img');canvas.setAttribute('aria-label',label);wrap.append(canvas);host.append(wrap);
    function draw(){const w=canvas.clientWidth,h=canvas.clientHeight;if(!w)return;const dpr=window.devicePixelRatio||1;canvas.width=w*dpr;canvas.height=h*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
      const values=[...daily,...base].map(r=>r[key]).filter(Number.isFinite);if(values.length!==daily.length+base.length)throw Error('曲線資料缺少數值');
      const min=Math.min(...values),max=key==='drawdown'?0:Math.max(...values),span=max-min||1,l=8,r=64,t=16,b=24,cw=w-l-r,ch=h-t-b,x=i=>l+i*cw/Math.max(1,daily.length-1),y=n=>t+ch-(n-min)/span*ch;
      ctx.font='10px -apple-system,sans-serif';ctx.fillStyle='#63746c';
      for(let i=0;i<=4;i++){const v=min+span*i/4,yy=y(v);ctx.strokeStyle='#e5ebe5';ctx.beginPath();ctx.moveTo(l,yy);ctx.lineTo(w-r,yy);ctx.stroke();ctx.fillText(key==='nav'?`${(v/10000).toFixed(0)}萬`:`${(v*100).toFixed(0)}%`,w-r+8,yy+3);}
      for(const [rows,color] of [[daily,'#087c70'],[base,'#b57628']]){if(!rows.length)continue;ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();rows.forEach((v,i)=>i?ctx.lineTo(x(i),y(v[key])):ctx.moveTo(x(i),y(v[key])));ctx.stroke();}
      ctx.strokeStyle='#6e7d73';ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(x(index),t);ctx.lineTo(x(index),h-b);ctx.stroke();ctx.setLineDash([]);ctx.fillText(daily[0].date,l,h-5);ctx.textAlign='right';ctx.fillText(daily.at(-1).date,w-r,h-5);
    }
    canvas.addEventListener('mousemove',event=>{index=Math.max(0,Math.min(daily.length-1,Math.round((event.offsetX-8)/(canvas.clientWidth-72)*(daily.length-1))));update();});
    charts.push(draw);new ResizeObserver(draw).observe(wrap);
  }
  function update(){slider.value=index;slider.setAttribute('aria-valuetext',daily[index].date);readout.textContent=`${daily[index].date} · 帳戶 ${number(daily[index].nav)} 元／回撤 ${pct(daily[index].drawdown)}${base.length?` · 0050 ${number(base[index].nav)} 元／回撤 ${pct(base[index].drawdown)}`:''}`;charts.forEach(draw=>draw());}
  slider.addEventListener('input',()=>{index=Number(slider.value);update();});host.append(slider,readout);requestAnimationFrame(update);
};
