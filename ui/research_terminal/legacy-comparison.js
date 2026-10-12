'use strict';

window.renderLegacyComparison = function(item,host,helpers) {
  const {el,tag,number,pct,api,renderAccountDetail}=helpers;
  host.replaceChildren();
  host.append(el('div','SAME CAPITAL · SAME PERIOD','eyebrow'),el('h2','949% 與 576%：放在同一尺度比較'));
  if (!item?.legacy_comparison || item.status!=='verified_publication') {
    host.append(el('p','同口徑結果尚未發布；舊版報酬不代表這次結果。','result-note'));return;
  }
  const report=item.legacy_comparison;
  host.append(el('p',`${report.start} → ${report.end} · 各 ${number(report.initial_cash)} 元 · 個股3個活躍名額 · ${report.completed_cases} / 12 個帳戶完成`,'caption'),
    tag('訊號隔日買 (最低＋最高) ÷ 2','neutral'),
    el('p','賣價維持 L＋30%×(H−L)。保留各套原選股與出場規則；HL2為事後價格代理，尚未通過實戰認證。','result-note'));
  for(const d of Object.values(report.diagnostics))host.append(el('p',`${d.name}：${d.benchmark_wins} / 4 組勝過0050，最差回撤 ${pct(d.worst_drawdown)}。`,'caption'));
  const select=el('select'),controls=el('div',undefined,'research-controls');select.setAttribute('aria-label','舊策略成本情境');
  const labels={baseline:'原成本',fee2:'佣金加倍',slip2:'滑價加倍',combined:'佣金與滑價同時加倍'};
  for(const [key,label]of Object.entries(labels)){const option=el('option',label);option.value=key;select.append(option);}
  controls.append(select);const costs=el('p',undefined,'caption'),wrap=el('div',undefined,'research-table comparison-table');
  const table=el('table'),head=el('thead'),header=el('tr'),body=el('tbody');
  for(const label of ['策略','累積淨報酬','相對0050','最大回撤','2024超額','2025超額','2026至10/2超額','總成本','入選批次','明細'])header.append(el('th',label));
  head.append(header);table.append(head,body);wrap.append(table);
  const detail=el('div',undefined,'comparison-detail');detail.setAttribute('aria-live','polite');
  host.append(controls,costs,el('p','表格可左右滑動；「曲線與明細」開啟逐日帳戶與成交。','caption'),wrap,detail);
  let generation=0;
  function render(){
    ++generation;body.replaceChildren();detail.replaceChildren();const mode=select.value||'baseline',cfg=report.modes[mode];
    costs.textContent=`每邊佣金 ${pct(cfg.commission,4)}、滑價 ${pct(cfg.slippage,2)}；最低每通路20元；賣出稅維持個股0.3%／0050 0.1%。`;
    for(const arm of ['benchmark','liquid_universe','poc_priority','poc_up_red']){
      const reference=arm==='poc_up_red';const row=reference?report.hard_filter_reference[mode]:report.cases[`${arm}:${mode}`];
      const tr=el('tr');if(arm==='benchmark')tr.className='comparison-benchmark';
      tr.append(el('td',reference?'POC硬篩選（上一輪同口徑）':row.name));
      if(!row.completed){const td=el('td',`未完成：${row.reason||'資料缺件'}`);td.setAttribute('colspan','9');tr.append(td);body.append(tr);continue;}
      for(const v of [pct(row.total_return),pct(row.excess_return),pct(row.max_drawdown),...row.annual_excess.map(a=>pct(a.excess)),number(row.metrics.costs.total_cost),number(row.metrics.funded_events)])tr.append(el('td',v));
      const td=el('td'),button=el('button','曲線與明細','mini-action');button.setAttribute('aria-label',`${reference?'POC硬篩選':row.name} ${labels[mode]} 比較明細`);
      button.addEventListener('click',async()=>{const request=++generation;button.disabled=true;detail.replaceChildren(el('p','讀取帳戶…'));
        try{const study=reference?'top3_validation_20261012':item.id;const data=await api(`/archive/${study}/cases/${arm}_${mode}`);
          if(request!==generation)return;detail.replaceChildren();
          detail.append(el('p',`資料排除 ${row.data_gap_exclusions} 筆；POC恢復原排序 ${row.selection_fallback_days??0} 日。`,'caption'));
          for(const note of row.settlement_evidence_notes||[])detail.append(el('p',note,'result-note'));
          renderAccountDetail(data,detail);detail.scrollIntoView({behavior:'smooth',block:'start'});
        }catch(error){if(request===generation)detail.replaceChildren(el('p',error.message,'alert error'));}finally{button.disabled=false;}});
      td.append(button);tr.append(td);body.append(tr);
    }
  }
  select.addEventListener('change',render);render();
  const differences=el('details',undefined,'comparison-method');differences.append(el('summary','差異到底在哪？舊版本、策略規則與資料'));
  for(const old of report.old_versions)differences.append(el('p',`${old.name} 舊報酬 ${pct(old.total_return)}（${old.start}～${old.end}）。`));
  for(const text of report.differences)differences.append(el('p',text));
  for(const [arm,path]of Object.entries(report.path_differences||{})){
    differences.append(el('p',`${report.cases[`${arm}:baseline`].name}：2024起舊／新原成本帳戶分別入選 ${path.old_funded_events}／${path.new_funded_events} 批，共同 ${path.common_funded_events} 批。期間重置、價格與容量會一起改變資金路徑。`));
    differences.append(el('p',`2024首日：舊帳戶承接資產 ${number(path.old_opening_nav_2024)} 元、${path.old_carry_positions.length} 檔舊持股；新帳戶從 ${number(path.new_opening_nav_2024)} 元現金開始。`));
    for(const [version,focus]of Object.entries(path.focus_5386)){
      differences.append(el('p',`5386 青雲・${version==='old'?'舊帳戶':'新原成本帳戶'}：${focus.trades.length} 筆成交。`));
      for(const t of focus.trades)differences.append(el('p',`${t.date} ${t.side==='buy'?'買':'賣'} ${number(t.qty)} 股 × ${number(t.reference_price,4)}（${t.channel}／${t.reason}）`,'caption'));
    }
    for(const missed of path.missed_5386_legacy_entries||[]){
      const reason={slots_full:'持股名額已滿',resource_slots_locked:'當日名額已鎖定',resource_cash_locked:'可用資金已鎖定'};
      differences.append(el('p',`舊帳戶 ${missed.date} 買青雲，新帳戶未買：${missed.new_orders.map(o=>reason[o.failure]||o.failure||'零成交').join('、')||'當日無買單'}。開盤前持股 ${missed.opening_holdings.map(h=>h.stock_id).join('、')||'無'}；現金 ${number(missed.opening_cash)} 元。`));
    }
    if(path.stock_profit_differences?.length){
      differences.append(el('p',path.stock_profit_note,'caption'));
      for(const p of path.stock_profit_differences.slice(0,5))differences.append(el('p',`${p.stock_id} 損益：舊 ${number(p.old_profit)} 元 → 新 ${number(p.new_profit)} 元；差 ${number(p.difference)} 元。`));
    }
  }
  const d=report.candidate_differences;
  differences.append(el('p',`擴大股池：截至舊終點，${number(d.old_count)} → ${number(d.current_count)} 個候選，新增 ${d.added.length}、刪除 ${d.removed.length}、既有內容變更 ${d.changed.length}；本輪共 ${number(report.candidate_count)} 個候選。`));
  host.append(differences,el('p',`保留 ${report.trial_accounting.total_recorded_attempts} 次帳戶嘗試，其中 ${report.trial_accounting.incomplete_attempts} 次失敗／中止。0050同成本帳本與上一輪逐欄核對。`,'caption'));
  for(const text of report.limitations)host.append(el('p',text,'result-note'));
};
