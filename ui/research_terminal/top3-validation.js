'use strict';

window.renderTop3Validation = function(item, host, helpers) {
  const {el, tag, number, pct, api, renderAccountDetail} = helpers;
  host.replaceChildren();
  host.append(el('div','FIXED TOP THREE · HL2','eyebrow'),el('h2','前三名：成本壓力與樣本外資格'));
  if (!item?.validation || item.status !== 'verified_publication') {
    host.append(el('p','驗證結果尚未發布；未取得新資格。','result-note'));
    return;
  }
  const report=item.validation;
  host.append(el('p',`${report.start} → ${report.end} · 本金 ${number(report.initial_cash)} 元 · ${report.completed_cases} / 16 個帳戶完成`,'caption'));
  host.append(tag('買價改為訊號隔日 (最低＋最高) ÷ 2','neutral'));
  host.append(el('p','賣價維持 L＋30%×(H−L)。高低價在收盤後才知道；這是歷史成交估算，尚無即時成交認證。','result-note'));
  const stats=el('div',undefined,'research-stats');
  for (const [label,value] of [['真正未見樣本',`${report.out_of_sample.sessions} 日`],['通過歷史壓力關卡',`${Object.values(report.qualification).filter(g=>g.checks.historical_stress_pass).length} / 3 套`],['已啟用紙上交易',`${report.paper.enabled_strategies.length} 套`]]) {
    const box=el('div',undefined,'research-stat');box.append(el('span',label),el('strong',value));stats.append(box);
  }
  host.append(stats,el('p',report.out_of_sample.explanation,'result-note'));
  if (report.trial_accounting) host.append(el('p',`本輪保留 ${report.trial_accounting.total_recorded_attempts} 次帳戶嘗試，其中 ${report.trial_accounting.incomplete_attempts} 次失敗／中止；未從紀錄刪除。`,'caption'));
  for (const row of Object.values(report.diagnostics||{})) {
    host.append(el('p',`${row.name}：${row.positive_stresses} / 4 組淨報酬為正，${row.benchmark_wins} / 4 組勝過0050；最差回撤 ${pct(row.worst_drawdown)}。`,'caption'));
  }
  const method=el('details',undefined,'comparison-method');method.append(el('summary','通過條件與尚缺資料'));
  for (const text of ['四個成本情境都須完整、淨報酬為正、勝過同期0050且最大回撤不差於0050。這是本次研究篩選門檻。',`紙上門檻另需版本固定後至少 ${report.out_of_sample.minimum_sessions} 個市場日、${report.out_of_sample.minimum_settled_cohorts} 個已結清批次，以及可執行成交驗證。`,report.paper.explanation,...report.limitations]) method.append(el('p',text));
  host.append(method);
  const control=el('div',undefined,'research-controls'),select=el('select');select.setAttribute('aria-label','前三名成本情境');
  const labels={baseline:'原成本',fee2:'佣金加倍',slip2:'滑價加倍',combined:'佣金與滑價同時加倍'};
  for (const [value,label] of Object.entries(labels)) {const option=el('option',label);option.value=value;select.append(option);}
  control.append(select);host.append(control,el('p','表格可左右滑動查看分年與成本；最右側「曲線與明細」可展開帳戶。','caption'));
  const costs=el('p',undefined,'caption'),wrap=el('div',undefined,'research-table comparison-table'),table=el('table'),head=el('tr');
  for (const label of ['策略','累積淨報酬','相對0050','最大回撤','2024超額','2025超額','2026至10/2超額','252日勝出窗口','總成本','明細']) head.append(el('th',label));
  const thead=el('thead'),body=el('tbody');thead.append(head);table.append(thead,body);wrap.append(table);host.append(costs,wrap);
  const detail=el('div',undefined,'comparison-detail');detail.setAttribute('aria-live','polite');host.append(detail);let generation=0;
  function render() {
    ++generation;detail.replaceChildren();body.replaceChildren();const mode=select.value||'baseline',cfg=report.modes[mode];
    costs.textContent=`每邊佣金 ${pct(cfg.commission,4)}、滑價 ${pct(cfg.slippage,2)}；最低每通路20元；賣出稅個股0.3%／0050 0.1%。252日窗口彼此重疊，屬已見歷史。`;
    for (const arm of ['benchmark','obv20_breakout','poc_up_red','entry_peer_narrow']) {
      const row=report.cases[`${arm}:${mode}`],tr=el('tr');if(arm==='benchmark')tr.className='comparison-benchmark';
      tr.append(el('td',row.name));
      if (!row.completed) {const td=el('td',`未完成：${row.reason||'資料缺件'}`);td.setAttribute('colspan','9');tr.append(td);body.append(tr);continue;}
      for (const value of [pct(row.total_return),pct(row.excess_return),pct(row.max_drawdown),...(row.annual_excess||[]).map(a=>pct(a.excess)),pct(row.rolling252?.win_fraction),number(row.metrics.costs.total_cost)]) tr.append(el('td',value));
      const td=el('td'),button=el('button','曲線與明細','mini-action');button.setAttribute('aria-label',`${row.name} ${labels[mode]} 驗證明細`);
      button.addEventListener('click',async()=>{const request=++generation;button.disabled=true;detail.replaceChildren(el('p','讀取帳戶…'));
        try {const data=await api(`/archive/${item.id}/cases/${encodeURIComponent(`${arm}_${mode}`)}`);if(request!==generation)return;detail.replaceChildren();
          detail.append(el('p',`資料排除 ${row.data_gap_exclusions} 筆；分年與滾動結果皆為已見資料。`,'caption'));
          for(const note of row.settlement_evidence_notes||[])detail.append(el('p',note,'result-note'));
          renderAccountDetail(data,detail);detail.scrollIntoView({behavior:'smooth',block:'start'});
        }catch(error){if(request===generation)detail.replaceChildren(el('p',error.message,'alert error'));}finally{button.disabled=false;}});
      td.append(button);tr.append(td);body.append(tr);
    }
  }
  select.addEventListener('change',render);render();
  const reconciliation=el('details',undefined,'comparison-method');reconciliation.append(el('summary','POC 不是900%？查看各版本差異'),el('p',report.poc_note));
  for(const row of report.poc_reconciliation) reconciliation.append(el('p',`${row.name}：+${number(row.return_pct,2)}%（${row.start}～${row.end}）。${row.difference}`));
  host.append(reconciliation,el('p','下方31組為已封存的買70／賣30結果；與本區新買價分開保留。','caption'));
};
