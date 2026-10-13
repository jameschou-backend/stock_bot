'use strict';

window.renderLegacyComparison = function(item,host,helpers) {
  const {el,tag,number,pct,api,renderAccountDetail}=helpers;
  host.replaceChildren();
  host.append(el('div','SAME CAPITAL · SAME PERIOD','eyebrow'),el('h2','兩套原策略，同口徑重跑'));
  if (!item?.legacy_comparison || item.status!=='verified_publication') {
    host.append(el('p','同口徑結果尚未發布；舊版報酬不代表這次結果。','result-note'));return;
  }
  const report=item.legacy_comparison;
  window.renderStudyOverview(item,host,helpers,'legacy');
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
