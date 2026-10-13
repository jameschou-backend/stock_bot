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
  window.renderStudyOverview(item,host,helpers,'top3');
  const method=el('details',undefined,'comparison-method');method.append(el('summary','通過條件與尚缺資料'));
  for(const text of [report.out_of_sample.explanation,report.paper.explanation,...report.limitations])method.append(el('p',text));
  if(report.trial_accounting)method.append(el('p',`保留 ${report.trial_accounting.total_recorded_attempts} 次帳戶嘗試，其中 ${report.trial_accounting.incomplete_attempts} 次失敗／中止。`));
  host.append(method);
  const reconciliation=el('details',undefined,'comparison-method');reconciliation.append(el('summary','POC 不是900%？查看各版本差異'),el('p',report.poc_note));
  for(const row of report.poc_reconciliation) reconciliation.append(el('p',`${row.name}：+${number(row.return_pct,2)}%（${row.start}～${row.end}）。${row.difference}`));
  host.append(reconciliation,el('p','歷史版本分頁另保留31組買70／賣30結果。','caption'));
};
