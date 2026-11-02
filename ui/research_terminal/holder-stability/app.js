(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const arms = {drawdown:'跌深共同母體', stable:'跌深＋持股穩定', confirmed:'跌深＋止跌確認',
    stable_confirmed:'跌深＋穩定＋確認', decreasing:'跌深＋持股減少', decreasing_confirmed:'跌深＋減少＋確認'};
  const states = {stable:'穩定', decreasing:'減少', mixed:'混合', unknown:'未知'};
  const statuses = {evaluated:'有效', immature:'未成熟', stock_path_missing:'個股路徑未知',
    benchmark_path_missing:'基準路徑未知', entry_below_stop:'開盤已跌破停損，跳過'};
  const number = new Intl.NumberFormat('zh-TW', {maximumFractionDigits:2});
  let data = null;
  let generation = 0;
  function openStock(row) { globalThis.HolderChart.open(row.stock_id, true); }
  const pct = (value, suffix='%') => Number.isFinite(value) ? `${value>0?'+':''}${(value*100).toFixed(2)}${suffix}` : '未知';
  const ratio = value => Number.isFinite(value) ? `${(value*100).toFixed(2)}%` : '未知';
  const increasing = row => Number.isFinite(row.drawdown60) && row.drawdown60 <= -.2 &&
    Number.isFinite(row.large_pct_delta4) && row.large_pct_delta4 > 0;
  const increasingOrder = (a,b) => b.large_pct_delta4-a.large_pct_delta4 || a.drawdown60-b.drawdown60 ||
    b.amount20-a.amount20 || a.stock_id.localeCompare(b.stock_id);
  const text = (tag, value, className='') => {
    const element = document.createElement(tag);
    element.textContent = value;
    if (className) element.className = className;
    return element;
  };
  function candidateRows(rows, search, scope, sort) {
    const query = search.trim().toLocaleLowerCase();
    // Freeze the top ten before searching; search must not promote another member.
    if (scope === 'increasing_top10') return rows.filter(increasing).sort(increasingOrder).slice(0,10)
      .filter(row => !query || `${row.name} ${row.stock_id}`.toLocaleLowerCase().includes(query));
    return rows.filter(row => (!query || `${row.name} ${row.stock_id}`.toLocaleLowerCase().includes(query)) &&
      (scope === 'all' || scope === 'named' && row.named_case || scope === 'both' && row.delayed_state === 'stable' ||
        scope === 'different' && row.delayed_state !== 'stable')).sort((a,b) => {
      const delta = sort === 'stock' ? 0 : sort === 'drawdown' ? a.drawdown60-b.drawdown60 :
        sort === 'holder' ? b.large_pct_delta4-a.large_pct_delta4 : b.amount20-a.amount20;
      return delta || a.stock_id.localeCompare(b.stock_id);
    });
  }
  function cases() {
    $('case-cards').replaceChildren();
    for (const id of ['6442','6223']) {
      const row = data.candidates.find(r => r.stock_id === id);
      if (!row) continue;
      const card = text('article','', 'case-card');
      const header = text('div','', 'case-header');
      const name = text('button',row.name,'case-name stock-open');
      name.type = 'button'; name.addEventListener('click',() => openStock(row));
      name.append(text('small',row.stock_id));
      header.append(name, text('span',row.price_confirmed?'已確認':'持股穩定 · 未止跌確認','tag amber-tag'));
      const values = text('div','', 'case-numbers');
      for (const [label,value,suffix] of [['距60日高收盤',row.drawdown60,'%'], ['持股占比四週變動',row.large_pct_delta4,' pp'], ['級距股數四週變動',row.large_units_relative4,'%']]) {
        const cell = text('div','');
        cell.append(text('span',label),text('strong',pct(value,suffix))); values.append(cell);
      }
      card.append(header,values,text('p',`持股觀察日 ${row.observed_date} · pp 為百分點 · 使用主設定`));
      $('case-cards').append(card);
    }
  }
  function comparison() {
    if (!data) return;
    const phase = $('period').value, rule = $('exit').value, lag = Number($('lag').value), cost = $('cost').value;
    const rows = data.statistics.filter(row => row.phase === phase && row.rule === rule && row.lag === lag && row.cost === cost);
    $('comparison-rows').replaceChildren(); $('coverage').replaceChildren();
    for (const row of rows) {
      const tr = text('tr','',row.arm === 'stable' || row.arm === 'stable_confirmed' ? 'featured' : '');
      const values = [arms[row.arm],`${number.format(row.evaluated)} / ${number.format(row.events)}`,
        pct(row.mean_net),pct(row.mean_excess,' pp'),pct(row.median_net),
        Number.isFinite(row.win_rate) ? `${(row.win_rate*100).toFixed(1)}%`:'未知',pct(row.worst_mae)];
      for (const value of values) tr.append(text('td',value));
      $('comparison-rows').append(tr);
      const reasons = Object.entries(row.status_counts).map(([key,count]) => `${statuses[key] || '未知狀態'} ${number.format(count)}`).join('、');
      $('coverage').append(text('p',`${arms[row.arm]}：${reasons || '沒有事件'}。`));
    }
    const lookup = Object.fromEntries(rows.map(row => [row.arm,row]));
    const delta = (a,b) => Number.isFinite(lookup[a]?.mean_net) && Number.isFinite(lookup[b]?.mean_net) ? pct(lookup[a].mean_net-lookup[b].mean_net,' 個百分點'):'未知';
    $('increment').textContent = `持股穩定的平均淨報酬增量：穩定組相對共同母體 ${delta('stable','drawdown')}；已有止跌確認時，再加穩定條件 ${delta('stable_confirmed','confirmed')}。`;
    $('comparison-context').textContent = `${$('period').selectedOptions[0].textContent} · ${$('exit').selectedOptions[0].textContent} · 資料延遲 ${lag} 日 · 單邊滑價 ${cost === 'main'?'0.45%':'0.9%'}。${rule==='stop20'?'收盤跌破參考後，次日開盤退出。':'次日開盤參考進場，於第 '+rule+' 個交易日收盤評價。'}`;
  }
  function watchlist() {
    if (!data) return;
    const positive = $('scope').value === 'increasing_top10';
    if (positive) $('sort').value='holder';
    $('sort').disabled=positive;
    $('watch-table').classList.toggle('positive-top10',positive);
    $('positive-filter-note').hidden=!positive;
    $('watch-title').textContent=positive?'跌深＋持股比例增加':'112 檔觀察名單';
    const ranked = positive ? candidateRows(data.candidates,'','increasing_top10','holder') : [];
    const rows = candidateRows(data.candidates,$('search').value,$('scope').value,$('sort').value);
    $('candidate-rows').replaceChildren();
    for (const row of rows) {
      const tr = text('tr','');
      const name = text('td','');
      const button = text('button',row.name+' ↗','stock-name stock-open');
      button.type = 'button'; button.addEventListener('click',() => openStock(row));
      name.append(button,text('span',`${row.stock_id}${positive?' · 第 '+(ranked.findIndex(r=>r.stock_id===row.stock_id)+1)+' 名':row.named_case?' · 指定案例':''}`,'stock-code'));
      tr.append(name);
      tr.append(text('td',Number.isFinite(row.close)?number.format(row.close):'未知','watch-general'),text('td',pct(row.drawdown60)));
      const holding = text('td','','watch-positive');
      const prior = Number.isFinite(row.large_pct) && Number.isFinite(row.large_pct_delta4) ? row.large_pct-row.large_pct_delta4 : null;
      holding.append(text('span',ratio(prior)+' →','ratio-before'),text('strong',ratio(row.large_pct),'ratio-now'));
      tr.append(holding,text('td',pct(row.large_pct_delta4,'')));
      const confirmation = text('td','','watch-positive');
      confirmation.append(text('span',row.price_confirmed===true?'已確認':row.price_confirmed===false?'尚未確認':'資料未知',
        'state '+(row.price_confirmed===true?'state-stable':'state-mixed')));
      tr.append(confirmation,text('td',pct(row.large_units_relative4),'watch-general'));
      const delayed = text('td','','watch-general');
      const state = Object.hasOwn(states,row.delayed_state)?row.delayed_state:'unknown';
      delayed.append(text('span',states[state],`state state-${state}`));tr.append(delayed);
      $('candidate-rows').append(tr);
    }
    $('list-count').textContent = positive ? `顯示 ${rows.length} / 前 10 檔 · 共 ${data.candidates.filter(increasing).length} / ${data.candidates.length} 檔符合條件` : `顯示 ${rows.length} / ${data.candidates.length} 檔`;
    $('empty').hidden = rows.length !== 0;
    $('empty').textContent=positive?'前 10 檔內沒有符合搜尋的股票，請調整搜尋或清除篩選。':'沒有符合搜尋條件的觀察股，請調整或清除篩選。';
    const url=new URL(location.href);
    if(positive)url.searchParams.set('scope','increasing_top10');else url.searchParams.delete('scope');
    history.replaceState(null,'',url);
  }
  async function load() {
    const current = ++generation, controller = new AbortController();
    data = null; $('study').hidden = true; $('error').hidden = true; $('loading').hidden = false;
    globalThis.HolderChart.clear();
    const timer = setTimeout(() => controller.abort(),30000);
    try {
      const response = await fetch('/research-terminal/api/holder-stability',{cache:'no-store',signal:controller.signal});
      if (!response.ok) throw new Error('研究封存暫時無法核對，請稍後重試。');
      const result = await response.json();
      if (current !== generation) return;
      if (result.study_id !== 'holder_stability_600_20261102' || result.candidates?.length !== 112 ||
          result.confirmed_count !== 0 || result.live_qualified !== false || result.adopted !== false) throw new Error('研究版本或名單不符，請核對後再載入。');
      data = result;
      $('watch-count').textContent = data.candidates.length;
      $('confirmed-count').textContent = data.confirmed_count;
      $('week-count').textContent = data.snapshot_weeks;
      const headline = data.statistics.find(row => row.arm === 'stable' && row.rule === '20' && row.phase === '2025_2026' && row.cost === 'main' && row.lag === 8);
      $('headline-return').textContent = pct(headline.mean_net);
      $('headline-excess').textContent = pct(headline.mean_excess,'');
      $('headline-excess').append(text('small','百分點'));
      $('headline-events').textContent = number.format(headline.evaluated);
      $('paired-rows').replaceChildren();
      for (const pair of data.paired_statistics.filter(r => r.phase === '2025_2026' && r.rule === '20' && r.cost === 'main')) {
        const row = text('tr','');
        for (const value of [`${pair.lag} 日 · ${pair.confirmed?'已確認':'未確認'}`, `${number.format(pair.evaluated)} / ${pair.dates}`,
          pct(pair.date_mean_delta,' pp'),`${pct(pair.descriptive_low,'')} ～ ${pct(pair.descriptive_high,' pp')}`]) row.append(text('td',value));
        $('paired-rows').append(row);
      }
      const params = new URLSearchParams(location.search);
      if(params.get('scope')==='increasing_top10') $('scope').value='increasing_top10';
      cases(); comparison(); watchlist(); $('study').hidden = false;
      const selected = params.get('stock');
      if (selected && data.candidates.some(row => row.stock_id === selected)) globalThis.HolderChart.open(selected, true);
      else if(params.get('scope')==='increasing_top10' && location.hash==='#watchlist') $('watchlist').scrollIntoView({block:'start',behavior:'auto'});
    } catch (error) {
      if (current !== generation) return;
      data = null; $('study').hidden = true; $('error').hidden = false;
      $('error-text').textContent = error.name === 'AbortError' ? '讀取逾時，請確認本機服務後重試。':'研究資料無法載入或核對，請確認本機服務與封存完整後重試。';
    } finally {
      clearTimeout(timer); if (current === generation) $('loading').hidden = true;
    }
  }
  function bind() {
    if (!globalThis.HolderChart) throw new Error('圖表元件尚未載入，請重新載入完整介面');
    globalThis.HolderChart.bind();
    for (const id of ['period','exit','lag','cost']) $(id).addEventListener('change',comparison);
    $('search').addEventListener('input',watchlist);
    for (const id of ['scope','sort']) $(id).addEventListener('change',watchlist);
    $('clear').addEventListener('click',() => { $('search').value=''; $('scope').value='all'; $('sort').value='liquidity'; watchlist(); });
    $('retry').addEventListener('click',load);
  }
  bind(); load();
})();
