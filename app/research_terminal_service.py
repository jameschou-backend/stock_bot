"""Read-only research terminal over explicitly sealed local evidence.

Daily screening, retrospective signal studies and account replays deliberately
have different response types. Opening this service never fetches market data,
changes the shared database, enables a scheduler, or submits a broker order.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DAILY_AUDIT = 'artifacts/forward_simulation/strategy_scanner_20261006.json'
DAILY_AUDIT_SHA = 'a3c0874367d4e8302f89285f459e4752df6d35963aad9df2a68eebadc5153b23'
STUDY_AUDIT = 'artifacts/forward_simulation/strategy_scanner_expansion_20261005.json'
STUDY_AUDIT_SHA = '7b5602319aeda6538728ae8f9fccfe4a0d6784d934b778950fd6cb40c30d3a64'
RALLY_REPORT = 'artifacts/forward_simulation/strategy_rally_attribution_20261006.json'
SERVICE_VERSION = 'research_terminal_v1'


class EvidenceError(ValueError):
    """Evidence is missing, changed, or incompatible; do not show stale results."""


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_bytes(value):
    return json.dumps(_clean(value), ensure_ascii=False, allow_nan=False,
                      separators=(',', ':'), sort_keys=True).encode()


def _iso(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError('日期必須是 YYYY-MM-DD')
    return value


def _stamp(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class ResearchTerminal:
    def __init__(self, root=ROOT, *, daily_audit=DAILY_AUDIT,
                 daily_sha=DAILY_AUDIT_SHA, study_audit=STUDY_AUDIT,
                 study_sha=STUDY_AUDIT_SHA):
        self.root = Path(root).resolve()
        self.daily_descriptor = dict(path=daily_audit, sha256=daily_sha)
        self.study_descriptor = dict(path=study_audit, sha256=study_sha)
        self._checked = {}
        self._lock = threading.RLock()
        self._scan = None
        self._study = None
        self._rally_cases = None
        self._chart_cache = {}
        self.jobs_dir = self.root / '.cache/research-terminal/jobs'

    def _path(self, name):
        if not isinstance(name, str) or Path(name).is_absolute():
            raise EvidenceError('研究資料路徑必須在專案內')
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise EvidenceError('研究資料路徑超出專案')
        return path

    def _verify(self, descriptor):
        if (not isinstance(descriptor, dict) or not re.fullmatch(
                r'[a-f0-9]{64}', str(descriptor.get('sha256', '')))):
            raise EvidenceError('研究資料缺少 SHA256 指紋')
        path, expected = self._path(descriptor.get('path')), descriptor['sha256']
        try:
            before = _stamp(path)
            prior = self._checked.get(path)
            if prior is not None:
                if prior != (before, expected):
                    raise EvidenceError('已載入的研究資料被變更，請核對來源後重啟服務')
                return path
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected or _stamp(path) != before:
                raise EvidenceError('研究資料 SHA256 核對失敗：' + path.name)
            self._checked[path] = (before, expected)
            return path
        except OSError as exc:
            raise EvidenceError('研究資料無法讀取：' + path.name) from exc

    def _read(self, descriptor):
        path = self._verify(descriptor)
        before = _stamp(path)
        def reject(value):
            raise EvidenceError('研究 JSON 含無效數字：' + value)
        result = json.loads(path.read_text(), parse_constant=reject)
        if _stamp(path) != before:
            raise EvidenceError('研究資料在讀取時變更')
        return result

    def _unchanged(self):
        for path, (stamp, _) in list(self._checked.items()):
            try:
                if _stamp(path) != stamp:
                    raise EvidenceError('研究來源已變更，停止提供舊快取：' + path.name)
            except OSError as exc:
                raise EvidenceError('研究來源已移除：' + path.name) from exc

    def _load(self):
        with self._lock:
            self._unchanged()
            if self._scan is not None:
                return
            audit = self._read(self.daily_descriptor)
            if audit.get('schema') != 'strategy_scanner_daily_continuation_v1':
                raise EvidenceError('不支援的每日掃描證據版本')
            receipt_desc = audit['artifacts']['scanner']
            receipt = self._read(receipt_desc)
            scan_dir = str(Path(receipt_desc['path']).parent)
            scan = self._read(dict(path=scan_dir+'/scan.json',
                                   sha256=receipt['files_sha256']['scan.json']))
            manifest_desc = audit['artifacts']['bundle']
            manifest = self._read(manifest_desc)
            bundle = str(Path(manifest_desc['path']).parent)
            if (scan.get('schema') != 'multi_strategy_scan_v1'
                    or scan.get('live_qualified') is not False
                    or scan['source_end'] != manifest['end']
                    or scan['provenance']['source_hashes']['manifest.json'] != manifest_desc['sha256']):
                raise EvidenceError('每日掃描與價格封存來源不一致')
            for filename in ('close-official.parquet', 'close-quality.parquet',
                             'eligibility.parquet', 'quotes-unmasked.parquet', 'companies.parquet'):
                self._verify(dict(path=bundle+'/'+filename,
                                  sha256=manifest['files_sha256'][filename]))
            poc = self._read(audit['artifacts']['poc'])
            if (poc.get('source_sha256', {}).get(manifest_desc['path']) != manifest_desc['sha256']
                    or poc.get('end') != manifest['end']):
                raise EvidenceError('POC 與行情封存來源不一致')
            self._verify(poc['profiles'])
            if scan['provenance']['poc']['report_sha256'] != audit['artifacts']['poc']['sha256']:
                raise EvidenceError('掃描與 POC 指紋不一致')
            self.bundle = self._path(bundle)
            self.manifest = manifest
            self.source_hashes = dict(scan=receipt['files_sha256']['scan.json'],
                                      manifest=manifest_desc['sha256'],
                                      poc=audit['artifacts']['poc']['sha256'])
            self._catalog = {row['id']: row for row in scan['strategies']}
            self._days = {}
            # Remove display-only metrics on negative rules to keep the live
            # service compact; preserve every outcome and its actual explanation.
            for day in scan['days']:
                for stock in day['stocks']:
                    for result in stock['results'].values():
                        if result['status'] != 'matched':
                            result['metrics'] = {}
                self._days[day['date']] = day
            schema = pq.ParquetFile(self.bundle/'close-official.parquet').schema_arrow.names
            self._ids = set(schema)-{'date'}
            self._calendar = pd.DatetimeIndex(pd.read_parquet(
                self.bundle/'close-official.parquet', columns=['date']).date)
            companies = pd.read_parquet(self.bundle/'companies.parquet')
            self._names = dict(zip(companies.stock_id, companies.name))
            self._names['0050'] = '元大台灣50'
            self._scan = {k: v for k, v in scan.items() if k != 'days'}
            self._unchanged()

    def _load_study(self):
        with self._lock:
            self._unchanged()
            if self._study is not None:
                return
            audit = self._read(self.study_descriptor)
            descriptor = audit['receipts']['study']
            receipt = self._read(descriptor)
            base = str(Path(descriptor['path']).parent)
            report = self._read(dict(path=base+'/summary.json',
                                     sha256=receipt['files_sha256']['summary.json']))
            if (report.get('schema') != 'strategy_scanner_signal_outcomes_v1'
                    or report.get('account_independent') is not True
                    or report.get('live_qualified') is not False):
                raise EvidenceError('不支援的訊號研究格式')
            self._study_events_descriptor = dict(path=base+'/events.parquet',
                                                 sha256=receipt['files_sha256']['events.parquet'])
            self._verify(self._study_events_descriptor)
            self._study_hash = receipt['files_sha256']['summary.json']
            self._study = report

    @staticmethod
    def _account_scope():
        return dict(mode='account_replay', strategy_id='frozen_original_account',
            label='原封存個股帳戶重播', start='2022-01-03', end='2026-09-09',
            initial_cash=1000000, candidate_events=458,
            price_assumptions='依封存案例設定執行；逐案列出成交價格、成本及容量假設',
            scope_note='這是原封存帳戶，不是任選 29 種訊號的組合帳戶。日期及資金固定，0050只作比較。',
            modes=['daily', 'strict'], policies=['mixed', 'board_only', 'all'],
            stress_levels=['control', 'combined', 'all'], live_qualified=False)

    def overview(self):
        self._load()
        self._load_study()
        last = self._days[max(self._days)]
        entries = {sid for sid, row in self._catalog.items() if row['kind'] == 'entry'}
        matches = [[r for sid, r in s['results'].items()
                    if sid in entries and r['status'] == 'matched'] for s in last['stocks']]
        return dict(source_end=self._scan['source_end'], dates=list(self._days),
            default_date=last['date'], universe_count=len(last['stocks']),
            active_strategies=len(self._scan['evaluated_strategy_ids']),
            catalog_count=len(self._catalog),
            pending_strategies=sum(s['status'] != 'active' for s in self._catalog.values()),
            entry_strategies=sum(s['kind'] == 'entry' and s['status'] == 'active' for s in self._catalog.values()),
            latest=dict(entry_matched_stocks=sum(bool(x) for x in matches),
                entry_matches=sum(len(x) for x in matches),
                first_entry_stocks=sum(any(r['first_signal'] is True for r in x) for x in matches),
                market_regime=last['market_regime']),
            study={k: self._study[k] for k in ('start', 'end', 'horizons', 'costs')},
            account_replay=self._account_scope(),
            qualification=dict(live_qualified=False, complete_historical_universe=False,
                actual_fill_verified=False, historical_period_already_researched=True),
            limitations=['目前是封存研究股票池，未認證完整歷史上市櫃名單。',
                'POC 僅覆蓋原突破候選；資料不足為未知，不視為不符合。',
                '訊號在收盤確認，最早下一交易日評估成交；不是盤中即時行情。',
                '策略目錄不是全球全部策略；未接資料的方法不能假裝已測。'],
            source_hashes=self.source_hashes)

    def _selected_day(self, selected):
        self._load()
        selected = _iso(selected) if selected else max(self._days)
        if selected not in self._days:
            raise ValueError('這個日期尚無封存掃描，請選擇已提供的交易日')
        return self._days[selected]

    def _assessment(self, stock):
        matches = [sid for sid, r in stock['results'].items()
                   if r['status'] == 'matched' and self._catalog[sid]['kind'] == 'entry']
        unknown = [self._catalog[sid]['name'] for sid, r in stock['results'].items()
                   if r['status'] == 'unknown' and self._catalog[sid]['kind'] == 'entry']
        status = 'research_candidate' if matches else ('data_incomplete' if unknown else 'no_entry_signal')
        label = {'research_candidate': '符合研究條件，列入觀察',
                 'data_incomplete': '資料不足，無法判斷', 'no_entry_signal': '目前無已知進場訊號'}[status]
        reasons = [f'符合 {len(matches)} 種進場規則；規則重疊不代表獨立確認。'] if matches else []
        if unknown:
            reasons.append(f'{len(unknown)} 種進場規則資料不足，未當成通過。')
        reasons.append('尚未通過實戰資格驗證，沒有自動買進建議或委託。')
        return dict(status=status, label=label, reasons=reasons,
                    action='watch' if matches else 'wait', live_qualified=False,
                    earliest_execution='next_market_session', unknown_entry_rules=unknown)

    def signals(self, selected=None, strategy_id='poc_up_red', first_only=False, search=''):
        day = self._selected_day(selected)
        if strategy_id != 'all' and strategy_id not in self._scan['evaluated_strategy_ids']:
            raise ValueError('請選擇已實作的策略；目錄待研究項目沒有訊號')
        query = search.strip().casefold()
        if len(query) > 64:
            raise ValueError('搜尋字串最多 64 字')
        rows = []
        for stock in day['stocks']:
            if query and query not in (stock['stock_id']+' '+stock['name']).casefold():
                continue
            matches = []
            for sid, r in stock['results'].items():
                if strategy_id == 'all' and self._catalog[sid]['kind'] != 'entry':
                    continue
                if strategy_id != 'all' and sid != strategy_id:
                    continue
                if r['status'] != 'matched' or first_only and r['first_signal'] is not True:
                    continue
                matches.append(dict(strategy_id=sid, name=self._catalog[sid]['name'], **r))
            if not matches:
                continue
            rows.append(dict(stock_id=stock['stock_id'], name=stock['name'], regime=stock['regime'],
                matched_count=len(matches), first_count=sum(m['first_signal'] is True for m in matches),
                strategy_ids=[m['strategy_id'] for m in matches], matches=matches,
                assessment=self._assessment(stock)))
        rows.sort(key=lambda r: (-r['first_count'], -r['matched_count'], r['stock_id']))
        return dict(date=day['date'], strategy_id=strategy_id, first_only=first_only,
                    total=len(rows), rows=rows, live_qualified=False,
                    sort_policy='first_signal_count_then_match_count_then_stock_id_not_return_ranking')

    def strategies(self):
        self._load()
        return dict(catalog=list(self._catalog.values()),
                    counts=dict(Counter(r['status'] for r in self._catalog.values())),
                    global_completeness_claim=False, live_qualified=False)

    def stock(self, stock_id, selected=None, sessions=120):
        self._load()
        selected = _iso(selected) if selected else max(self._days)
        if pd.Timestamp(selected) not in self._calendar or selected > self._scan['source_end']:
            raise ValueError('圖表日期須是封存來源內已觀測的交易日')
        day = self._days.get(selected, dict(date=selected, stocks=[]))
        if not re.fullmatch(r'[1-9][0-9]{3}|0050', stock_id) or stock_id not in self._ids:
            raise ValueError('查無這個研究股票代碼')
        if type(sessions) is not int or not 20 <= sessions <= 500:
            raise ValueError('圖表範圍須為 20 到 500 個交易日')
        key = stock_id, day['date'], sessions
        with self._lock:
            if key in self._chart_cache:
                return self._chart_cache[key]
        end = pd.Timestamp(day['date'])
        calendar = self._calendar[self._calendar <= end][-sessions:]
        # Read one previous observation for the same source-return check used
        # by the scanner, then remove it from the displayed as-of window.
        pos = self._calendar.get_loc(calendar[0])
        first = self._calendar[max(0, pos-1)]
        matrices = {}
        for filename in ('close-official.parquet', 'close-quality.parquet', 'eligibility.parquet'):
            matrices[filename] = pd.read_parquet(self.bundle/filename,
                columns=['date', stock_id], filters=[('date', '>=', first), ('date', '<=', end)]
            ).set_index('date')[stock_id]
        a, b = matrices['close-official.parquet'], matrices['close-quality.parquet']
        eligible = matrices['eligibility.parquet']
        q = pd.read_parquet(self.bundle/'quotes-unmasked.parquet',
            columns=['date', 'stock_id', 'open', 'high', 'low', 'close', 'volume'],
            filters=[('stock_id', '=', stock_id), ('date', '>=', calendar[0]), ('date', '<=', end)])
        if q.date.duplicated().any():
            raise EvidenceError('股票圖表有重複日期，停止顯示')
        q = q.set_index('date').reindex(calendar)
        valid = np.isfinite(q[['open', 'high', 'low', 'close', 'volume']]).all(axis=1)
        valid &= q[['open', 'high', 'low', 'close', 'volume']].gt(0).all(axis=1)
        valid &= q.volume.eq(np.floor(q.volume))
        valid &= q.low.le(q[['open', 'close']].min(axis=1)) & q.high.ge(q[['open', 'close']].max(axis=1))
        valid &= a.reindex(calendar).gt(0) & b.reindex(calendar).gt(0)
        known = a.gt(0) & b.gt(0) & a.shift(1).gt(0) & b.shift(1).gt(0)
        disagree = (a/a.shift(1)-b/b.shift(1)).abs().gt(.005) & known
        valid &= ~disagree.reindex(calendar).fillna(False)
        factor = a.reindex(calendar)/q.close
        candles = [dict(date=str(d.date()), **{k: float(q.at[d, k]*factor.loc[d])
                   for k in ('open', 'high', 'low', 'close')}, volume=int(q.at[d, 'volume']),
                   eligible=None if pd.isna(eligible.get(d)) else bool(eligible.loc[d]))
                   for d in calendar if valid.loc[d]]
        results, marker_map = [], {}
        found = next((s for s in day['stocks'] if s['stock_id'] == stock_id), None)
        self._load_study()
        # This projection reads only event identity; future outcome/exit columns
        # are never read into the as-of chart response.
        historical_events = pd.read_parquet(self._verify(self._study_events_descriptor),
            columns=['strategy_id', 'signal_date', 'stock_id'], filters=[
                ('stock_id', '=', stock_id), ('horizon', '=', 5),
                ('signal_date', '>=', str(calendar[0].date())), ('signal_date', '<=', selected)])
        for event in historical_events.to_dict('records'):
            sid = event['strategy_id']
            marker_map[(event['signal_date'], sid)] = dict(date=event['signal_date'],
                strategy_id=sid, name=self._catalog[sid]['name'], first_signal=True,
                coverage_type='known_first_event')
        for d, item in self._days.items():
            if str(calendar[0].date()) <= d <= day['date']:
                item_stock = next((s for s in item['stocks'] if s['stock_id'] == stock_id), None)
                if item_stock:
                    for sid, r in item_stock['results'].items():
                        if self._catalog[sid]['kind'] == 'entry' and r['status'] == 'matched':
                            marker_map[(d, sid)] = dict(date=d, strategy_id=sid, name=self._catalog[sid]['name'],
                                first_signal=r['first_signal'], coverage_type='daily_snapshot')
        if found:
            results = [dict(strategy_id=sid, name=self._catalog[sid]['name'],
                            kind=self._catalog[sid]['kind'], **r) for sid, r in found['results'].items()]
        result = dict(stock_id=stock_id, name=self._names.get(stock_id, stock_id), date=day['date'],
            price_basis='adjusted', price_basis_note='依每日還原收盤與原始收盤比例還原 OHLC；不是當時可掛單價格。',
            candles=candles, markers=[marker_map[k] for k in sorted(marker_map)], results=results,
            assessment=self._assessment(found) if found else (
                dict(status='benchmark_only', label='0050 僅作比較基準', action='none') if stock_id == '0050' else
                dict(status='historical_reference', label='歷史走勢與已封存首日訊號；沒有當日完整掃描快照', action='none')),
            coverage=dict(start=str(calendar[0].date()), end=day['date'],
                missing_sessions=int((~valid).sum()), requested_sessions=len(calendar),
                missing_dates=[str(d.date()) for d in calendar if not valid.loc[d]],
                marker_start=self._study['start'], marker_end=day['date'],
                daily_snapshot_available=found is not None,
                marker_scope='2024 起已知首日事件＋最近五日每日訊號；早期無標記不代表無訊號或資料完整'),
            source_hashes=self.source_hashes, live_qualified=False)
        self._unchanged()
        with self._lock:
            if len(self._chart_cache) >= 64:
                self._chart_cache.pop(next(iter(self._chart_cache)))
            self._chart_cache[key] = result
        return result

    def studies(self):
        self._load_study()
        return self._study

    def research(self):
        path = self._path(RALLY_REPORT)
        if not path.exists():
            return dict(status='not_available', message='飆股啟動前訊號交叉研究尚未完成封存，未提供推測數字。')
        sidecar = path.with_suffix('.sha256')
        if not sidecar.exists():
            raise EvidenceError('飆股研究缺少指紋，暫不顯示')
        report = self._read(dict(path=RALLY_REPORT, sha256=sidecar.read_text().strip()))
        if report.get('schema') != 'rally_precursor_study_v1' or report.get('live_qualified') is not False:
            raise EvidenceError('飆股研究格式不正確')
        self._load()
        if report.get('source_provenance', {}).get('source_hashes', {}).get('manifest.json') != self.source_hashes['manifest']:
            raise EvidenceError('飆股研究與工作台封存來源不一致')
        self._verify(report['all_episodes_artifact'])
        return dict(status='ready', **report)

    def research_episodes(self, stock_id=None, year=None, horizon=None, offset=0, limit=50):
        if (stock_id is not None and not re.fullmatch(r'[1-9][0-9]{3}', stock_id)
                or year is not None and not re.fullmatch(r'20[0-9]{2}', year)
                or horizon is not None and horizon not in (20, 60)
                or not 0 <= offset <= 1000000 or not 1 <= limit <= 200):
            raise ValueError('飆股案例篩選條件錯誤')
        report = self.research()
        if report['status'] != 'ready':
            return dict(report, rows=[], total=0, offset=offset, limit=limit)
        with self._lock:
            if self._rally_cases is None:
                rows = self._read(report['all_episodes_artifact'])
                if not isinstance(rows, list) or len(rows) != report['all_episodes_artifact']['count']:
                    raise EvidenceError('飆股研究案例數與封存報告不一致')
                self._rally_cases = rows
        rows = [r for r in self._rally_cases if (stock_id is None or r['stock_id'] == stock_id)
                and (year is None or r['anchor_date'].startswith(year))
                and (horizon is None or r['horizon'] == horizon)]
        return dict(status='ready', rows=rows[offset:offset+limit], total=len(rows), offset=offset, limit=limit,
                    selection_policy='全部封存案例依原順序；不依漲幅選樣。案例起點為事後標記，不能回灌買訊。',
                    live_qualified=False)

    def _event_frame(self, params):
        self._load_study()
        sid, start, end, horizon = (params[k] for k in ('strategy_id', 'start', 'end', 'horizon'))
        _iso(start); _iso(end)
        if sid not in self._study['strategies']:
            raise ValueError('訊號研究只支援已封存的 29 種進場規則')
        if type(horizon) is not int or horizon not in self._study['horizons']:
            raise ValueError('持有期間只支援已封存的 5、20、60 個交易日')
        if not self._study['start'] <= start <= end <= self._study['end']:
            raise ValueError('訊號研究日期須在 '+self._study['start']+' 到 '+self._study['end'])
        frame = pd.read_parquet(self._verify(self._study_events_descriptor), filters=[
            ('strategy_id', '=', sid), ('horizon', '=', horizon),
            ('signal_date', '>=', start), ('signal_date', '<=', end)])
        self._unchanged()
        return frame.sort_values(['signal_date', 'stock_id'], kind='stable')

    def backtest(self, params):
        if params['mode'] == 'account_replay':
            from app import workbench_jobs
            request = workbench_jobs.WorkRequest(kind='verified_backtest',
                replay_mode=params.get('replay_mode', 'daily'),
                replay_policy=params.get('replay_policy', 'mixed'),
                replay_stress=params.get('replay_stress', 'control'),
                replay_preflight=params.get('preflight', False), replay_fresh=params.get('fresh', False))
            job = workbench_jobs.submit(request)
            return dict(job_id='account_'+job['job_id'], status=job['status'],
                        message=job['message'], mode='account_replay', scope=self._account_scope())
        if params['mode'] != 'signal_study':
            raise ValueError('不支援的研究方式')
        self._load_study()
        canonical = {k: params[k] for k in ('mode', 'strategy_id', 'start', 'end', 'horizon')}
        identity = dict(params=canonical, service=SERVICE_VERSION, study_sha256=self._study_hash,
                        events_sha256=self._study_events_descriptor['sha256'])
        job_id = hashlib.sha256(_json_bytes(identity)).hexdigest()[:32]
        path = self.jobs_dir/(job_id+'.json')
        if path.exists():
            job = self.get_backtest(job_id)
            return dict(job, cache_hit=True)
        started = time.perf_counter()
        frame = self._event_frame(canonical)
        from skills.strategy_scanner.outcomes import _stats
        definition = next(s for s in self._study['strategy_definitions'] if s['id'] == canonical['strategy_id'])
        years = [dict(year=key, **_stats(group)) for key, group in frame.groupby(frame.signal_date.str[:4])]
        months = [dict(month=key, **_stats(group)) for key, group in frame.groupby(frame.signal_date.str[:7])]
        result = dict(**canonical, name=definition['name'], stats=_stats(frame), yearly=years, monthly=months,
            events=_clean(frame.head(200).to_dict('records')), event_count=len(frame), events_preview_limit=200,
            signal_coverage=dict(first_event_date=frame.signal_date.min() if len(frame) else None,
                last_event_date=frame.signal_date.max() if len(frame) else None,
                full_study_counts=self._study['signal_counts'][canonical['strategy_id']],
                note='無訊號不等於資料完整；POC 歷史覆蓋較短。未知股票日另列，不能視為策略未成立。'),
            costs=self._study['costs'], signal_policy=self._study['signal_policy'],
            entry_price=self._study['entry_price'], exit_price=self._study['exit_price'],
            observation_end=self._study['end'], study_type='signal_study', account_independent=True,
            cumulative_return=None, max_drawdown=None, live_qualified=False,
            explanation='以選定訊號日期篩選封存逐筆事件，重新統計；每筆持有期可超過訊號篩選迄日，但不超過觀測資料迄日。',
            limitations=['事件互相重疊；平均報酬不是帳戶複利報酬。',
                '開盤與收盤還原價僅為研究成交假設；未驗證逐筆容量、零股撮合與最低手續費。',
                '這段歷史已研究過；未完成多重比較校正，不能以排行選定實戰策略。'],
            source_hashes=identity)
        job = dict(job_id=job_id, status='completed', mode='signal_study', result=result,
            cache_hit=False, elapsed_seconds=round(time.perf_counter()-started, 3),
            created_at=datetime.now(timezone.utc).isoformat())
        raw = _json_bytes(job)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            temporary = path.with_suffix('.'+str(os.getpid())+'.tmp')
            temporary.write_bytes(raw)
            os.replace(temporary, path)
            path.with_suffix('.sha256').write_text(hashlib.sha256(raw).hexdigest()+'\n')
        return _clean(job)

    def get_backtest(self, job_id):
        if re.fullmatch(r'account_[a-f0-9]{32}', job_id):
            from app import workbench_jobs
            from app.backtest_tool_ui import load_report
            raw_id = job_id.removeprefix('account_')
            job = workbench_jobs.read_job(raw_id)
            if job.get('request', {}).get('kind') != 'verified_backtest':
                raise ValueError('這不是原封存帳戶回測工作')
            response = dict(job_id=job_id, status=job['status'], mode='account_replay',
                message=job.get('message'), elapsed_seconds=job.get('elapsed_seconds'),
                scope=self._account_scope())
            if job['status'] == 'completed':
                expected = workbench_jobs.JOBS_DIR/(raw_id+'.result.json')
                if Path(job['result_path']).resolve() != expected.resolve():
                    raise EvidenceError('帳戶回測結果路徑不一致')
                report = load_report(str(expected), root=self.root, expected_sha256=job['result_sha256'])
                response['result'] = report
            return response
        if not re.fullmatch(r'[a-f0-9]{32}', job_id):
            raise ValueError('工作識別碼錯誤')
        path = self.jobs_dir/(job_id+'.json')
        if not path.exists() or not path.with_suffix('.sha256').exists():
            raise ValueError('找不到已完成的研究工作')
        descriptor = dict(path=str(path.relative_to(self.root)), sha256=path.with_suffix('.sha256').read_text().strip())
        job = self._read(descriptor)
        self._load_study()
        source = job['result']['source_hashes']
        if source['study_sha256'] != self._study_hash or source['events_sha256'] != self._study_events_descriptor['sha256']:
            raise EvidenceError('研究工作與目前封存來源不一致')
        return job

    def backtest_events(self, job_id, offset=0, limit=200):
        if not 0 <= offset <= 1000000 or not 1 <= limit <= 2000:
            raise ValueError('事件分頁範圍無效')
        job = self.get_backtest(job_id)
        if job['mode'] != 'signal_study':
            raise ValueError('帳戶明細由已核對的案例報告提供')
        params = {k: job['result'][k] for k in ('mode', 'strategy_id', 'start', 'end', 'horizon')}
        frame = self._event_frame(params)
        return dict(job_id=job_id, total=len(frame), offset=offset, limit=limit,
                    rows=_clean(frame.iloc[offset:offset+limit].to_dict('records')))


@lru_cache(maxsize=1)
def get_terminal():
    return ResearchTerminal()
