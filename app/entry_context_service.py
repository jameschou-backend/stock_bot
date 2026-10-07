"""Sealed, account-independent entry filters over first parent signal events.

This supplementary snapshot has its own calendar and source fingerprint. It does
not replace the ordinary scanner or supply outcome data to daily views.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from app.research_terminal_service import EvidenceError, ResearchTerminal, _iso

PUBLICATION = 'artifacts/forward_simulation/entry_context_terminal_20261007.json'
PUBLICATION_SHA256 = '872d189be103546cc99d8976329377680a77a8a1026103d43a1c565a452cc0ba'
STRATEGY_IDS = ('entry_contraction_narrow', 'entry_peer_narrow')
_RULES = {
    STRATEGY_IDS[0]: ('前期價格收斂＋市場廣度偏弱', 'contraction_and_narrow', 'contraction'),
    STRATEGY_IDS[1]: ('價格同儕強勢＋市場廣度偏弱', 'peer_and_narrow', 'peer_breadth'),
}
PARENT_DESCRIPTION = ('400 日價量高點：收盤 ≥ 含當日 400 日最高收盤的 99.8%，'
    '成交量 ≥ 含當日 10 日最大量，20 日平均估算成交值 ≥ 5 億元；只取已知首日事件。')
_SOURCE_FILES = ('close-official.parquet', 'close-quality.parquet', 'eligibility.parquet',
                 'quotes-unmasked.parquet', 'companies.parquet', 'raw-close.parquet', 'raw-volume.parquet')


def _conjunction(left, right):
    return None if left is None or right is None else left and right


def _number(value, *, fraction=False):
    return (type(value) in (int, float) and math.isfinite(value) and value >= 0
            and (not fraction or value <= 1))


def _flag(value):
    return value is None or type(value) is bool


class EntryContextProvider:
    def __init__(self, root, descriptor=None):
        self.evidence = ResearchTerminal(root)
        self.descriptor = descriptor or dict(path=PUBLICATION, sha256=PUBLICATION_SHA256)
        self._publication = None
        self._charts = {}

    def available(self):
        self.evidence._unchanged()
        return self.evidence._path(self.descriptor['path']).exists()

    def _load(self):
        with self.evidence._lock:
            self.evidence._unchanged()
            if self._publication is not None:
                return
            path = self.evidence._path(self.descriptor['path'])
            sidecar = path.with_suffix('.sha256')
            try:
                checksum = sidecar.read_text().strip()
            except OSError as exc:
                raise EvidenceError('進場情境研究缺少封存指紋') from exc
            if checksum != self.descriptor.get('sha256'):
                raise EvidenceError('進場情境研究指紋與指定版本不一致')
            # Pin the sidecar too, so a loaded cache cannot hide later changes.
            import hashlib
            self.evidence._verify(dict(path=str(sidecar.relative_to(self.evidence.root)),
                sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest()))
            publication = self.evidence._read(self.descriptor)
            if (publication.get('schema') != 'entry_context_terminal_v1'
                    or publication.get('live_qualified') is not False):
                raise EvidenceError('不支援的進場情境研究版本')
            report_descriptor = publication['artifacts']['report']
            bundle_descriptor = publication['artifacts']['bundle']
            report = self.evidence._read(report_descriptor)
            manifest = self.evidence._read(bundle_descriptor)
            refs = report.get('source_sha256', {})
            if (report.get('schema') != 'entry60_current_signal_check_v1'
                    or report.get('parent') != 'legacy_course_breakout'
                    or report.get('live_qualified') is not False
                    or report.get('account_backtest') is not False
                    or report.get('start') != publication['start']
                    or report.get('end') != publication['end']
                    or report['end'] != manifest['end']
                    or refs.get(bundle_descriptor['path']) != bundle_descriptor['sha256']):
                raise EvidenceError('進場情境研究與行情封存來源不一致')
            for name, sha in refs.items():
                self.evidence._verify(dict(path=name, sha256=sha))
            for name, sha in publication.get('source_sha256', {}).items():
                self.evidence._verify(dict(path=name, sha256=sha))
            self.bundle = self.evidence._path(str(Path(bundle_descriptor['path']).parent))
            for filename in _SOURCE_FILES:
                name = str((self.bundle/filename).relative_to(self.evidence.root))
                expected = manifest['files_sha256'][filename]
                if name in refs and refs[name] != expected:
                    raise EvidenceError('進場情境行情矩陣指紋不一致')
                self.evidence._verify(dict(path=name, sha256=expected))
            self.calendar = pd.DatetimeIndex(pd.read_parquet(self.bundle/'close-official.parquet', columns=['date']).date)
            dates = [str(d.date()) for d in self.calendar
                     if publication['start'] <= str(d.date()) <= publication['end']]
            if publication['dates'] != dates or not dates:
                raise EvidenceError('進場情境交易日曆與行情不一致')
            context = publication['day_context']
            if [x['date'] for x in context] != dates:
                raise EvidenceError('進場情境逐日資料不完整')
            self.days = {x['date']: x for x in context}
            self.events = {d: [] for d in dates}
            identities = set()
            for event in report['all_first_signals']:
                identity = event['signal_date'], event['stock_id']
                if (identity in identities or identity[0] not in self.events
                        or not re.fullmatch(r'[1-9][0-9]{3}', identity[1])):
                    raise EvidenceError('進場情境首日事件重複或日期不合法')
                identities.add(identity)
                for key in ('contraction', 'peer_breadth', 'market_narrow', 'contraction_and_narrow', 'peer_and_narrow'):
                    if not _flag(event.get(key)):
                        raise EvidenceError('進場情境條件必須保留是、否、未知三種狀態')
                ratio, peer = event.get('contraction_ratio'), event.get('peer_breadth_value')
                if ratio is not None and not _number(ratio) or peer is not None and not _number(peer, fraction=True):
                    raise EvidenceError('進場情境數值不合法')
                contraction = None if ratio is None else ratio <= .75
                peer_flag = None if peer is None or event.get('peer_issue') else peer >= .5
                if event['contraction'] is not contraction or event['peer_breadth'] is not peer_flag:
                    raise EvidenceError('進場情境數值與條件門檻不一致')
                cutoff = event.get('group_cutoff_date')
                if cutoff is None or _iso(cutoff) >= identity[0][:7]+'-01':
                    raise EvidenceError('價格同儕必須使用訊號月份之前的分組')
                for _, field, condition in _RULES.values():
                    if event[field] is not _conjunction(event[condition], event['market_narrow']):
                        raise EvidenceError('進場情境條件與嚴格未知規則不一致')
                self.events[identity[0]].append(event)
            for d, context in self.days.items():
                if context['parent_candidates'] != len(self.events[d]):
                    raise EvidenceError('進場情境原始候選數與首日事件不一致')
                if not _flag(context.get('market_narrow')):
                    raise EvidenceError('市場廣度狀態不合法')
                value, coverage = context.get('market_breadth_value'), context.get('market_breadth_coverage')
                if any(x is not None and not _number(x, fraction=True) for x in (value, coverage)):
                    raise EvidenceError('市場廣度數值不合法')
                known = value is not None and coverage is not None and coverage >= .8 and context['valid60_stocks'] >= 500
                narrow = value < .5 if known else None
                if context['market_narrow'] is not narrow:
                    raise EvidenceError('市場廣度數值與偏弱門檻不一致')
                for event in self.events[d]:
                    if event['market_narrow'] is not context['market_narrow']:
                        raise EvidenceError('逐日市場廣度條件與候選不一致')
                    for key in ('market_breadth_value', 'market_breadth_coverage'):
                        left, right = event.get(key), context.get(key)
                        if ((left is None) != (right is None)
                                or left is not None and not math.isclose(left, right, abs_tol=1e-8)):
                            raise EvidenceError('逐日市場廣度與候選數值不一致')
            companies = pd.read_parquet(self.bundle/'companies.parquet')
            self.names = dict(zip(companies.stock_id, companies.name))
            self.names['0050'] = '元大台灣50'
            self.ids = set(pq.ParquetFile(self.bundle/'close-official.parquet').schema_arrow.names)-{'date'}
            self.hashes = dict(publication=self.descriptor['sha256'], report=report_descriptor['sha256'],
                               manifest=bundle_descriptor['sha256'])
            self._publication = publication
            self.evidence._unchanged()

    def metadata(self):
        if not self.available():
            return dict(status='not_available', strategy_ids=list(STRATEGY_IDS), dates=[],
                        message='進場情境研究尚未封存；原每日掃描維持自己的資料日期。')
        self._load()
        return dict(status='ready', strategy_ids=list(STRATEGY_IDS), dates=list(self.days),
                    start=self._publication['start'], end=self._publication['end'],
                    source_end=self._publication['end'], default_date=self._publication['end'],
                    default_strategy_id=STRATEGY_IDS[0], signal_only=True, account_independent=True,
                    scope_note='只評估400 日價量高點的已知首日事件；全部策略仍使用一般掃描來源。',
                    source_hashes=self.hashes, live_qualified=False)

    def catalog(self):
        if not self.available():
            return []
        self._load()
        return [dict(id=sid, name=name, kind='entry', status='active', family='entry_context',
                     version='entry_context_v1', signal_only=True, source_end=self._publication['end'],
                     start=self._publication['start'], end=self._publication['end'],
                     dates=list(self.days), required_data=['收盤價量', '市場廣度', '前月價格同儕'],
                     explanation=PARENT_DESCRIPTION + ('搭配前期區間收斂 ≤ 0.75 與市場廣度 < 50%。'
                         if sid == STRATEGY_IDS[0] else
                         '搭配至少 50% 價格同儕站上 20 日線與市場廣度 < 50%。'),
                     limitations=['獨立進場訊號，未接入帳戶回測；歷史勝率不代表下一筆勝率。'])
                for sid, (name, _, _) in _RULES.items()]

    def _day(self, selected):
        self._load()
        selected = _iso(selected) if selected else self._publication['end']
        if selected not in self.days:
            raise ValueError('此日期不在進場情境研究封存交易日內')
        return selected

    def _check(self, event, strategy_id):
        name, field, condition = _RULES[strategy_id]
        flag = event[field]
        status = 'unknown' if flag is None else ('matched' if flag else 'not_matched')
        reasons = ['400 日價量高點的已知首日訊號。']
        breadth = event['market_narrow']
        reasons.append('市場廣度資料不足，無法判定。' if breadth is None else
            ('市場廣度低於 50%，符合偏弱條件。' if breadth else '市場廣度未低於 50%，不符合偏弱條件。'))
        value = event[condition]
        if condition == 'contraction':
            reasons.append('前期收斂資料不足。' if value is None else
                ('前期價格區間收斂符合 ≤ 0.75。' if value else '前期價格區間未收斂至 ≤ 0.75。'))
        else:
            reasons.append('價格同儕資料不足，無法判定。' if value is None else
                ('至少 50% 價格同儕站上 20 日線。' if value else '不足 50% 價格同儕站上 20 日線。'))
        metrics = {k: event.get(k) for k in ('contraction_ratio', 'contraction', 'market_breadth_value',
            'market_breadth_coverage', 'market_narrow', 'peer_breadth_value', 'peer_breadth', 'peer_issue', 'group_cutoff_date', 'peer_ids')}
        return dict(stock_id=event['stock_id'], name=event.get('name') or self.names.get(event['stock_id'], event['stock_id']),
                    status=status, strategy_id=strategy_id, strategy_name=name, reasons=reasons, metrics=metrics,
                    first_signal=True if flag is True else None, regime_fit=None)

    def signals(self, selected=None, strategy_id=STRATEGY_IDS[0], first_only=False, search=''):
        if strategy_id not in STRATEGY_IDS:
            raise ValueError('不支援的進場情境策略')
        selected = self._day(selected)
        query = search.strip().casefold()
        if len(query) > 64:
            raise ValueError('搜尋字串最多 64 字')
        checks = [self._check(x, strategy_id) for x in self.events[selected]]
        rows = []
        for check in checks:
            if check['status'] != 'matched' or query and query not in (check['stock_id']+' '+check['name']).casefold():
                continue
            match = {k: check[k] for k in ('strategy_id', 'status', 'first_signal', 'reasons', 'metrics', 'regime_fit')}
            match['name'] = check['strategy_name']
            rows.append(dict(stock_id=check['stock_id'], name=check['name'], regime='context_research',
                matched_count=1, first_count=1, strategy_ids=[strategy_id], matches=[match],
                assessment=dict(status='research_candidate', label='符合研究條件，列入觀察',
                    action='watch', reasons=check['reasons'], live_qualified=False,
                    earliest_execution='next_market_session', unknown_entry_rules=[])))
        counts = Counter(x['status'] for x in checks)
        day = self.days[selected]
        breadth = day.get('market_narrow')
        explanation = ('市場廣度資料不足，不能判定偏弱。' if breadth is None else
            ('市場廣度符合偏弱條件，仍須原始首日與個股條件同時成立。' if breadth else
             '市場廣度未低於 50%；個股原始訊號不代表通過本組條件。'))
        if not checks:
            explanation += ' 當天沒有400 日價量高點的已知首日候選；並非把缺資料股票當成未符合。'
        return dict(date=selected, source_end=self._publication['end'], strategy_id=strategy_id,
            first_only=first_only, total=len(rows), rows=sorted(rows, key=lambda x: x['stock_id']),
            context_summary=dict(market_breadth_value=day.get('market_breadth_value'),
                market_breadth_coverage=day.get('market_breadth_coverage'), market_breadth_threshold=.5,
                market_narrow=breadth, valid60_stocks=day.get('valid60_stocks'),
                eligible_stocks=day.get('eligible_stocks'), parent_candidates=len(checks), matched=counts['matched'],
                unknown=counts['unknown'], not_matched=counts['not_matched'], explanation=explanation,
                checks=checks, scope='known_first_parent_events_only'),
            signal_only=True, account_independent=True, live_qualified=False,
            sort_policy='stock_id_not_return_ranking', source_hashes=self.hashes)

    def stock(self, stock_id, selected=None, sessions=120, strategy_id=STRATEGY_IDS[0]):
        selected = self._day(selected)
        if strategy_id not in STRATEGY_IDS:
            raise ValueError('不支援的進場情境策略')
        if not re.fullmatch(r'[1-9][0-9]{3}|0050', stock_id) or stock_id not in self.ids:
            raise ValueError('查無這個研究股票代碼')
        if type(sessions) is not int or not 20 <= sessions <= 500:
            raise ValueError('圖表範圍須為 20 到 500 個交易日')
        key = stock_id, selected, sessions, strategy_id
        if key in self._charts:
            return self._charts[key]
        end = pd.Timestamp(selected)
        calendar = self.calendar[self.calendar <= end][-sessions:]
        first = self.calendar[max(0, self.calendar.get_loc(calendar[0])-1)]
        matrices = {name: pd.read_parquet(self.bundle/name, columns=['date', stock_id],
            filters=[('date', '>=', first), ('date', '<=', end)]).set_index('date')[stock_id]
            for name in ('close-official.parquet', 'close-quality.parquet', 'eligibility.parquet')}
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
        markers = [dict(date=d, strategy_id=strategy_id, name=_RULES[strategy_id][0], first_signal=True,
                       coverage_type='known_first_parent_event')
                   for d, events in self.events.items() if str(calendar[0].date()) <= d <= selected
                   for e in events if e['stock_id'] == stock_id and e[_RULES[strategy_id][1]] is True]
        event = next((e for e in self.events[selected] if e['stock_id'] == stock_id), None)
        check = self._check(event, strategy_id) if event else None
        results = [dict(strategy_id=strategy_id, name=check['strategy_name'], kind='entry',
                        **{k: check[k] for k in ('status', 'first_signal', 'reasons', 'metrics', 'regime_fit')})] if check else []
        status = check['status'] if check else 'no_parent_first_event'
        result = dict(stock_id=stock_id, name=self.names.get(stock_id, stock_id), date=selected,
            strategy_id=strategy_id, source_end=self._publication['end'], price_basis='adjusted',
            price_basis_note='依每日還原收盤與原始收盤比例還原 OHLC；不是當時可掛單價格。',
            candles=candles, markers=markers, results=results,
            assessment=dict(status=status, label={'matched': '符合研究條件，列入觀察',
                'unknown': '資料不足，無法判斷', 'not_matched': '當日條件未成立',
                'no_parent_first_event': '當日沒有400 日價量高點的已知首日事件'}[status],
                action='watch' if status == 'matched' else 'wait', live_qualified=False),
            coverage=dict(start=str(calendar[0].date()), end=selected, missing_sessions=int((~valid).sum()),
                requested_sessions=len(calendar), missing_dates=[str(d.date()) for d in calendar if not valid.loc[d]],
                marker_start=self._publication['start'], marker_end=selected, daily_snapshot_available=bool(event),
                marker_scope='所選進場情境規則的已知首日事件；沒有原始首日事件不代表全市場已評估。'),
            source_hashes=self.hashes, signal_only=True, live_qualified=False)
        self.evidence._unchanged()
        if len(self._charts) >= 64:
            self._charts.pop(next(iter(self._charts)))
        self._charts[key] = result
        return result
