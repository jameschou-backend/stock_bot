"""Bounded FinMind extension preserving the sealed account's financial history."""
from copy import deepcopy
from datetime import date, datetime, timezone
import json
import math
from pathlib import Path
import re

import pandas as pd

from app.file_lock import file_lock
from skills.frozen_dividend_copy import ensure_dividend_copy
from skills.replay_market_feeds import ReplayDataUnavailable
from scripts.prepare_poc_latest_inputs import bind, digest, read, require, write

ANCHOR = '2026-09-09'
END = '2026-10-02'
START = '2018-01-01'
DIRECTORY = '.cache/poc-latest-20261003/execution-v1'
DATASETS = ('TaiwanStockDividend', 'TaiwanStockPriceLimit')
REPORTS = {
    '.cache/red-volume-exit-20261003/anchors-v2/report.json': '4d670409c617d2d562f60630dcafaef40c775598878a2b920415ba10c0a2aef2',
    '.cache/red-volume-exit-20261003/full-v1/report.json': '2b33236460533746c4fab4f2e65a9dd3df04efb2ab0bd5031b61a1c4d8e37a55',
}
# This is the exact sealed runner's precedence, not filesystem discovery order.
CACHE_ORDER = (
    '.cache/market-input-repair-20261002/execution-v1',
    '.cache/three-black-20261001/execution-v1', '.cache/drawdown-control-20261001/execution-v1',
    '.cache/liquidity-universe-20261001/execution-v1', '.cache/waiting-exit-20260930/execution-v1',
    '.cache/entry-filters-20260930/execution-v1', '.cache/holding-release-20260929/execution-v1',
    '.cache/stock-universe-2019-20260929/execution-v1', '.cache/stock-universe-five-20260929/execution-v1',
    '.cache/liquidity-account-20261001/execution-v1', '.cache/allocation-2019-20260929/execution-v1',
    '.cache/candidate-quality-20260929/execution-v1', '.cache/partial-risk-2019-20260929/execution-v1',
    '.cache/rotation-2024-20260929/execution-v1',
)
CASH_AMOUNTS = ('CashEarningsDistribution', 'CashStatutorySurplus')
STOCK_AMOUNTS = ('StockEarningsDistribution', 'StockStatutorySurplus')
CAPITAL_AMOUNTS = ('TotalNumberOfCashCapitalIncrease', 'CashIncreaseSubscriptionRate',
                   'CashIncreaseSubscriptionpRrice')


def optional_day(value):
    if value is None or pd.isna(value) or str(value).strip() in ('', '0', '0000-00-00'):
        return None
    result = str(pd.Timestamp(value).date())
    require(result == str(value)[:10], 'Malformed dividend ex-date')
    return result


def amount(row, columns):
    total = 0.
    for column in columns:
        value = row.get(column, 0.)
        value = 0. if value is None or pd.isna(value) or value == '' else float(value)
        require(math.isfinite(value) and value >= 0, 'Invalid dividend amount')
        total += value
    return total


def dividend_components(frame):
    """Split independent cash/share ex-dates; a mixed row is never one event."""
    components = []
    for source in frame.to_dict('records'):
        for kind, key, columns in (('cash', 'CashExDividendTradingDate', CASH_AMOUNTS),
                                   ('stock', 'StockExDividendTradingDate', STOCK_AMOUNTS)):
            ex = optional_day(source.get(key))
            value = amount(source, columns)
            capital = amount(source, CAPITAL_AMOUNTS[:1]) if kind == 'stock' else 0.
            if ex is None:
                # Undated announcements do not create dated account rights.
                continue
            row = deepcopy(source)
            if kind == 'cash':
                row['StockExDividendTradingDate'] = ''
                for column in (*STOCK_AMOUNTS, *CAPITAL_AMOUNTS):
                    if column in row:
                        row[column] = 0.
            else:
                row['CashExDividendTradingDate'] = ''
                if 'CashDividendPaymentDate' in row:
                    row['CashDividendPaymentDate'] = ''
                for column in CASH_AMOUNTS:
                    if column in row:
                        row[column] = 0.
            components.append(dict(kind=kind, ex=ex, amount=value, capital=capital, row=row))
    return components


def economic_components(frame, *, through=ANCHOR):
    """Accounting-relevant historical fields, independent of split row shape."""
    values = []
    for component in dividend_components(frame):
        if component['ex'] <= through:
            row = component['row']
            values.append((component['kind'], component['ex'], component['amount'],
                component['capital'], optional_day(row.get('CashDividendPaymentDate'))
                if component['kind'] == 'cash' else None,
                row.get('AnnouncementDate') or None))
    return sorted(set(values), key=lambda row: tuple('' if x is None else str(x) for x in row))


def merge_dividends(old, fresh):
    """Freeze old rights by ex-date, including payment terms; accept only new rights."""
    if old is None:
        return fresh.copy(deep=True)
    selected = ([p['row'] for p in dividend_components(old) if p['ex'] <= ANCHOR]
                + [p['row'] for p in dividend_components(fresh) if ANCHOR < p['ex'] <= END])
    columns = list(old.columns) + [c for c in fresh.columns if c not in old]
    result = pd.DataFrame(selected, columns=columns)
    require(economic_components(result) == economic_components(old),
            'Historical dividend rights or payment terms changed')
    # Conflicting revisions are data errors, not multiple distributions to add.
    seen = {}
    for component in dividend_components(result):
        if not component['amount'] and not component['capital']:
            continue
        key = (component['kind'], component['ex'])
        row = component['row']
        signature = (component['amount'], component['capital'],
                     optional_day(row.get('CashDividendPaymentDate')) if component['kind'] == 'cash' else None)
        require(key not in seen or seen[key] == signature, 'Conflicting dividend revisions at ' + str(key))
        seen[key] = signature
    return result


def validate_frame(frame, sid, dataset, start, end):
    if frame.empty:
        return
    require({'stock_id', 'date'}.issubset(frame.columns), 'Execution data lacks identity/date')
    require(set(frame.stock_id.astype(str)) == {sid}, 'Execution data has wrong stock identity')
    days = pd.to_datetime(frame.date)
    require(days.between(start, end).all(), 'Execution data outside requested date range')
    if dataset == 'TaiwanStockPriceLimit':
        require(not days.duplicated().any(), 'Duplicate price-limit date')
        require({'limit_down', 'limit_up', 'reference_price'}.issubset(frame.columns), 'Incomplete legal limits')
        for low, high in zip(frame.limit_down, frame.limit_up):
            low, high = float(low), float(high)
            require(math.isfinite(low + high) and 0 <= low <= high and (low == 0) == (high == 0),
                    'Invalid legal limits')
    else:
        require(set((*CASH_AMOUNTS, *STOCK_AMOUNTS, 'CashExDividendTradingDate',
                    'StockExDividendTradingDate', 'CashDividendPaymentDate')).issubset(frame.columns),
                'Incomplete dividend fields')
        dividend_components(frame)


class LatestExecutionData:
    def __init__(self, root, online=False, *, source_refs=None, maximum_requests=400, fetcher=None, config=None):
        self.root = Path(root).resolve()
        self.directory = self.root / DIRECTORY
        self.dividend_directory = self.directory / 'dividends'
        self.online, self.refs, self.loaded = online, {}, {}
        require(type(maximum_requests) is int and 0 <= maximum_requests <= 400, 'Execution budget exceeds preregistration')
        self.maximum, self.fetcher, self.config = maximum_requests, fetcher, config
        self.directory.mkdir(parents=True, exist_ok=True)
        if source_refs is None:
            self.source_refs = {}
            for name, expected in REPORTS.items():
                report = read(bind(self.root / name, self.root, self.refs, expected))
                require(report.get('end') == ANCHOR and report.get('live_qualified') is False,
                        'Execution reference scope changed')
                for source, source_sha in report['source_sha256'].items():
                    require(source not in self.source_refs or self.source_refs[source] == source_sha,
                            'Conflicting sealed execution source')
                    self.source_refs[source] = source_sha
        else:
            self.source_refs = dict(source_refs)

    def _old(self, sid, dataset):
        for folder in CACHE_ORDER:
            name = folder + '/' + sid + '-' + dataset + '.parquet'
            if name not in self.source_refs:
                continue
            path = bind(self.root / name, self.root, self.refs, self.source_refs[name])
            meta_name = str(path.with_suffix('.json').relative_to(self.root))
            require(meta_name in self.source_refs, 'Old execution metadata is not sealed')
            meta = read(bind(self.root / meta_name, self.root, self.refs, self.source_refs[meta_name]))
            require(meta == dict(stock_id=sid, dataset=dataset, start=START, end=ANCHOR,
                                 sha256=self.source_refs[name]), 'Old execution query identity changed')
            frame = pd.read_parquet(path)
            validate_frame(frame, sid, dataset, START, ANCHOR)
            return frame, {name:self.source_refs[name], meta_name:self.source_refs[meta_name]}
        return None, {}

    def _request(self, sid, dataset, start):
        key = sid + '-' + dataset
        query = dict(stock_id=sid, dataset=dataset, start=start, end=END)
        attempt = self.directory / 'attempts' / (key + '.json')
        receipt = self.directory / 'receipts' / (key + '.json')
        raw_path = self.directory / 'raw' / (key + '.parquet')
        if receipt.exists():
            record = read(receipt)
            require(record.get('query') == query, 'Execution receipt query changed')
            if record.get('status') != 'received':
                raise ReplayDataUnavailable('Previous latest execution request failed; no automatic retry: ' + key)
            require(attempt.exists(), 'Received execution data lacks reserved attempt')
            bind(attempt, self.root, self.refs, record['attempt_sha256'])
            require(read(attempt)['query'] == query, 'Execution attempt query changed')
            bind(receipt, self.root, self.refs)
            bind(raw_path, self.root, self.refs, record['raw_sha256'])
            frame = pd.read_parquet(raw_path)
            validate_frame(frame, sid, dataset, start, END)
            return frame
        if not self.online:
            raise ReplayDataUnavailable('Latest execution cache missing: ' + key)
        with file_lock(self.directory / '.request.lock', timeout=0):
            require(not attempt.exists(), 'Previous execution attempt exists; automatic retry prohibited')
            attempt.parent.mkdir(parents=True, exist_ok=True)
            require(len(list(attempt.parent.glob('*.json'))) < self.maximum, 'Persistent execution request budget exhausted')
            reserved = dict(query=query, status='started', started_at=datetime.now(timezone.utc).isoformat(),
                            maximum_requests=self.maximum, retries=0)
            with attempt.open('x') as handle:
                json.dump(reserved, handle, ensure_ascii=False)
        if self.config is None:
            from app.config import load_config
            self.config = load_config()
        fetcher = self.fetcher
        if fetcher is None:
            from app.finmind import fetch_dataset
            fetcher = fetch_dataset
        record = dict(query=query, attempt_sha256=digest(attempt), started_at=reserved['started_at'])
        receipt.parent.mkdir(parents=True, exist_ok=True)
        try:
            frame = fetcher(dataset, date.fromisoformat(start), date.fromisoformat(END), data_id=sid,
                token=self.config.finmind_token, requests_per_hour=min(5400, self.config.finmind_requests_per_hour),
                max_retries=0, timeout=40)
            validate_frame(frame, sid, dataset, start, END)
            require(not raw_path.exists(), 'Latest execution raw cache appeared during request')
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(raw_path, index=False)
            record.update(status='received', rows=len(frame), raw_path=str(raw_path.relative_to(self.root)),
                raw_sha256=digest(raw_path), cache_hit=bool(frame.attrs.get('cache_hit', False)),
                retrieved_at=frame.attrs.get('retrieved_at'))
        except Exception as exc:
            # Arbitrary provider messages can contain credentials. Store only type.
            record.update(status='failed', error_type=type(exc).__name__)
            write(receipt, record)
            bind(attempt, self.root, self.refs)
            bind(receipt, self.root, self.refs)
            raise ReplayDataUnavailable('Latest execution acquisition failed: ' + key + ' ' + type(exc).__name__) from None
        write(receipt, record)
        bind(attempt, self.root, self.refs, record['attempt_sha256'])
        bind(receipt, self.root, self.refs)
        bind(raw_path, self.root, self.refs, record['raw_sha256'])
        return frame

    def finmind(self, sid, dataset):
        require(isinstance(sid, str) and re.fullmatch(r'\d{4}', sid) is not None and dataset in DATASETS,
                'Unplanned latest execution query')
        key = (sid, dataset)
        if key in self.loaded:
            return self.loaded[key].copy(deep=True)
        old, old_refs = self._old(sid, dataset)
        start = '2026-09-10' if dataset == 'TaiwanStockPriceLimit' and old is not None else START
        fresh = self._request(sid, dataset, start)
        if dataset == 'TaiwanStockDividend':
            combined = merge_dividends(old, fresh)
        else:
            combined = fresh.copy(deep=True) if old is None else pd.concat([old, fresh], ignore_index=True)
            validate_frame(combined, sid, dataset, START, END)
            if old is not None:
                require(combined.iloc[:len(old)].reset_index(drop=True).equals(old.reset_index(drop=True)),
                        'Historical legal limits changed')
        dest = self.directory / (sid + '-' + dataset + '.parquet')
        meta_path = dest.with_suffix('.json')
        meta = dict(stock_id=sid, dataset=dataset, start=START, end=END,
                    historical_rights_frozen_through=ANCHOR if old is not None else None,
                    old_source_sha256=old_refs, new_query_start=start,
                    preparation_rule='ex_date_components' if dataset == 'TaiwanStockDividend' else 'append_only')
        if dest.exists() or meta_path.exists():
            require(dest.exists() and meta_path.exists(), 'Interrupted latest execution publication')
            previous = read(meta_path)
            require(previous == dict(meta, sha256=digest(dest)), 'Latest execution receipt changed')
            require(combined.equals(pd.read_parquet(dest)), 'Latest execution combined data changed')
        else:
            combined.to_parquet(dest, index=False)
            write(meta_path, dict(meta, sha256=digest(dest)))
        bind(dest, self.root, self.refs)
        bind(meta_path, self.root, self.refs)
        if dataset == 'TaiwanStockDividend':
            dividend_path = self.dividend_directory / (sid + '.parquet')
            ensure_dividend_copy(combined, dividend_path, prepare=True)
            bind(dividend_path, self.root, self.refs)
        self.loaded[key] = combined.copy(deep=True)
        return combined
