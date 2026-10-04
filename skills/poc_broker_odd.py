"""Offline, run-frozen official odd-lot supplementation for new account paths.

These whole-market daily tables are capacity evidence, not executable depth.
No request, cache write, missing-row zero fill, or source precedence override
occurs here. The caller must keep its original frozen sources authoritative.
"""
from copy import deepcopy
from datetime import date
from pathlib import Path
from hashlib import sha256
import json
import re

from skills import replay_market_feeds
from skills.official_daily_acquisition import digest
from skills.poc_latest_data import FrozenOddSources
from skills.poc_latest_execution import CACHE_ORDER
from skills.replay_market_feeds import ReplayDataUnavailable, parse_odd


class OddMarketDayMissing(ReplayDataUnavailable):
    """No compatible source for this market/day; safe to seek new evidence."""


class SupplementaryOddSources(FrozenOddSources):
    """Freeze discovered indexes once; validate selected bytes on every get.

    ``source_refs`` optionally restricts discovery to an already sealed source
    closure. If omitted, discover existing compatible indexes and the explicit
    legacy execution directories, then pin their current identities in memory.
    ``refs`` contains the source closure actually used, while ``source_refs`` is
    the frozen inventory. Serialize both in the new research manifest to reuse
    the exact same inventory on subsequent replays.
    """

    def __init__(self, root, source_refs=None, *, start='2024-01-02', end='2026-10-02'):
        self.root = Path(root).resolve()
        self.start, self.end = date.fromisoformat(start).isoformat(), date.fromisoformat(end).isoformat()
        if (self.start != start or self.end != end or self.start > self.end
                or self.start < '2024-01-02' or self.end > '2026-10-02'):
            raise ValueError('Supplementary odd period is outside the registered scope')
        supplied = dict(source_refs) if source_refs is not None else None
        self._restricted = supplied is not None
        self.source_refs = {} if supplied is None else supplied
        self._verified_reads = {}
        self.refs, self.sources, self.simple_sources = {}, {}, {}
        self.selected_sources, self.queries, self.excluded_indexes = {}, [], []
        self.parser_sha256 = digest(Path(replay_market_feeds.__file__))
        indexes = (sorted({p.parent / 'index.json' for p in self.root.joinpath('.cache').glob('**/odd-*.raw.json')
                           if re.fullmatch(r'odd-(twse|tpex)-\d{4}-\d{2}-\d{2}\.raw\.json', p.name)
                           and (p.parent / 'index.json').is_file()}) if supplied is None
                   else sorted(self._path(k) for k in supplied if Path(k).name == 'index.json'))
        for path in indexes:
            name = self._name(path)
            if supplied is None:
                state, index_hash = self._snapshot(path)
            else:
                state = self._read(name)
                index_hash = supplied[name]
            if not isinstance(state, dict):
                continue
            entries = state.get('entries', {})
            if not isinstance(entries, dict) or not any(str(k).startswith('odd:') for k in entries):
                continue
            if state.get('schema') != 1 or state.get('parser_sha256') != self.parser_sha256:
                self.excluded_indexes.append(name)
                continue
            self._pin(name, index_hash)
            directory = str(Path(name).parent)
            files = state.get('files_sha256', {})
            for key, entry in entries.items():
                match = re.fullmatch(r'odd:(twse|tpex):(\d{4}-\d{2}-\d{2})', key)
                if not match or not self.start <= match[2] <= self.end:
                    continue
                if not isinstance(entry, dict):
                    raise ReplayDataUnavailable('Malformed supplementary odd index entry')
                for field in ('raw_file', 'rows_file'):
                    child = self._child(directory, entry[field])
                    expected = files.get(entry[field])
                    if not isinstance(expected, str) or not re.fullmatch(r'[a-f0-9]{64}', expected):
                        raise ReplayDataUnavailable('Supplementary odd index lacks source hashes')
                    if supplied is not None and supplied.get(child) != expected:
                        raise ReplayDataUnavailable('Supplementary odd file outside sealed closure: ' + child)
                    self._pin(child, expected)
                self.sources.setdefault(key, []).append((directory, name, deepcopy(entry)))
        for folder in CACHE_ORDER:
            paths = (sorted((self.root / folder).glob('odd-*.json')) if supplied is None
                     else sorted(self._path(k) for k in supplied if str(Path(k).parent) == folder))
            for path in paths:
                match = re.fullmatch(r'odd-(twse|tpex)-(\d{4}-\d{2}-\d{2})\.json', path.name)
                if not match or not self.start <= match[2] <= self.end:
                    continue
                name = self._name(path)
                self._pin(name, digest(path) if supplied is None else supplied[name])
                key = 'odd:' + match[1] + ':' + match[2]
                self.simple_sources.setdefault(key, []).append(name)
                wrapper = self._read(name)
                if any(field in wrapper for field in
                       ('source_receipt', 'source_receipt_sha256', 'source_raw', 'source_raw_sha256')):
                    # The first persisted inventory must already contain every
                    # receipt ancestor; discovering these only at get() makes
                    # a strict replay depend on which stocks were queried first.
                    self._simple(name, match[1], match[2])
        # Discovery is not usage: only later get() records sources as consumed.
        self.refs = {}
        self.available_market_days = tuple(sorted(set(self.sources) | set(self.simple_sources)))

    def _path(self, name):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ReplayDataUnavailable('Supplementary odd path escapes repository')
        return path

    def _name(self, path):
        return str(self._path(path).relative_to(self.root))

    @staticmethod
    def _json(path):
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError, TypeError) as exc:
            raise ReplayDataUnavailable('Supplementary odd source cannot be read: ' + str(path)) from exc

    @staticmethod
    def _snapshot(path):
        """Bind parsed discovery content to the very same bytes, not a later read."""
        try:
            before = SupplementaryOddSources._fingerprint(path)
            content = Path(path).read_bytes()
            if SupplementaryOddSources._fingerprint(path) != before:
                raise ReplayDataUnavailable('Supplementary odd index changed during discovery: ' + str(path))
            return json.loads(content), sha256(content).hexdigest()
        except (OSError, ValueError, TypeError) as exc:
            raise ReplayDataUnavailable('Supplementary odd index cannot be read: ' + str(path)) from exc

    def _pin(self, name, expected):
        self._path(name)
        if name in self.source_refs and self.source_refs[name] != expected:
            raise ReplayDataUnavailable('Conflicting supplementary source hash: ' + name)
        self.source_refs[name] = expected

    def _read(self, name, expected=None):
        try:
            path = self._path(name)
            frozen = self.source_refs.get(name)
            if frozen is None or expected is not None and frozen != expected:
                raise ReplayDataUnavailable('Supplementary odd file outside sealed closure: ' + name)
            before = self._fingerprint(path)
            cached = self._verified_reads.get(name)
            if cached is None or cached[0] != before:
                if digest(path) != frozen:
                    raise ReplayDataUnavailable('Supplementary odd source changed: ' + name)
                value = self._json(path)
                if self._fingerprint(path) != before:
                    raise ReplayDataUnavailable('Supplementary odd source changed during read: ' + name)
                cached = self._verified_reads[name] = (before, value)
            self.refs[name] = frozen
            return cached[1]
        except (OSError, ValueError, TypeError) as exc:
            raise ReplayDataUnavailable('Supplementary odd source changed or unreadable: ' + name) from exc

    @staticmethod
    def _fingerprint(path):
        value = path.stat()
        return value.st_size, value.st_mtime_ns, value.st_ctime_ns

    def _simple(self, name, market, day):
        record = self._read(name)
        links = ('source_receipt', 'source_receipt_sha256', 'source_raw', 'source_raw_sha256')
        if any(field in record for field in links):
            if not all(field in record for field in links):
                raise ReplayDataUnavailable('Supplementary odd receipt chain is incomplete: ' + name)
            # Newer normal-probe wrappers require their complete accepted chain;
            # copying just the HTTP200 wrapper must not drop authorization/proof.
            from scripts.prepare_volume_profile_odd import OddCompletion, BASE
            verifier = OddCompletion.__new__(OddCompletion)
            verifier.root, verifier.cache, verifier.refs = self.root, self.root / BASE, {}
            try:
                checked = verifier.verify_cached(self._path(name))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise ReplayDataUnavailable('Supplementary odd receipt chain is invalid: ' + name) from exc
            for dependency, expected in checked['source_sha256'].items():
                # A supplied closure may not silently acquire new ancestors.
                if self._restricted and self.source_refs.get(dependency) != expected:
                    raise ReplayDataUnavailable('Supplementary odd ancestor outside sealed closure: ' + dependency)
                self._pin(dependency, expected)
                self._read(dependency, expected) if dependency.endswith('.json') else self._mark_bytes(dependency, expected)
            return checked['rows']
        return parse_odd(record, market, day)

    def _mark_bytes(self, name, expected):
        if digest(self._path(name)) != expected:
            raise ReplayDataUnavailable('Supplementary odd ancestor changed: ' + name)
        self.refs[name] = expected

    def get(self, day, sid, market):
        if (not isinstance(day, str) or date.fromisoformat(day).isoformat() != day
                or not self.start <= day <= self.end):
            raise ValueError('Supplementary odd query outside scope')
        if not isinstance(sid, str) or not re.fullmatch(r'\d{4}', sid):
            raise ValueError('Supplementary odd query requires a four-digit stock')
        market = str(market).lower()
        if market not in ('twse', 'tpex'):
            raise ValueError('Supplementary odd query market is invalid')
        key = 'odd:' + market + ':' + day
        self.queries.append(dict(date=day, stock_id=sid, market=market))
        if key not in self.sources and key not in self.simple_sources:
            raise OddMarketDayMissing('Supplementary official odd day missing: ' + key)
        before = set(self.refs)
        groups = self._raw_rows(key, market, day) if key in self.sources else []
        groups += [self._simple(name, market, day) for name in self.simple_sources.get(key, [])]
        values = [rows[sid] for rows in groups if sid in rows]
        if not values:
            raise ReplayDataUnavailable('Supplementary odd stock absent: ' + sid + ' ' + day)
        if any(value != values[0] for value in values[1:]):
            raise ReplayDataUnavailable('Conflicting supplementary odd rows: ' + sid + ' ' + day)
        selected = self.selected_sources.setdefault(key, dict(market=market, date=day, stock_ids=[], sources=[]))
        if sid not in selected['stock_ids']:
            selected['stock_ids'].append(sid)
        names = ({item[1] for item in self.sources.get(key, [])}
                 | set(self.simple_sources.get(key, [])) | (set(self.refs) - before))
        selected['sources'] = sorted(set(selected['sources']) | names)
        return deepcopy(values[0])
