"""Reuse parsed frozen odd tables while checking their evidence on every use.

This adapter belongs only to the all-entry experiment. It leaves discovery,
source precedence, missing-row errors, conflict checks and query logs in the
original provider. A cache entry stores a parsed table and every byte source
read to derive it; each hit passes those sources through the original verifier.
"""
from skills.poc_broker_odd import SupplementaryOddSources


class CachedSupplement(SupplementaryOddSources):
    @classmethod
    def from_verified_inventory(cls, original):
        if type(original) is not SupplementaryOddSources:
            raise ValueError('Odd cache requires the reviewed frozen provider')
        result = cls.__new__(cls)
        result.__dict__ = dict(original.__dict__)
        result._parsed_tables, result._capture = {}, None
        return result

    def _read(self, name, expected=None):
        value = super()._read(name, expected)
        if self._capture is not None:
            self._capture[name] = (self.source_refs[name], True)
        return value

    def _mark_bytes(self, name, expected):
        super()._mark_bytes(name, expected)
        if self._capture is not None:
            self._capture[name] = (expected, False)

    def _parsed(self, key, load):
        cached = self._parsed_tables.get(key)
        if cached is not None:
            value, dependencies = cached
            for name, (expected, is_json) in dependencies.items():
                if is_json:
                    self._read(name, expected)
                else:
                    self._mark_bytes(name, expected)
            return value
        if self._capture is not None:
            raise ValueError('Unexpected nested odd table parser')
        self._capture = {}
        try:
            value = load()
            dependencies = dict(self._capture)
            if not dependencies:
                raise ValueError('Parsed odd table has no verified sources')
            self._parsed_tables[key] = (value, dependencies)
            return value
        finally:
            self._capture = None

    def _raw_rows(self, key, market, day):
        return self._parsed(('raw', key, market, day),
            lambda: super(CachedSupplement, self)._raw_rows(key, market, day))

    def _simple(self, name, market, day):
        return self._parsed(('simple', name, market, day),
            lambda: super(CachedSupplement, self)._simple(name, market, day))


def install_parsed_cache(provider):
    provider.intraday_odds.supplement = CachedSupplement.from_verified_inventory(
        provider.intraday_odds.supplement)
