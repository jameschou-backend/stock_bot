"""Avoid fetching dividends for orders already rejected by the opening slot lock."""
class DeferredCorporatePreparation:
    preparation_engine = None

    def prepare(self, sid):
        engine = self.preparation_engine
        if engine is not None and sid not in getattr(self, 'loaded', {}):
            held = engine.holdings.get(sid, {}).get('qty', 0)
            pending = any(r.get('stock_id') == sid for r in engine.receivables)
            plans = [p for p in engine.day_plans.values()
                     if p['stock_id'] == sid and p['side'] == 'buy']
            if not held and not pending and plans and all(
                p['rejection'] == 'opening_slots_locked' and p['planned_qty'] == 0
                and p['reserved_cash'] == 0 for p in plans
            ):
                if not hasattr(self, 'deferred_preparations'):
                    self.deferred_preparations = set()
                self.deferred_preparations.update((p['date'], sid) for p in plans)
                return
        return super().prepare(sid)
