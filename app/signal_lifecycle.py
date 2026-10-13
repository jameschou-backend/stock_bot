"""As-of descriptions of known signal records, never trading instructions."""
from bisect import bisect_left, bisect_right
import math


class SignalHistory:
    """Keep transitions distinct from first-ever signals and missing history.

    Records contain matched dates only. Daily snapshots override sparse event
    archives before construction. A missing record is never evidence of absence.
    """

    WINDOW = 120
    LABELS = {'first_observed': '區間首筆', 'retrigger': '再次觸發',
              'continuing': '持續符合', 'unknown': '起點待核'}

    def __init__(self, calendar, records):
        self.calendar = tuple(str(d)[:10] for d in calendar)
        self.positions = {d: i for i, d in enumerate(self.calendar)}
        groups = {}
        for (stock_id, strategy_id, day), first in records.items():
            groups.setdefault((stock_id, strategy_id), {})[day] = first
        self.groups = {key: (sorted(rows), rows) for key, rows in groups.items()}

    def describe(self, stock_id, strategy_id, selected, first_signal):
        end = bisect_right(self.calendar, selected)
        start = self.calendar[max(0, end-self.WINDOW)]
        dates, flags = self.groups.get((stock_id, strategy_id), ([], {}))
        prior = dates[bisect_left(dates, start):bisect_left(dates, selected)]
        triggers = [d for d in prior if flags[d] is True]
        previous = triggers[-1] if triggers else None
        state = ('unknown' if first_signal is None else 'continuing' if first_signal is False
                 else 'retrigger' if prior else 'first_observed')
        previous_session = self.calendar[end-2] if end >= 2 else None
        explanation = {
            'unknown': '本日符合，但前一交易日的條件狀態未知，無法判定是否重新觸發。',
            'continuing': '前一交易日與本日皆符合；不是新的條件轉換。',
            'retrigger': '前一交易日未符合，本日再次符合；觀察區間已有同策略紀錄。',
            'first_observed': '前一交易日未符合，本日轉為符合；這是觀察區間內首筆已知紀錄，不代表歷史首次或剛起漲。',
        }[state]
        return dict(state=state, label=self.LABELS[state], explanation=explanation,
                    window_sessions=self.WINDOW, observed_sessions=min(end, self.WINDOW),
                    window_start=start, window_end=selected, coverage='known_records_only',
                    coverage_note='僅比較已封存訊號；未記錄不代表未發生，區間首筆不等於歷史首次。',
                    previous_session=previous_session, previous_trigger_date=previous,
                    previous_match_date=prior[-1] if prior else None,
                    sessions_since_previous_trigger=(self.positions[selected]-self.positions[previous]
                        if previous in self.positions and selected in self.positions else None),
                    prior_known_triggers=len(triggers), price_change_since_previous_trigger=None)


def add_price_context(lifecycle, prices):
    """Compare verified, same-basis closes at exact dates; never carry a price."""
    previous = prices.get(lifecycle['previous_trigger_date'])
    current = prices.get(lifecycle['window_end'])
    valid = all(isinstance(p, (float, int)) and math.isfinite(p) and p > 0
                for p in (previous, current))
    return dict(lifecycle, previous_trigger_close=previous, current_close=current,
                price_change_since_previous_trigger=current/previous-1 if valid else None,
                price_basis='adjusted_close',
                price_note='同基準還原收盤價變化，不是進場報酬；缺少任一日有效價格時不估算。')
