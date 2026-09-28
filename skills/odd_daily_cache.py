"""Read-only reuse of authenticated official odd-lot daily tables, never ticks."""
from copy import deepcopy
from pathlib import Path
import json
from skills.replay_market_feeds import ReplayMarketFeeds,ReplayDataUnavailable,parse_odd,_sha


class OddDailyCache:
    def __init__(self,root,primary):
        self.root=Path(root);self.files={};self.queries=[];self.cached={};self.sources={}
        directories=[Path(primary),*sorted({p.parent for p in (self.root/'.cache').glob('**/odd-*.raw.json')})]
        for directory in dict.fromkeys(directories):
            index=directory/'index.json'
            if not index.exists():continue
            state=json.loads(index.read_text())
            if state.get('schema')!=1 or state.get('parser_sha256')!=_sha(__import__('skills.replay_market_feeds',fromlist=['x']).__file__):continue
            for key,entry in state['entries'].items():
                if key.startswith('odd:'):self.sources.setdefault(key,[]).append((directory,state,entry))

    def get_odd(self,day,sid,market):
        market=str(market).lower();key=f'odd:{market}:{day}'
        self.queries.append(dict(date=day,stock_id=sid,market=market))
        if key not in self.cached:
            source=self.sources.get(key)
            if not source:raise ReplayDataUnavailable('No cached official daily odd source: '+key)
            # Identical raw hashes need only one parse; disagreeing versions are
            # compared for the requested stock rather than silently preferred.
            unique={}
            for directory,state,entry in source:
                digest=state['files_sha256'][entry['raw_file']]
                unique.setdefault(digest,(directory,state,entry))
            rows=[]
            for directory,state,entry in unique.values():
                feed=ReplayMarketFeeds(directory,offline=True)
                normalized=feed._cached(state,key)
                raw=directory/entry['raw_file']
                parsed=parse_odd(json.loads(raw.read_text()),market,__import__('datetime').date.fromisoformat(day))
                if parsed!=normalized:raise ReplayDataUnavailable('Odd raw and normalized rows disagree')
                rows.append(normalized)
                for path in (directory/'index.json',raw,directory/entry['rows_file']):self.files[str(path.relative_to(self.root))]=_sha(path)
            self.cached[key]=rows
        values=[r[sid] for r in self.cached[key] if sid in r]
        if not values:raise ReplayDataUnavailable(f'Odd daily row missing: {sid} {day}')
        if any(v!=values[0] for v in values[1:]):raise ReplayDataUnavailable(f'Conflicting odd daily rows: {sid} {day}')
        return deepcopy(values[0])
