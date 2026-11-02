"""Read-only US landing status API; deliberately does not expose Taiwan routes/data."""
from fastapi import FastAPI, Query
from .store import Store


def create_app(store: Store):
    app = FastAPI(title="US research data landing",version="1")

    @app.get("/health")
    def health():
        return dict(status="ok",market="US",strategy_ready=False,live_qualified=False,
                    snapshot_count=len(store.snapshots()),storage=store.capacity())

    @app.get("/picks")
    def picks():
        return dict(market="US",status="strategy_not_implemented",items=[])

    @app.get("/models")
    def models():
        return dict(market="US",status="training_not_implemented",items=[])

    @app.get("/jobs")
    def jobs(limit: int=Query(default=10,ge=1,le=100)):
        return [dict(snapshot_id=s["snapshot_id"],dataset=s["query"]["dataset"],ticker=s["query"]["ticker"],
                     rows=s["rows"],status=s["data_status"],live_qualified=False)
                for s in store.snapshots()[-limit:]]

    return app
