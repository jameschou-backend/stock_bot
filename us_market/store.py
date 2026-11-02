"""Append-only query snapshots, bounded disk use and resumable page receipts."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

from .config import GIB, safe_root
from .client import safe_url


def encoded(value):
    return json.dumps(value,default=str,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Store:
    def __init__(self, root, *, max_bytes=10*GIB, min_free_bytes=40*GIB):
        self.root = safe_root(Path(root))
        self.max_bytes, self.min_free_bytes = max_bytes, min_free_bytes
        if max_bytes <= 0 or min_free_bytes < 0:
            raise ValueError("Invalid US storage budget")
        marker = self.root / "owner.json"
        if self.root.exists() and not marker.exists() and any(self.root.iterdir()):
            raise ValueError("Refusing to adopt a nonempty, unmarked data directory")
        self.root.mkdir(parents=True,exist_ok=True)
        self._path("owner.json")
        owner = encoded(dict(namespace="stock_bot_us",schema=1))
        if marker.exists():
            if marker.read_bytes() != owner:
                raise ValueError("US storage ownership mismatch")
        else:
            with marker.open("xb") as f:
                f.write(owner)

    def _path(self, relative):
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("Unsafe US storage path")
        path = self.root / rel
        if any(p.is_symlink() for p in [path,*path.parents]):
            raise ValueError("US storage path contains a symbolic link")
        return path

    def capacity(self, additional=0):
        used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        free = shutil.disk_usage(self.root).free
        if used + additional > self.max_bytes or free - additional < self.min_free_bytes:
            raise ValueError("US storage quota/free-space reserve reached; no old data will be deleted")
        return dict(used_bytes=used,free_bytes=free,max_bytes=self.max_bytes,min_free_bytes=self.min_free_bytes)

    @contextmanager
    def lock(self):
        # One writer across ALL queries: quota checks and publishing are serialized.
        with self._path("writer.lock").open("a") as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Another US ingestion writer is active") from None
            try:
                yield
            finally:
                fcntl.flock(lock,fcntl.LOCK_UN)

    def put(self, relative, data):
        path = self._path(relative)
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("Immutable US object conflicts; choose a new revision")
            return dict(path=str(relative),sha256=sha(data),bytes=len(data))
        self.capacity(len(data))
        path.parent.mkdir(parents=True,exist_ok=True)
        fd, temp = tempfile.mkstemp(prefix=".writing-",dir=path.parent)
        try:
            with os.fdopen(fd,"wb") as out:
                out.write(data); out.flush(); os.fsync(out.fileno())
            # Caller holds the global writer lock. Never replace an existing object.
            os.link(temp,path)
        finally:
            Path(temp).unlink(missing_ok=True)
        return dict(path=str(relative),sha256=sha(data),bytes=len(data))

    def read_object(self, ref):
        data = self._path(ref["path"]).read_bytes()
        if sha(data) != ref["sha256"] or len(data) != ref["bytes"]:
            raise ValueError("US snapshot object checksum mismatch")
        return data

    def read_json(self, path):
        return json.loads(self._path(path).read_bytes())

    def has(self, path):
        return self._path(path).exists()

    def page(self, query, index):
        path = f"queries/{query.key}/page-{index:06d}.json"
        if not self.has(path):
            return None
        item = self.read_json(path)
        if item["query"] != query.descriptor() or item["index"] != index:
            raise ValueError("US page receipt identity mismatch")
        raw = self.read_object(item["source"])
        self.read_object(item["parquet"])
        payload = json.loads(gzip.decompress(raw))
        return item, payload

    def save_page(self, query, index, url, payload, table):
        folder = f"queries/{query.key}"
        source = self.put(f"{folder}/source-{index:06d}.json.gz",gzip.compress(encoded(payload),mtime=0))
        out = pa.BufferOutputStream()
        pq.write_table(table,out,compression="zstd",row_group_size=100_000)
        parquet = self.put(f"{folder}/data-{index:06d}.parquet",out.getvalue().to_pybytes())
        receipt = dict(query=query.descriptor(),index=index,url=safe_url(url),source=source,
                       parquet=parquet,rows=table.num_rows,retrieved_at=datetime.now(timezone.utc).isoformat(),
                       source_encoding="canonical JSON; exact decimal values represented as strings")
        self.put(f"{folder}/page-{index:06d}.json",encoded(receipt))
        return receipt

    def publish(self, query, pages):
        result = dict(schema="massive_us_snapshot_v1",snapshot_id=query.key,query=query.descriptor(),
                      complete_pagination=True,pages=pages,rows=sum(p["rows"] for p in pages),
                      data_status="synthetic_fixture" if query.source_kind == "synthetic_fixture" else "provider_landing_only",session_scope="provider_day_unfiltered",
                      snapshot_consistency="provider pagination; not guaranteed point-in-time revisions",
                      corporate_actions_reconciled=False,historical_universe_verified=False,
                      calendar_verified=False,live_qualified=False,strategy_ready=False)
        ref = self.put(f"snapshots/{query.key}.json",encoded(result))
        return result, ref

    def snapshots(self):
        folder = self._path("snapshots")
        return [json.loads(p.read_bytes()) for p in sorted(folder.glob("*.json"))] if folder.exists() else []
