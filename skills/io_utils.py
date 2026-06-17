"""原子檔案寫入與安全讀取工具。

state / portfolio 等含真實持倉的 JSON 檔，若寫入中途被中斷（launchd 在睡眠/關機
邊界 kill process）會留下截斷的半個檔案，下次讀取直接崩潰並遺失停損基準（peak_price）。
原子寫入（temp file → fsync → os.replace）確保讀者永遠看到「完整的舊檔」或「完整的新檔」，
不會看到半寫狀態。

讀取採 fail-loud 策略：檔案不存在視為正常（回 default，等同全新開始）；但檔案存在卻
損毀時不靜默回 default（那會讓系統誤以為無持倉而重新進場），而是先試 .bak 備份，
全部失敗才 raise，逼使用者正視。
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path | str, obj: Any, *, indent: int = 2) -> None:
    """原子寫入 JSON：寫同目錄 temp 檔 → fsync → os.replace 換上。

    os.replace 在同一檔案系統上是原子操作，並發讀者不會讀到半寫內容。
    寫入失敗會清掉 temp 檔，不留垃圾。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def safe_read_json(path: Path | str, default: Any = None, *, fallback_to_bak: bool = True) -> Any:
    """讀 JSON。不存在 → 回 default；損毀 → 試 .bak 備份；全失敗則 raise（不靜默吞）。

    與「禁止 silent fallback」規範一致：不存在是正常狀態（全新開始），
    但存在卻損毀是異常，必須讓使用者知道，而非靜默用 default 蓋過真實持倉。
    """
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        if fallback_to_bak:
            for bak in sorted(path.parent.glob(f"{path.name}.bak*"), reverse=True):
                try:
                    return json.loads(bak.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                    continue
        raise ValueError(
            f"{path} JSON 損毀且無可用 .bak 備份：{exc}。"
            "請手動修復或刪除（刪除將遺失持倉狀態，請先確認真實部位）。"
        ) from exc
