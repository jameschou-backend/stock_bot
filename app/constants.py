from __future__ import annotations

from enum import Enum


class StockStatusType(str, Enum):
    HALT = "HALT"
    SUSPEND = "SUSPEND"
    DISPOSITION = "DISPOSITION"
    FULL_DELIVERY = "FULL_DELIVERY"
    DELISTED = "DELISTED"
    NOT_LISTED = "NOT_LISTED"
    OTHER = "OTHER"


NON_TRADABLE_STATUSES = {
    StockStatusType.HALT.value,
    StockStatusType.SUSPEND.value,
    StockStatusType.DISPOSITION.value,
    StockStatusType.FULL_DELIVERY.value,
    StockStatusType.DELISTED.value,
    StockStatusType.NOT_LISTED.value,
}


_EXTERNAL_STATUS_KEYWORDS = {
    "halt": StockStatusType.HALT.value,
    "suspend": StockStatusType.SUSPEND.value,
    "suspension": StockStatusType.SUSPEND.value,
    "disposition": StockStatusType.DISPOSITION.value,
    "full_delivery": StockStatusType.FULL_DELIVERY.value,
    "full delivery": StockStatusType.FULL_DELIVERY.value,
    "delisted": StockStatusType.DELISTED.value,
    "not_listed": StockStatusType.NOT_LISTED.value,
    "not listed": StockStatusType.NOT_LISTED.value,
    "處置": StockStatusType.DISPOSITION.value,
    "全額交割": StockStatusType.FULL_DELIVERY.value,
    "停止交易": StockStatusType.HALT.value,
    "暫停交易": StockStatusType.SUSPEND.value,
    "下市": StockStatusType.DELISTED.value,
    "未上市": StockStatusType.NOT_LISTED.value,
}


def map_external_status(raw_status: str | None) -> str:
    if not raw_status:
        return StockStatusType.OTHER.value
    normalized = str(raw_status).strip()
    lowered = normalized.lower()

    if normalized in StockStatusType.__members__:
        return StockStatusType[normalized].value
    if normalized in {s.value for s in StockStatusType}:
        return normalized
    if lowered in _EXTERNAL_STATUS_KEYWORDS:
        return _EXTERNAL_STATUS_KEYWORDS[lowered]
    for keyword, status in _EXTERNAL_STATUS_KEYWORDS.items():
        if keyword in lowered or keyword in normalized:
            return status
    return StockStatusType.OTHER.value
