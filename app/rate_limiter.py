"""Rate Limiter 模組

提供 API 請求速率控制，避免觸發 FinMind API 限流。

FinMind 限制：
- 免費用戶：每小時 600 次
- 付費用戶：每小時 6000 次（或更高）
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class RateLimitStats:
    """Rate Limiter 統計資訊"""
    requests_in_window: int
    window_start_time: float
    total_requests: int
    total_wait_time: float


class RateLimiter:
    """滑動視窗速率限制器
    
    使用滑動視窗演算法，精確控制每小時 API 請求次數。
    
    Example:
        limiter = RateLimiter(requests_per_hour=6000)
        
        for i in range(10000):
            limiter.acquire()  # 會自動等待如果超過限制
            make_api_call()
    """
    
    def __init__(
        self,
        requests_per_hour: int = 6000,
        buffer_percent: float = 0.1,
    ):
        """
        Args:
            requests_per_hour: 每小時允許的請求數
            buffer_percent: 預留緩衝比例（預設 10%）避免邊界問題
        """
        self._requests_per_hour = requests_per_hour
        self._buffer_percent = buffer_percent
        self._effective_limit = int(requests_per_hour * (1 - buffer_percent))
        self._window_seconds = 3600  # 1 小時
        
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()
        self._total_requests = 0
        self._total_wait_time = 0.0
    
    @property
    def requests_per_hour(self) -> int:
        return self._requests_per_hour
    
    @property
    def effective_limit(self) -> int:
        return self._effective_limit
    
    def _clean_old_timestamps(self, now: float) -> None:
        """清除超過視窗時間的舊記錄"""
        cutoff = now - self._window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
    
    def acquire(self, timeout: Optional[float] = None) -> bool:
        """取得一次 API 請求許可
        
        如果已達速率限制，會等待直到可以請求。
        
        Args:
            timeout: 最大等待時間（秒），None 表示無限等待
            
        Returns:
            True 如果成功取得許可，False 如果 timeout
        """
        start_wait = time.time()
        
        with self._lock:
            now = time.time()
            self._clean_old_timestamps(now)
            
            # 檢查是否需要等待
            while len(self._timestamps) >= self._effective_limit:
                if timeout is not None:
                    elapsed = time.time() - start_wait
                    if elapsed >= timeout:
                        return False
                
                # 計算需要等待的時間
                oldest = self._timestamps[0]
                wait_time = oldest + self._window_seconds - time.time() + 0.1
                
                if wait_time > 0:
                    # 釋放鎖再等待
                    self._lock.release()
                    try:
                        actual_wait = min(wait_time, 60)  # 最多等 60 秒再重新檢查
                        time.sleep(actual_wait)
                        self._total_wait_time += actual_wait
                    finally:
                        self._lock.acquire()
                    
                    now = time.time()
                    self._clean_old_timestamps(now)
            
            # 記錄這次請求
            self._timestamps.append(time.time())
            self._total_requests += 1
            return True
    
    def get_stats(self) -> RateLimitStats:
        """取得目前統計資訊"""
        with self._lock:
            now = time.time()
            self._clean_old_timestamps(now)
            return RateLimitStats(
                requests_in_window=len(self._timestamps),
                window_start_time=self._timestamps[0] if self._timestamps else now,
                total_requests=self._total_requests,
                total_wait_time=self._total_wait_time,
            )
    
    def remaining_requests(self) -> int:
        """取得目前視窗內剩餘可用請求數"""
        with self._lock:
            now = time.time()
            self._clean_old_timestamps(now)
            return max(0, self._effective_limit - len(self._timestamps))
    
    def reset(self) -> None:
        """重置計數器（用於測試）"""
        with self._lock:
            self._timestamps.clear()
            self._total_requests = 0
            self._total_wait_time = 0.0


# 全域 Rate Limiter 實例
_global_limiter: Optional[RateLimiter] = None
_global_lock = threading.Lock()


def get_rate_limiter(requests_per_hour: int = 6000) -> RateLimiter:
    """取得全域 Rate Limiter
    
    首次呼叫時會建立實例，後續呼叫返回同一實例。
    
    Args:
        requests_per_hour: 每小時允許的請求數（僅首次有效）
        
    Returns:
        全域 RateLimiter 實例
    """
    global _global_limiter
    
    with _global_lock:
        if _global_limiter is None:
            _global_limiter = RateLimiter(requests_per_hour=requests_per_hour)
        return _global_limiter


def reset_global_limiter() -> None:
    """重置全域 Rate Limiter（用於測試）"""
    global _global_limiter
    
    with _global_lock:
        _global_limiter = None
