"""
Thread-safe storage for bot run results.
Keeps the last 50 runs in memory for the web dashboard.
"""
import threading
from collections import deque

_lock = threading.Lock()
_runs = deque(maxlen=50)


def add_run(result: dict):
    with _lock:
        _runs.append(result)


def get_latest() -> dict | None:
    with _lock:
        return _runs[-1] if _runs else None


def get_last_n(n: int = 10) -> list[dict]:
    with _lock:
        items = list(_runs)
    return items[-n:]
