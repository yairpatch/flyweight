from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock

from .cache import LayeredExpertCache
from .expert import Expert, ExpertKey
from .storage import ExpertStore


class ResidencyManager:
    """Unified disk-to-RAM residency API; VRAM backends can implement this boundary."""

    def __init__(
        self,
        store: ExpertStore,
        cache: LayeredExpertCache,
        *,
        io_workers: int = 4,
    ):
        self.store = store
        self.cache = cache
        self._executor = ThreadPoolExecutor(
            max_workers=io_workers, thread_name_prefix="coli-io"
        )
        self._prefetches: dict[ExpertKey, Future[Expert]] = {}
        self._lock = RLock()
        self.disk_loads = 0
        self.prefetch_requests = 0
        self.prefetch_hits = 0

    def get(self, key: ExpertKey) -> Expert:
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        with self._lock:
            future = self._prefetches.pop(key, None)
        if future is not None:
            expert = future.result()
            self.prefetch_hits += 1
            self.cache.put(expert)
            return expert

        expert = self._load(key)
        self.cache.put(expert)
        return expert

    def prefetch(self, keys: list[ExpertKey]) -> None:
        for key in keys:
            if self.cache.peek(key) is not None:
                continue
            with self._lock:
                if key in self._prefetches:
                    continue
                self.prefetch_requests += 1
                self._prefetches[key] = self._executor.submit(self._load, key)

    def pin(self, keys: list[ExpertKey]) -> None:
        for key in keys:
            expert = self.cache.peek(key) or self._load(key)
            self.cache.pin(expert)

    def stats(self) -> dict[str, int | float]:
        return {
            **self.cache.stats(),
            "disk_loads": self.disk_loads,
            "prefetch_requests": self.prefetch_requests,
            "prefetch_hits": self.prefetch_hits,
        }

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "ResidencyManager":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _load(self, key: ExpertKey) -> Expert:
        expert = self.store.load(key)
        with self._lock:
            self.disk_loads += 1
        return expert
