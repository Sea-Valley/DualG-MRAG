import heapq
from typing import Any


class KSmallest:
    """Keep k smallest scored candidates."""

    def __init__(self, k: int):
        self.k = max(1, int(k))
        self._heap: list[tuple[float, Any, bool]] = []

    def add(self, score: float, payload: Any, reuse_nodes: bool) -> None:
        heapq.heappush(self._heap, (-score, payload, reuse_nodes))
        if len(self._heap) > self.k:
            heapq.heappop(self._heap)

    def get(self) -> list[tuple[float, Any, bool]]:
        ranked = sorted(
            [
                (-neg_score, payload, reuse_nodes)
                for neg_score, payload, reuse_nodes in self._heap
            ],
            key=lambda item: item[0],
        )
        # Prefer a non-node-reuse solution when scores are similar.
        ranked.sort(key=lambda item: (item[2], item[0]))
        return ranked[: self.k]

    def max_score(self) -> float:
        if len(self._heap) < self.k:
            return float("inf")
        return -self._heap[0][0]
