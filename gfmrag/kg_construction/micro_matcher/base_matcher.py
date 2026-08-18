from abc import ABC, abstractmethod
from typing import Any


class BaseMicroMatcher(ABC):
    """Abstract interface for micro-KG pattern matching."""

    @abstractmethod
    def build_index(self, micro_triples: list[dict[str, Any]]) -> None:
        """Build searchable structures from stage1 micro triples."""
        pass

    @abstractmethod
    def match_pattern(
        self,
        pattern_triples: list[tuple[str, str, str]],
        topk: int = 20,
        mode: str = "greedy",
    ) -> list[dict[str, Any]]:
        """Match condition pattern triples against indexed micro-KG."""
        pass
