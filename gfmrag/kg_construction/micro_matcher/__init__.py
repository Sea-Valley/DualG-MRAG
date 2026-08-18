from typing import Any

from .base_matcher import BaseMicroMatcher

__all__ = ["BaseMicroMatcher", "SimGRAGMicroMatcher"]


def __getattr__(name: str) -> Any:
    if name == "BaseMicroMatcher":
        return BaseMicroMatcher
    if name == "SimGRAGMicroMatcher":
        from .simgrag_matcher import SimGRAGMicroMatcher

        return SimGRAGMicroMatcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
