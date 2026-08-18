from typing import Any

from .base_extractor import BaseMultimodalExtractor

__all__ = ["BaseMultimodalExtractor", "RuleBasedMultimodalExtractor"]


def __getattr__(name: str) -> Any:
    if name == "BaseMultimodalExtractor":
        return BaseMultimodalExtractor
    if name == "RuleBasedMultimodalExtractor":
        from .rule_based_extractor import RuleBasedMultimodalExtractor

        return RuleBasedMultimodalExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
