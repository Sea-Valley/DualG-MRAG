from typing import Any

from .base_model import BaseNERModel

__all__ = ["BaseNERModel", "LLMNERModel"]


def __getattr__(name: str) -> Any:
    if name == "BaseNERModel":
        return BaseNERModel
    if name == "LLMNERModel":
        from .llm_ner_model import LLMNERModel

        return LLMNERModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
