from typing import Any

from .base_model import BaseOPENIEModel

__all__ = ["BaseOPENIEModel", "LLMOPENIEModel"]


def __getattr__(name: str) -> Any:
    if name == "BaseOPENIEModel":
        return BaseOPENIEModel
    if name == "LLMOPENIEModel":
        from .llm_openie_model import LLMOPENIEModel

        return LLMOPENIEModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
