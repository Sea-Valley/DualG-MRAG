from typing import Any

from .base_language_model import BaseLanguageModel

__all__ = ["BaseLanguageModel", "HfCausalModel", "ChatGPT"]


def __getattr__(name: str) -> Any:
    if name == "BaseLanguageModel":
        return BaseLanguageModel
    if name == "HfCausalModel":
        from .base_hf_causal_model import HfCausalModel

        return HfCausalModel
    if name == "ChatGPT":
        from .chatgpt import ChatGPT

        return ChatGPT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
