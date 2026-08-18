from typing import Any

__all__ = [
    "BaseKGConstructor",
    "KGConstructor",
    "BaseQAConstructor",
    "QAConstructor",
    "BaseMultimodalExtractor",
    "RuleBasedMultimodalExtractor",
    "BaseMicroMatcher",
    "SimGRAGMicroMatcher",
]


def __getattr__(name: str) -> Any:
    if name in {"BaseKGConstructor", "KGConstructor"}:
        from .kg_constructor import BaseKGConstructor, KGConstructor

        mapping = {
            "BaseKGConstructor": BaseKGConstructor,
            "KGConstructor": KGConstructor,
        }
        return mapping[name]
    if name in {"BaseQAConstructor", "QAConstructor"}:
        from .qa_constructor import BaseQAConstructor, QAConstructor

        mapping = {
            "BaseQAConstructor": BaseQAConstructor,
            "QAConstructor": QAConstructor,
        }
        return mapping[name]
    if name in {"BaseMultimodalExtractor", "RuleBasedMultimodalExtractor"}:
        from .multimodal_extractor import (
            BaseMultimodalExtractor,
            RuleBasedMultimodalExtractor,
        )

        mapping = {
            "BaseMultimodalExtractor": BaseMultimodalExtractor,
            "RuleBasedMultimodalExtractor": RuleBasedMultimodalExtractor,
        }
        return mapping[name]
    if name in {"BaseMicroMatcher", "SimGRAGMicroMatcher"}:
        from .micro_matcher import BaseMicroMatcher, SimGRAGMicroMatcher

        mapping = {
            "BaseMicroMatcher": BaseMicroMatcher,
            "SimGRAGMicroMatcher": SimGRAGMicroMatcher,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
