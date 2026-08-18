from typing import Any

__all__ = ["GFMRetriever", "KGIndexer"]


def __getattr__(name: str) -> Any:
    if name == "GFMRetriever":
        from .gfmrag_retriever import GFMRetriever

        return GFMRetriever
    if name == "KGIndexer":
        from .kg_indexer import KGIndexer

        return KGIndexer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
