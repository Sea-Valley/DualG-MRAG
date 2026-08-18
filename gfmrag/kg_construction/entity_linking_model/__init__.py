from typing import Any

from .base_model import BaseELModel

__all__ = ["BaseELModel", "ColbertELModel", "DPRELModel", "NVEmbedV2ELModel"]


def __getattr__(name: str) -> Any:
    if name == "BaseELModel":
        return BaseELModel
    if name == "ColbertELModel":
        from .colbert_el_model import ColbertELModel

        return ColbertELModel
    if name in {"DPRELModel", "NVEmbedV2ELModel"}:
        from .dpr_el_model import DPRELModel, NVEmbedV2ELModel

        mapping = {
            "DPRELModel": DPRELModel,
            "NVEmbedV2ELModel": NVEmbedV2ELModel,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
