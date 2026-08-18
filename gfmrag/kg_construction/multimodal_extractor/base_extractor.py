from abc import ABC, abstractmethod
from typing import Any


class BaseMultimodalExtractor(ABC):
    @abstractmethod
    def describe_image_for_macro(
        self, image_item: dict[str, Any], doc_item: dict[str, Any]
    ) -> str:
        """Generate a short image description for macro OpenIE corpus."""
        pass

    @abstractmethod
    def extract_from_image(
        self, image_item: dict[str, Any], doc_item: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract micro triples from one image item."""
        pass

    @abstractmethod
    def extract_from_table(
        self, table_item: dict[str, Any], doc_item: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Extract micro triples from one table item."""
        pass
