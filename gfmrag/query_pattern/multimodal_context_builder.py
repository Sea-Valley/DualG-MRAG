from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MultimodalContextBuilder:
    """Build final-generation prompt text and media inputs for VLM."""

    def __init__(
        self,
        dataset_root: str,
        data_name: str,
        pictures_root: str,
        tables_image_root: str,
        table_image_extensions: list[str] | None = None,
        max_images: int = 6,
        include_media_metadata_in_prompt: bool = True,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.data_name = data_name
        self.pictures_root = Path(pictures_root)
        self.tables_image_root = Path(tables_image_root)
        self.table_image_extensions = table_image_extensions or [
            "png",
            "jpg",
            "jpeg",
            "webp",
        ]
        self.max_images = max_images
        self.include_media_metadata_in_prompt = include_media_metadata_in_prompt

        self._image_meta: dict[str, dict[str, Any]] = {}
        self._table_meta: dict[str, dict[str, Any]] = {}
        self._image_title2id: dict[str, str] = {}
        self._table_title2id: dict[str, str] = {}
        self._load_metadata()

    def _read_json(self, path: Path) -> Any:
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as fin:
            return json.load(fin)

    def _load_metadata(self) -> None:
        image_corpus_path = (
            self.dataset_root / self.data_name / "dataset_corpus_image.json"
        )
        table_corpus_path = (
            self.dataset_root / self.data_name / "dataset_corpus_table.json"
        )

        image_payload = self._read_json(image_corpus_path)
        if isinstance(image_payload, dict):
            self._image_meta = image_payload
            for doc_id, item in image_payload.items():
                title = str(item.get("title", "")).strip().lower()
                if title:
                    self._image_title2id[title] = str(doc_id)

        table_payload = self._read_json(table_corpus_path)
        if isinstance(table_payload, dict):
            self._table_meta = table_payload
            for doc_id, item in table_payload.items():
                title = str(item.get("title", "")).strip().lower()
                if title:
                    self._table_title2id[title] = str(doc_id)

    def _candidate_doc_ids(self, doc_title: str) -> list[str]:
        normalized = doc_title.strip().lower()
        candidates: list[str] = []
        if not normalized:
            return candidates
        # Most stage3 docs use doc_id as title.
        candidates.append(doc_title.strip())
        if normalized in self._image_title2id:
            candidates.append(self._image_title2id[normalized])
        if normalized in self._table_title2id:
            candidates.append(self._table_title2id[normalized])
        return list(dict.fromkeys([item for item in candidates if item]))

    def _resolve_image_path(self, doc_id: str) -> str | None:
        meta = self._image_meta.get(doc_id, {})
        path_str = str(meta.get("path", "")).strip()
        if path_str:
            candidate = self.pictures_root / path_str
            if candidate.exists():
                return str(candidate.resolve())
        for ext in self.table_image_extensions:
            candidate = self.pictures_root / f"{doc_id}.{ext}"
            if candidate.exists():
                return str(candidate.resolve())
        return None

    def _resolve_table_image_path(self, doc_id: str) -> str | None:
        for ext in self.table_image_extensions:
            candidate = self.tables_image_root / f"{doc_id}.{ext}"
            if candidate.exists():
                return str(candidate.resolve())
        return None

    def _build_media_items(
        self, retrieved_docs: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        media_items: list[dict[str, str]] = []
        seen_paths: set[str] = set()
        for doc in retrieved_docs:
            doc_title = str(doc.get("title", "")).strip()
            if not doc_title:
                continue
            for doc_id in self._candidate_doc_ids(doc_title):
                image_path = self._resolve_image_path(doc_id)
                if image_path and image_path not in seen_paths:
                    meta = self._image_meta.get(doc_id, {})
                    title = str(meta.get("title", doc_title))
                    caption = title
                    media_items.append(
                        {
                            "doc_id": doc_id,
                            "modality": "image",
                            "image_path": image_path,
                            "title": title,
                            "caption": caption,
                        }
                    )
                    seen_paths.add(image_path)

                table_image_path = self._resolve_table_image_path(doc_id)
                if table_image_path and table_image_path not in seen_paths:
                    meta = self._table_meta.get(doc_id, {})
                    title = str(meta.get("title", doc_title))
                    caption = f"Table image for {title}"
                    media_items.append(
                        {
                            "doc_id": doc_id,
                            "modality": "table_image",
                            "image_path": table_image_path,
                            "title": title,
                            "caption": caption,
                        }
                    )
                    seen_paths.add(table_image_path)
        return media_items

    def _build_prompt_text(
        self,
        question: str,
        retrieved_docs: list[dict[str, Any]],
        media_items: list[dict[str, str]],
    ) -> str:
        lines = [
            "You are a multimodal QA assistant. Use provided text and images to answer.",
            f"Question: {question}",
            "",
            "Retrieved textual evidence:",
        ]
        for idx, doc in enumerate(retrieved_docs, start=1):
            title = str(doc.get("title", ""))
            content = str(doc.get("content", ""))
            lines.append(f"[Doc#{idx}] Title: {title}")
            lines.append(content)
            lines.append("")

        if self.include_media_metadata_in_prompt and media_items:
            lines.append("Media evidence metadata:")
            for idx, item in enumerate(media_items, start=1):
                lines.append(
                    f"[Image#{idx}] doc_id={item['doc_id']} modality={item['modality']} "
                    f"title={item['title']} caption={item['caption']}"
                )
            lines.append("")

        lines.append("Answer the question directly and concisely.")
        return "\n".join(lines)

    def build(
        self, question: str, retrieved_docs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        media_items = self._build_media_items(retrieved_docs)
        prompt_text = self._build_prompt_text(question, retrieved_docs, media_items)
        image_paths = [item["image_path"] for item in media_items]
        return {
            "prompt_text": prompt_text,
            "image_paths": image_paths,
            "media_items": media_items,
        }
