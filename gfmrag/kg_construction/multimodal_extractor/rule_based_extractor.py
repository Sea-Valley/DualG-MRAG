import json
import logging
import os
from datetime import datetime
from typing import Any

from gfmrag.kg_construction.utils import extract_json_dict

from .base_extractor import BaseMultimodalExtractor

logger = logging.getLogger(__name__)


class RuleBasedMultimodalExtractor(BaseMultimodalExtractor):
    """
    A lightweight multimodal extractor:
    - Images: optional VLM extraction, fallback to caption-based pseudo triples
    - Tables: rule-based row/header conversion to triples
    """

    def __init__(
        self,
        use_vlm_for_images: bool = False,
        fallback_to_caption_when_vlm_fails: bool = True,
        require_vlm_triples: bool = False,
        failure_log_path: str | None = None,
        failure_log_preview_chars: int = 800,
        image_model_name_or_path: str = "Qwen/Qwen3-VL-8B-Instruct",
        image_tensor_parallel_size: int = 2,
        image_gpu_memory_utilization: float = 0.92,
        image_max_num_batched_tokens: int = 16384,
        image_max_model_len: int = 16384,
        image_allowed_local_media_path: str | None = None,
        image_prompt: str = (
            "Extract core factual triples from the image. "
            'Return strict JSON object: {"triples": [[head, relation, tail], ...]}.'
        ),
        macro_image_description_prompt: str = (
            "Write 3-5 short factual sentences for retrieval/OpenIE. "
            "You may use both image evidence and metadata (doc_title, image_title, image_caption). "
            "Prefer reusing concrete nouns/entities from image_title/image_caption in the output. "
            "Use OpenIE-friendly atomic sentences (one fact per sentence, simple subject-predicate-object). "
            "Avoid vague placeholders like 'object' or 'thing' if title/caption provides a specific noun. "
            "Output plain text only."
        ),
        max_macro_description_chars: int = 240,
        default_image_confidence: float = 0.7,
        default_table_confidence: float = 0.8,
    ) -> None:
        self.use_vlm_for_images = use_vlm_for_images
        self.fallback_to_caption_when_vlm_fails = bool(
            fallback_to_caption_when_vlm_fails
        )
        self.require_vlm_triples = bool(require_vlm_triples)
        self.failure_log_path = (
            str(failure_log_path).strip() if failure_log_path else ""
        )
        self.failure_log_preview_chars = int(failure_log_preview_chars)
        self.image_model_name_or_path = image_model_name_or_path
        self.image_tensor_parallel_size = int(image_tensor_parallel_size)
        self.image_gpu_memory_utilization = float(image_gpu_memory_utilization)
        self.image_max_num_batched_tokens = int(image_max_num_batched_tokens)
        self.image_max_model_len = int(image_max_model_len)
        self.image_allowed_local_media_path = image_allowed_local_media_path
        self.image_prompt = image_prompt
        self.macro_image_description_prompt = macro_image_description_prompt
        self.max_macro_description_chars = int(max_macro_description_chars)
        self.default_image_confidence = default_image_confidence
        self.default_table_confidence = default_table_confidence
        self._vlm_client: Any = None

    def _init_vlm_if_needed(self) -> None:
        if not self.use_vlm_for_images or self._vlm_client is not None:
            return
        try:
            from gfmrag.llms.qwenvl_vllm import get_or_create_qwen_vl_client

            self._vlm_client = get_or_create_qwen_vl_client(
                model_name_or_path=self.image_model_name_or_path,
                tensor_parallel_size=self.image_tensor_parallel_size,
                gpu_memory_utilization=self.image_gpu_memory_utilization,
                max_num_batched_tokens=self.image_max_num_batched_tokens,
                max_model_len=self.image_max_model_len,
                allowed_local_media_path=self.image_allowed_local_media_path,
            )
        except Exception as exc:
            logger.warning(
                "Failed to initialize Qwen3VLVLLM, fallback to caption-only mode: %s",
                exc,
            )
            self._vlm_client = None

    def _shorten(self, text: Any) -> str:
        value = str(text or "").replace("\n", " ").strip()
        if self.failure_log_preview_chars > 0:
            return value[: self.failure_log_preview_chars]
        return value

    def _append_failure_log(self, payload: dict[str, Any]) -> None:
        if not self.failure_log_path:
            return
        try:
            log_path = os.path.abspath(self.failure_log_path)
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(
                "Failed to write failure log %s: %s", self.failure_log_path, exc
            )

    def _build_failure_payload(
        self,
        reason: str,
        image_item: dict[str, Any],
        doc_item: dict[str, Any],
        image_path: str,
        output_preview: str = "",
        error: str = "",
        json_payload: Any = None,
    ) -> dict[str, Any]:
        doc_id = str(doc_item.get("doc_id", "")).strip()
        doc_title = str(doc_item.get("title", "")).strip()
        image_id = str(image_item.get("image_id", image_item.get("id", ""))).strip()
        caption = str(image_item.get("caption", "")).strip()
        doc_text = str(doc_item.get("text", "") or "")

        abs_image_path = os.path.abspath(image_path) if image_path else ""
        image_exists = bool(abs_image_path and os.path.exists(abs_image_path))
        image_size_bytes = os.path.getsize(abs_image_path) if image_exists else -1

        allowed_root = str(self.image_allowed_local_media_path or "").strip()
        allowed_ok = None
        if allowed_root:
            abs_allowed = os.path.abspath(allowed_root)
            allowed_ok = abs_image_path.startswith(abs_allowed)

        payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "reason": reason,
            "doc_id": doc_id,
            "doc_title": self._shorten(doc_title),
            "doc_text_len": len(doc_text),
            "doc_text_preview": self._shorten(doc_text),
            "image_id": image_id,
            "image_path": image_path,
            "image_exists": image_exists,
            "image_size_bytes": image_size_bytes,
            "allowed_local_media_path": allowed_root,
            "path_under_allowed_root": allowed_ok,
            "caption": self._shorten(caption),
            "vlm_model": self.image_model_name_or_path,
            "use_vlm_for_images": bool(self.use_vlm_for_images),
            "require_vlm_triples": bool(self.require_vlm_triples),
            "output_preview": self._shorten(output_preview),
            "error": self._shorten(error),
        }
        if isinstance(json_payload, dict):
            triples = json_payload.get("triples")
            payload["json_has_triples_key"] = "triples" in json_payload
            payload["json_triples_len"] = (
                len(triples) if isinstance(triples, list) else -1
            )
        return payload

    def _safe_list(self, value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _build_image_prompt(
        self, image_item: dict[str, Any], doc_item: dict[str, Any]
    ) -> str:
        """Build a metadata-aware prompt for image triple extraction."""
        doc_id = str(doc_item.get("doc_id", "")).strip()
        doc_title = str(doc_item.get("title", "")).strip()
        image_id = str(image_item.get("image_id", image_item.get("id", ""))).strip()
        image_title = str(image_item.get("title", "")).strip()
        caption = str(image_item.get("caption", "")).strip()
        image_path = str(
            image_item.get("path_or_url") or image_item.get("path") or ""
        ).strip()

        metadata_lines = [
            f"- doc_id: {doc_id or 'N/A'}",
            f"- doc_title: {doc_title or 'N/A'}",
            f"- image_id: {image_id or 'N/A'}",
            f"- image_title: {image_title or 'N/A'}",
            f"- image_caption: {caption or 'N/A'}",
            f"- image_path: {image_path or 'N/A'}",
        ]
        metadata_block = "\n".join(metadata_lines)

        return (
            f"{self.image_prompt}\n\n"
            "You are extracting micro-KG triples for retrieval.\n"
            "Use both visual content and provided metadata.\n"
            "Prefer concrete entities/relations; avoid vague predicates.\n"
            "try to cennect metadata with image content.\n"
            'Return strict JSON only: {"triples": [[head, relation, tail], ...]}.\n\n'
            "Image metadata:\n"
            f"{metadata_block}\n"
        )

    def describe_image_for_macro(
        self, image_item: dict[str, Any], doc_item: dict[str, Any]
    ) -> str:
        doc_title = str(doc_item.get("title", "")).strip()
        image_title = str(image_item.get("title", "")).strip()
        caption = str(image_item.get("caption", "")).strip()
        image_path = str(
            image_item.get("path_or_url") or image_item.get("path") or ""
        ).strip()

        fallback_parts = [part for part in [image_title, caption] if part]
        fallback = ". ".join(fallback_parts).strip()
        if not fallback:
            fallback = f"Image related to {doc_title or 'the document'}."

        description = fallback
        if self.use_vlm_for_images and image_path:
            self._init_vlm_if_needed()
            if self._vlm_client is not None:
                metadata_lines = [
                    f"- doc_title: {doc_title or 'N/A'}",
                    f"- image_title: {image_title or 'N/A'}",
                    f"- image_caption: {caption or 'N/A'}",
                    f"- image_path: {image_path or 'N/A'}",
                ]
                prompt = (
                    f"{self.macro_image_description_prompt}\n"
                    "Return plain text only, no JSON.\n"
                    "Keep the description short and factual.\n\n"
                    "Image metadata:\n" + "\n".join(metadata_lines)
                )
                try:
                    output = self._vlm_client.chat(
                        prompt=prompt,
                        images=image_path,
                        sampling_overrides={"max_tokens": 128},
                    )
                    candidate = str(output).strip()
                    if candidate:
                        description = candidate
                except Exception as exc:
                    logger.warning(
                        "Image macro description generation failed for %s: %s",
                        image_item.get(
                            "image_id", image_item.get("id", "unknown_image")
                        ),
                        exc,
                    )

        description = " ".join(description.split())
        if self.max_macro_description_chars > 0:
            description = description[: self.max_macro_description_chars].strip()
        return description or fallback

    def extract_from_image(
        self, image_item: dict[str, Any], doc_item: dict[str, Any]
    ) -> list[dict[str, Any]]:
        triples: list[dict[str, Any]] = []
        image_id = str(
            image_item.get("image_id", image_item.get("id", "unknown_image"))
        )
        image_path = image_item.get("path_or_url") or image_item.get("path")
        caption = image_item.get("caption")
        vlm_attempted = False
        vlm_succeeded = False

        if self.use_vlm_for_images:
            self._init_vlm_if_needed()
            if (
                self._vlm_client is not None
                and isinstance(image_path, str)
                and image_path
            ):
                vlm_attempted = True
                try:
                    output = self._vlm_client.chat(
                        prompt=self._build_image_prompt(image_item, doc_item),
                        images=image_path,
                        sampling_overrides={"max_tokens": 512},
                    )
                    json_payload = extract_json_dict(output)
                    if isinstance(json_payload, dict):
                        for triple in self._safe_list(json_payload.get("triples")):
                            if isinstance(triple, list) and len(triple) == 3:
                                triples.append(
                                    {
                                        "head": triple[0],
                                        "relation": triple[1],
                                        "tail": triple[2],
                                        "modality": "image",
                                        "source_ref": image_path,
                                        "confidence": self.default_image_confidence,
                                        "image_id": image_id,
                                        "doc_id": doc_item.get(
                                            "doc_id", doc_item.get("title", "")
                                        ),
                                    }
                                )
                        if triples:
                            vlm_succeeded = True
                    if not triples:
                        preview = str(output).strip().replace("\n", " ")[:300]
                        logger.warning(
                            "Image VLM returned no valid triples for %s (preview=%s)",
                            image_id,
                            preview,
                        )
                        self._append_failure_log(
                            self._build_failure_payload(
                                reason="vlm_empty_or_invalid_triples",
                                image_item=image_item,
                                doc_item=doc_item,
                                image_path=str(image_path),
                                output_preview=str(output),
                                json_payload=json_payload,
                            )
                        )
                except Exception as exc:
                    logger.warning(
                        "Image VLM extraction failed for %s: %s", image_id, exc
                    )
                    self._append_failure_log(
                        self._build_failure_payload(
                            reason="vlm_exception",
                            image_item=image_item,
                            doc_item=doc_item,
                            image_path=str(image_path),
                            error=str(exc),
                        )
                    )
            elif self.use_vlm_for_images:
                self._append_failure_log(
                    self._build_failure_payload(
                        reason="vlm_not_attempted_no_client_or_path",
                        image_item=image_item,
                        doc_item=doc_item,
                        image_path=str(image_path or ""),
                        error="VLM client is None or image_path is empty/non-string.",
                    )
                )

        if (
            self.require_vlm_triples
            and self.use_vlm_for_images
            and (vlm_attempted or self._vlm_client is None)
        ):
            if not vlm_succeeded:
                self._append_failure_log(
                    self._build_failure_payload(
                        reason="require_vlm_triples_failed",
                        image_item=image_item,
                        doc_item=doc_item,
                        image_path=str(image_path or ""),
                        error="Strict mode requires non-empty VLM triples.",
                    )
                )
                raise RuntimeError(
                    f"VLM triples required but unavailable for image {image_id}. "
                    "Check model initialization, allowed_local_media_path, and VLM output format."
                )

        # Fallback keeps pipeline robust when VLM is disabled/unavailable.
        allow_caption_fallback = (
            not self.use_vlm_for_images
        ) or self.fallback_to_caption_when_vlm_fails
        if (
            not triples
            and allow_caption_fallback
            and isinstance(caption, str)
            and caption.strip()
        ):
            triples.append(
                {
                    "head": image_id,
                    "relation": "describes",
                    "tail": caption.strip(),
                    "modality": "image",
                    "source_ref": image_path or image_id,
                    "confidence": max(0.3, self.default_image_confidence - 0.2),
                    "image_id": image_id,
                    "doc_id": doc_item.get("doc_id", doc_item.get("title", "")),
                }
            )
        return triples

    def extract_from_table(
        self, table_item: dict[str, Any], doc_item: dict[str, Any]
    ) -> list[dict[str, Any]]:
        triples: list[dict[str, Any]] = []
        table_id = str(
            table_item.get("table_id", table_item.get("id", "unknown_table"))
        )
        headers = self._safe_list(
            table_item.get("schema", table_item.get("headers", []))
        )
        rows = self._safe_list(table_item.get("rows", []))

        if not rows:
            return triples

        def append_triple(subject: Any, relation: Any, obj: Any, row_idx: int) -> None:
            if subject is None or obj is None or relation is None:
                return
            triples.append(
                {
                    "head": str(subject),
                    "relation": str(relation),
                    "tail": str(obj),
                    "modality": "table",
                    "source_ref": f"{table_id}#row{row_idx}",
                    "confidence": self.default_table_confidence,
                    "table_id": table_id,
                    "doc_id": doc_item.get("doc_id", doc_item.get("title", "")),
                }
            )

        for idx, row in enumerate(rows):
            if isinstance(row, dict):
                keys = list(row.keys())
                if not keys:
                    continue
                subject_key = keys[0]
                subject_val = row.get(subject_key)
                for key in keys[1:]:
                    append_triple(subject_val, key, row.get(key), idx)
            elif isinstance(row, list):
                if not row:
                    continue
                if headers and len(headers) == len(row):
                    subject_val = row[0]
                    for col_idx in range(1, len(row)):
                        append_triple(subject_val, headers[col_idx], row[col_idx], idx)
                else:
                    subject_val = row[0]
                    for col_idx in range(1, len(row)):
                        append_triple(
                            subject_val, f"column_{col_idx}", row[col_idx], idx
                        )
            else:
                append_triple(table_id, "has_value", row, idx)

        return triples
