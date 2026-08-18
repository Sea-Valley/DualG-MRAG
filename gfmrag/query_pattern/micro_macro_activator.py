from __future__ import annotations

import json
import math
import os
from typing import Any

from gfmrag.kg_construction.micro_matcher import SimGRAGMicroMatcher
from gfmrag.kg_construction.utils import processing_phrases


class MicroMacroActivator:
    """Map micro matcher hits to macro seed entities for graph activation."""

    def __init__(
        self,
        micro_matcher: SimGRAGMicroMatcher,
        cross_tier_index_path: str,
        micro_kg_path: str,
        micro_match_topk: int = 10,
        max_match_score: float | None = None,
        min_match_score: float | None = None,
        node_doc_match_topk: int = 30,
        node_doc_max_distance: float | None = 1.0,
        node_doc_anchor_min_tokens: int = 1,
        node_doc_anchor_min_chars: int = 4,
        node_doc_include_tail_nodes: bool = False,
    ):
        self.micro_matcher = micro_matcher
        self.cross_tier_index_path = cross_tier_index_path
        self.micro_kg_path = micro_kg_path
        self.micro_match_topk = micro_match_topk
        self.max_match_score = (
            max_match_score if max_match_score is not None else min_match_score
        )
        self._cross_tier_index: dict[str, list[str]] | None = None
        self._micro_triple_meta: dict[str, dict[str, Any]] | None = None
        self._node_to_triple_ids: dict[str, set[str]] | None = None
        self._index_ready = False
        self.node_doc_match_topk = int(node_doc_match_topk)
        self.node_doc_max_distance = node_doc_max_distance
        self.node_doc_anchor_min_tokens = max(1, int(node_doc_anchor_min_tokens))
        self.node_doc_anchor_min_chars = max(1, int(node_doc_anchor_min_chars))
        self.node_doc_include_tail_nodes = bool(node_doc_include_tail_nodes)

    def _load_json(self, path: str) -> Any:
        with open(path, encoding="utf-8") as fin:
            return json.load(fin)

    def _load_jsonl(self, path: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def _ensure_cross_tier(self) -> dict[str, list[str]]:
        if self._cross_tier_index is not None:
            return self._cross_tier_index
        if not os.path.exists(self.cross_tier_index_path):
            self._cross_tier_index = {}
            return self._cross_tier_index
        payload = self._load_json(self.cross_tier_index_path)
        self._cross_tier_index = payload if isinstance(payload, dict) else {}
        return self._cross_tier_index

    def _ensure_matcher_index(self) -> None:
        if self._index_ready:
            return
        if not os.path.exists(self.micro_kg_path):
            raise FileNotFoundError(f"micro_kg.jsonl not found: {self.micro_kg_path}")
        micro_rows = self._load_jsonl(self.micro_kg_path)
        self._micro_triple_meta = {}
        self._node_to_triple_ids = {}
        for row in micro_rows:
            triple_id = str(row.get("triple_id", "")).strip()
            if not triple_id:
                continue
            self._micro_triple_meta[triple_id] = {
                "triple_id": triple_id,
                "doc_id": str(row.get("doc_id", "")),
                "modality": str(row.get("modality", "")),
                "source_ref": str(row.get("source_ref", "")),
                "head": str(row.get("head", "")),
                "relation": str(row.get("relation", "")),
                "tail": str(row.get("tail", "")),
            }
            for raw_node in (row.get("head", ""), row.get("tail", "")):
                node = processing_phrases(raw_node)
                if not node:
                    continue
                self._node_to_triple_ids.setdefault(node, set()).add(triple_id)
        self.micro_matcher.build_index(micro_rows)
        self._index_ready = True

    def _collect_doc_micro_scores(
        self, hits: list[dict[str, Any]], triple_meta: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, float], list[dict[str, Any]], set[str]]:
        cross_tier_index = self._ensure_cross_tier()
        activated_entities: set[str] = set()
        matched_hits: list[dict[str, Any]] = []
        doc_micro_scores: dict[str, float] = {}

        for hit in hits:
            score = float(hit.get("score", 0.0))
            if self.max_match_score is not None and score > self.max_match_score:
                continue
            safe_distance = max(0.0, score)
            hit_micro_score = 1.0 / (1.0 + safe_distance)
            matched_triple_ids = [str(x) for x in hit.get("matched_triple_ids", [])]
            matched_triples = []
            for triple_id in matched_triple_ids:
                linked_entities = cross_tier_index.get(triple_id, [])
                for entity in linked_entities:
                    activated_entities.add(entity)
                triple_info = dict(triple_meta.get(triple_id, {"triple_id": triple_id}))
                triple_info["linked_macro_entities"] = linked_entities
                doc_id = str(triple_info.get("doc_id", "")).strip()
                if doc_id:
                    prev = doc_micro_scores.get(doc_id, 0.0)
                    if hit_micro_score > prev:
                        doc_micro_scores[doc_id] = hit_micro_score
                matched_triples.append(triple_info)
            matched_hits.append(
                {
                    "score": score,
                    "doc_micro_score": hit_micro_score,
                    "matched_triple_ids": matched_triple_ids,
                    "matched_triples": matched_triples,
                }
            )
        return doc_micro_scores, matched_hits, activated_entities

    def _collect_node_anchors(
        self,
        condition_triples: list[tuple[str, str, str]],
        anchor_texts: list[str] | None = None,
    ) -> list[str]:
        anchors: list[str] = []
        seen: set[str] = set()
        for candidate in anchor_texts or []:
            node = processing_phrases(candidate)
            if (
                not node
                or "unknown" in node
                or len(node) < self.node_doc_anchor_min_chars
                or len(node.split()) < self.node_doc_anchor_min_tokens
            ):
                continue
            if node in seen:
                continue
            seen.add(node)
            anchors.append(node)
        if anchors:
            return anchors
        for head, _, tail in condition_triples:
            candidates = [head]
            if self.node_doc_include_tail_nodes:
                candidates.append(tail)
            for candidate in candidates:
                node = processing_phrases(candidate)
                if (
                    not node
                    or "unknown" in node
                    or len(node) < self.node_doc_anchor_min_chars
                    or len(node.split()) < self.node_doc_anchor_min_tokens
                ):
                    continue
                if node in seen:
                    continue
                seen.add(node)
                anchors.append(node)
        return anchors

    def _node_anchor_weight(self, anchor: str) -> float:
        token_count = len(anchor.split())
        if token_count >= 3:
            return 1.0
        if token_count == 2:
            return 0.9
        if len(anchor) >= 10:
            return 0.6
        return 0.25

    def score_docs_with_node_anchors_with_trace(
        self,
        condition_triples: list[tuple[str, str, str]],
        anchor_texts: list[str] | None = None,
    ) -> dict[str, Any]:
        if not condition_triples and not anchor_texts:
            return {"doc_micro_scores": {}, "matched_hits": [], "node_anchors": []}
        self._ensure_matcher_index()
        triple_meta = self._micro_triple_meta or {}
        node_to_triple_ids = self._node_to_triple_ids or {}

        anchors = self._collect_node_anchors(
            condition_triples, anchor_texts=anchor_texts
        )
        if not anchors:
            return {"doc_micro_scores": {}, "matched_hits": [], "node_anchors": []}

        anchor_vectors = self.micro_matcher._encode(anchors)
        raw_hits = self.micro_matcher.node_store.search(
            anchor_vectors, self.node_doc_match_topk
        )

        doc_anchor_scores: dict[str, dict[str, float]] = {}
        node_hits: list[dict[str, Any]] = []

        for anchor, hits in zip(anchors, raw_hits):
            anchor_weight = self._node_anchor_weight(anchor)
            exact_added = False
            if anchor in node_to_triple_ids:
                hits = [{"distance": 0.0, "entity": {"name": anchor}}] + hits
                exact_added = True

            seen_nodes: set[str] = set()
            for hit in hits:
                matched_node = processing_phrases(hit.get("entity", {}).get("name", ""))
                if not matched_node or matched_node in seen_nodes:
                    continue
                seen_nodes.add(matched_node)
                raw_distance = float(hit.get("distance", 0.0))
                distance = (
                    0.0
                    if matched_node == anchor and exact_added
                    else math.sqrt(max(0.0, raw_distance))
                )
                if (
                    self.node_doc_max_distance is not None
                    and distance > self.node_doc_max_distance
                ):
                    continue

                triple_ids = sorted(node_to_triple_ids.get(matched_node, set()))
                if not triple_ids:
                    continue

                hit_score = anchor_weight * (1.0 / (1.0 + distance))
                matched_triples = []
                for triple_id in triple_ids:
                    triple_info = dict(
                        triple_meta.get(triple_id, {"triple_id": triple_id})
                    )
                    matched_triples.append(triple_info)
                    doc_id = str(triple_info.get("doc_id", "")).strip()
                    if not doc_id:
                        continue
                    doc_anchor_scores.setdefault(doc_id, {})
                    prev = doc_anchor_scores[doc_id].get(anchor, 0.0)
                    if hit_score > prev:
                        doc_anchor_scores[doc_id][anchor] = hit_score

                node_hits.append(
                    {
                        "anchor": anchor,
                        "matched_node": matched_node,
                        "distance": distance,
                        "doc_micro_score": hit_score,
                        "matched_triple_ids": triple_ids,
                        "matched_triples": matched_triples,
                    }
                )

        doc_micro_scores = {
            doc_id: float(sum(anchor_scores.values()))
            for doc_id, anchor_scores in doc_anchor_scores.items()
        }
        node_hits.sort(
            key=lambda item: (
                -float(item.get("doc_micro_score", 0.0)),
                float(item.get("distance", 0.0)),
                str(item.get("matched_node", "")),
            )
        )
        return {
            "doc_micro_scores": doc_micro_scores,
            "matched_hits": node_hits,
            "node_anchors": anchors,
        }

    def score_docs_with_node_anchors(
        self,
        condition_triples: list[tuple[str, str, str]],
        anchor_texts: list[str] | None = None,
    ) -> dict[str, float]:
        payload = self.score_docs_with_node_anchors_with_trace(
            condition_triples, anchor_texts=anchor_texts
        )
        return payload["doc_micro_scores"]

    def activate(self, condition_triples: list[tuple[str, str, str]]) -> list[str]:
        payload = self.activate_with_trace(condition_triples)
        return payload["activated_entities"]

    def activate_with_trace(
        self, condition_triples: list[tuple[str, str, str]]
    ) -> dict[str, Any]:
        if not condition_triples:
            return {
                "activated_entities": [],
                "matched_hits": [],
                "doc_micro_scores": {},
            }
        self._ensure_matcher_index()
        triple_meta = self._micro_triple_meta or {}

        hits = self.micro_matcher.match_pattern(
            condition_triples,
            topk=self.micro_match_topk,
            mode="greedy",
        )
        doc_micro_scores, matched_hits, activated_entities = (
            self._collect_doc_micro_scores(hits, triple_meta)
        )
        return {
            "activated_entities": sorted(activated_entities),
            "matched_hits": matched_hits,
            "doc_micro_scores": doc_micro_scores,
        }
