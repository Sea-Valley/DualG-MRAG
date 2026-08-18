# mypy: ignore-errors

import json
import logging
import math
import os
import re

import torch
from torch import distributed as dist

from gfmrag.ultra import variadic

logger = logging.getLogger(__name__)

_NORMALIZE_ENTITY_RE = re.compile(r"[^0-9a-z]+")


def normalize_entity_text(text: str) -> str:
    value = str(text).strip().lower()
    value = _NORMALIZE_ENTITY_RE.sub(" ", value)
    return " ".join(value.split())


class DocumentRetriever:
    """
    Return documents based on document ranking
    """

    def __init__(self, docs: dict, id2doc: dict) -> None:
        self.docs = docs
        self.id2doc = id2doc

    def __call__(self, doc_ranking: torch.Tensor, top_k: int = 1) -> list:
        k = min(int(top_k), int(doc_ranking.shape[0]))
        top_k_docs = doc_ranking.topk(k).indices
        norm_doc_scors = mini_max_scale(doc_ranking)
        return [
            {
                "title": self.id2doc[doc.item()],
                "content": self.docs[self.id2doc[doc.item()]],
                "score": doc_ranking[doc].item(),
                "norm_score": norm_doc_scors[doc].item(),
            }
            for doc in top_k_docs
        ]


class DualBranchDocumentRetriever:
    """
    Merge text/table branch and image branch document rankings.
    """

    def __init__(
        self,
        docs: dict,
        id2doc: dict,
        image_doc_ids: set[str] | None = None,
    ) -> None:
        self.docs = docs
        self.id2doc = id2doc
        self.image_doc_ids = image_doc_ids or set()
        self._cached_stage1_dir: str | None = None
        self._cached_doc2micro: dict[str, list[str]] = {}
        self._cached_micro_by_id: dict[str, dict] = {}
        self._cached_text_doc2entities: dict[str, list[str]] = {}
        self._cached_image_doc2entities: dict[str, list[str]] = {}
        self._cached_equivalent_dir: str | None = None
        self._cached_entity_alias_canonical: dict[str, str] = {}

    def _ensure_equivalent_resources(self, stage1_dir: str | None) -> dict[str, str]:
        if not stage1_dir:
            if self._cached_equivalent_dir != "<empty>":
                logger.warning(
                    "stage1_dir not set; equivalent-entity resources disabled."
                )
                self._cached_equivalent_dir = "<empty>"
                self._cached_entity_alias_canonical = {}
            return self._cached_entity_alias_canonical
        base_dir = os.path.abspath(stage1_dir)
        if self._cached_equivalent_dir == base_dir:
            return self._cached_entity_alias_canonical

        kg_path = os.path.join(base_dir, "kg.txt")
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            root = parent.setdefault(x, x)
            while parent[root] != root:
                parent[root] = parent[parent[root]]
                root = parent[root]
            while parent[x] != x:
                next_x = parent[x]
                parent[x] = root
                x = next_x
            return root

        def union(a: str, b: str) -> None:
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

        if os.path.exists(kg_path):
            with open(kg_path, encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",", 2)
                    if len(parts) != 3:
                        continue
                    head, relation, tail = (part.strip() for part in parts)
                    if relation != "equivalent":
                        continue
                    head_norm = normalize_entity_text(head)
                    tail_norm = normalize_entity_text(tail)
                    if head_norm and tail_norm and head_norm != tail_norm:
                        union(head_norm, tail_norm)

        canonical_map: dict[str, str] = {}
        for entity in list(parent.keys()):
            canonical_map[entity] = find(entity)

        self._cached_equivalent_dir = base_dir
        self._cached_entity_alias_canonical = canonical_map
        return canonical_map

    def _entity_lookup_keys(
        self,
        entity: str,
        alias_canonical: dict[str, str],
    ) -> set[str]:
        raw = str(entity).strip()
        normalized = normalize_entity_text(raw)
        keys = set()
        if raw:
            keys.add(raw)
        if normalized:
            keys.add(normalized)
            canonical = alias_canonical.get(normalized)
            if canonical:
                keys.add(canonical)
        return keys

    def _ensure_micro_resources(
        self, stage1_dir: str | None
    ) -> tuple[dict[str, list[str]], dict[str, dict]]:
        if not stage1_dir:
            if self._cached_stage1_dir != "<empty>":
                logger.warning("stage1_dir not set; micro-graph resources disabled.")
                self._cached_stage1_dir = "<empty>"
                self._cached_doc2micro = {}
                self._cached_micro_by_id = {}
            return self._cached_doc2micro, self._cached_micro_by_id
        base_dir = os.path.abspath(stage1_dir)
        if self._cached_stage1_dir == base_dir:
            return self._cached_doc2micro, self._cached_micro_by_id

        doc2micro_path = os.path.join(base_dir, "doc2micro.json")
        micro_kg_path = os.path.join(base_dir, "micro_kg.jsonl")
        doc2micro: dict[str, list[str]] = {}
        micro_by_id: dict[str, dict] = {}

        if os.path.exists(doc2micro_path):
            with open(doc2micro_path, encoding="utf-8") as fin:
                payload = json.load(fin)
            if isinstance(payload, dict):
                for doc_id, triple_ids in payload.items():
                    if isinstance(triple_ids, list):
                        doc2micro[str(doc_id)] = [str(tid) for tid in triple_ids]

        if os.path.exists(micro_kg_path):
            with open(micro_kg_path, encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    triple_id = str(row.get("triple_id", "")).strip()
                    if triple_id:
                        micro_by_id[triple_id] = row

        self._cached_stage1_dir = base_dir
        self._cached_doc2micro = doc2micro
        self._cached_micro_by_id = micro_by_id
        return doc2micro, micro_by_id

    def _ensure_doc_entity_resources(
        self, stage1_dir: str | None
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        if not stage1_dir:
            if self._cached_stage1_dir != "<empty>":
                logger.warning(
                    "stage1_dir not set; document-entity resources disabled."
                )
                self._cached_stage1_dir = "<empty>"
                self._cached_text_doc2entities = {}
                self._cached_image_doc2entities = {}
            return self._cached_text_doc2entities, self._cached_image_doc2entities
        base_dir = os.path.abspath(stage1_dir)
        if (
            self._cached_stage1_dir == base_dir
            and self._cached_text_doc2entities
            and self._cached_image_doc2entities is not None
        ):
            return self._cached_text_doc2entities, self._cached_image_doc2entities

        text_path = os.path.join(base_dir, "document2entities.json")
        image_path = os.path.join(base_dir, "mm_document2entities.json")
        text_doc2entities: dict[str, list[str]] = {}
        image_doc2entities: dict[str, list[str]] = {}

        if os.path.exists(text_path):
            with open(text_path, encoding="utf-8") as fin:
                payload = json.load(fin)
            if isinstance(payload, dict):
                for doc_id, entities in payload.items():
                    if isinstance(entities, list):
                        text_doc2entities[str(doc_id)] = [
                            str(entity) for entity in entities
                        ]

        if os.path.exists(image_path):
            with open(image_path, encoding="utf-8") as fin:
                payload = json.load(fin)
            if isinstance(payload, dict):
                for doc_id, entities in payload.items():
                    if isinstance(entities, list):
                        image_doc2entities[str(doc_id)] = [
                            str(entity) for entity in entities
                        ]

        self._cached_stage1_dir = base_dir
        self._cached_text_doc2entities = text_doc2entities
        self._cached_image_doc2entities = image_doc2entities
        return text_doc2entities, image_doc2entities

    def _build_doc_path_relations(
        self,
        selected_doc_ids: list[str],
        stage1_dir: str | None,
        explain_record: dict | None,
        max_neighbors_per_doc: int = 5,
        max_evidence_per_neighbor: int = 3,
        score_field: str = "viterbi_score",
        bridge_max_window: int = 2,
        bridge_decay: float = 0.4,
    ) -> dict[str, dict]:
        doc_link_payload = {
            "per_doc": {
                doc_id: {
                    "path_relation_score": 0.0,
                    "path_relation_neighbor_count": 0,
                    "path_relation_neighbors": [],
                }
                for doc_id in selected_doc_ids
            },
            "global_links": [],
        }

        if not explain_record:
            return doc_link_payload

        text_doc2entities, image_doc2entities = self._ensure_doc_entity_resources(
            stage1_dir
        )
        alias_canonical = self._ensure_equivalent_resources(stage1_dir)
        doc_entities: dict[str, set[str]] = {}
        entity_to_docs: dict[str, set[str]] = {}
        for doc_id in selected_doc_ids:
            entities = set(text_doc2entities.get(doc_id, [])) | set(
                image_doc2entities.get(doc_id, [])
            )
            doc_entities[doc_id] = entities
            for entity in entities:
                for key in self._entity_lookup_keys(entity, alias_canonical):
                    entity_to_docs.setdefault(key, set()).add(doc_id)

        pair_scores: dict[tuple[str, str], float] = {}
        pair_hop_counts: dict[tuple[str, str], int] = {}
        pair_evidence: dict[tuple[str, str], list[dict]] = {}
        targets = explain_record.get("targets", [])
        for target in targets if isinstance(targets, list) else []:
            if not isinstance(target, dict):
                continue
            candidate_paths = target.get("paths", [])
            normalized_paths: list[dict] = []
            if isinstance(candidate_paths, list) and candidate_paths:
                for path_item in candidate_paths:
                    if not isinstance(path_item, dict):
                        continue
                    one_path = path_item.get("path", [])
                    if isinstance(one_path, list) and one_path:
                        normalized_paths.append(path_item)
            else:
                one_path = target.get("path", [])
                if isinstance(one_path, list) and one_path:
                    normalized_paths.append(
                        {
                            "rank": 0,
                            "score": target.get(
                                score_field, target.get("path_mass", 0.0)
                            ),
                            "path": one_path,
                        }
                    )

            if not normalized_paths:
                continue
            for path_item in normalized_paths:
                path = path_item.get("path", [])
                if not isinstance(path, list) or not path:
                    continue
                raw_score = path_item.get(
                    "score", target.get(score_field, target.get("path_mass", 0.0))
                )
                try:
                    path_score = float(raw_score)
                except (TypeError, ValueError):
                    path_score = 0.0
                if not math.isfinite(path_score) or path_score <= 0:
                    path_score = 1.0
                hop_weight = path_score / max(len(path), 1)

                for hop_idx, hop in enumerate(path):
                    if not isinstance(hop, dict):
                        continue
                    head = str(hop.get("head", "")).strip()
                    tail = str(hop.get("tail", "")).strip()
                    relation = str(hop.get("relation", "")).strip()
                    if not head or not tail:
                        continue
                    head_docs: set[str] = set()
                    tail_docs: set[str] = set()
                    for key in self._entity_lookup_keys(head, alias_canonical):
                        head_docs.update(entity_to_docs.get(key, set()))
                    for key in self._entity_lookup_keys(tail, alias_canonical):
                        tail_docs.update(entity_to_docs.get(key, set()))
                    if not head_docs or not tail_docs:
                        continue
                    for doc_a in head_docs:
                        for doc_b in tail_docs:
                            if doc_a == doc_b:
                                continue
                            key = tuple(sorted((doc_a, doc_b)))
                            pair_scores[key] = pair_scores.get(key, 0.0) + hop_weight
                            pair_hop_counts[key] = pair_hop_counts.get(key, 0) + 1
                            pair_evidence.setdefault(key, []).append(
                                {
                                    "hop_index": hop_idx,
                                    "head": head,
                                    "relation": relation,
                                    "tail": tail,
                                    "target_entity": str(
                                        target.get("target_entity", "")
                                    ),
                                    "path_rank": int(path_item.get("rank", 0)),
                                    "hop_score": hop_weight,
                                    "match_type": "direct_hop",
                                }
                            )

                if len(path) >= 2 and bridge_max_window >= 2:
                    for window_size in range(2, max(2, int(bridge_max_window)) + 1):
                        if len(path) < window_size:
                            continue
                        window_weight = hop_weight * float(bridge_decay) ** (
                            window_size - 1
                        )
                        if window_weight <= 0:
                            continue
                        for start_idx in range(0, len(path) - window_size + 1):
                            start_hop = path[start_idx]
                            end_hop = path[start_idx + window_size - 1]
                            if not isinstance(start_hop, dict) or not isinstance(
                                end_hop, dict
                            ):
                                continue
                            bridge_head = str(start_hop.get("head", "")).strip()
                            bridge_tail = str(end_hop.get("tail", "")).strip()
                            if not bridge_head or not bridge_tail:
                                continue
                            bridge_head_docs: set[str] = set()
                            bridge_tail_docs: set[str] = set()
                            for key in self._entity_lookup_keys(
                                bridge_head, alias_canonical
                            ):
                                bridge_head_docs.update(entity_to_docs.get(key, set()))
                            for key in self._entity_lookup_keys(
                                bridge_tail, alias_canonical
                            ):
                                bridge_tail_docs.update(entity_to_docs.get(key, set()))
                            if not bridge_head_docs or not bridge_tail_docs:
                                continue
                            bridge_relations = [
                                str(path[idx].get("relation", "")).strip()
                                for idx in range(start_idx, start_idx + window_size)
                                if isinstance(path[idx], dict)
                            ]
                            for doc_a in bridge_head_docs:
                                for doc_b in bridge_tail_docs:
                                    if doc_a == doc_b:
                                        continue
                                    key = tuple(sorted((doc_a, doc_b)))
                                    pair_scores[key] = (
                                        pair_scores.get(key, 0.0) + window_weight
                                    )
                                    pair_hop_counts[key] = (
                                        pair_hop_counts.get(key, 0) + 1
                                    )
                                    pair_evidence.setdefault(key, []).append(
                                        {
                                            "hop_index": start_idx,
                                            "head": bridge_head,
                                            "relation": " => ".join(bridge_relations),
                                            "tail": bridge_tail,
                                            "target_entity": str(
                                                target.get("target_entity", "")
                                            ),
                                            "path_rank": int(path_item.get("rank", 0)),
                                            "hop_score": window_weight,
                                            "match_type": f"window_{window_size}",
                                        }
                                    )

        sorted_links = sorted(
            pair_scores.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )

        doc_neighbors: dict[str, list[dict]] = {
            doc_id: [] for doc_id in selected_doc_ids
        }
        global_links: list[dict] = []
        for (doc_a, doc_b), score in sorted_links:
            evidence = sorted(
                pair_evidence.get((doc_a, doc_b), []),
                key=lambda item: (
                    -float(item.get("hop_score", 0.0)),
                    int(item.get("hop_index", 0)),
                ),
            )[: max(1, int(max_evidence_per_neighbor))]
            link_payload = {
                "doc_a": doc_a,
                "doc_b": doc_b,
                "score": float(score),
                "hop_count": int(pair_hop_counts.get((doc_a, doc_b), 0)),
                "evidence": evidence,
            }
            global_links.append(link_payload)
            doc_neighbors.setdefault(doc_a, []).append(
                {
                    "doc_id": doc_b,
                    "score": float(score),
                    "hop_count": int(pair_hop_counts.get((doc_a, doc_b), 0)),
                    "evidence": evidence,
                }
            )
            doc_neighbors.setdefault(doc_b, []).append(
                {
                    "doc_id": doc_a,
                    "score": float(score),
                    "hop_count": int(pair_hop_counts.get((doc_a, doc_b), 0)),
                    "evidence": evidence,
                }
            )

        for doc_id in selected_doc_ids:
            neighbors = sorted(
                doc_neighbors.get(doc_id, []),
                key=lambda item: (
                    -float(item.get("score", 0.0)),
                    str(item.get("doc_id", "")),
                ),
            )[: max(1, int(max_neighbors_per_doc))]
            total_score = float(
                sum(float(item.get("score", 0.0)) for item in neighbors)
            )
            doc_link_payload["per_doc"][doc_id] = {
                "path_relation_score": total_score,
                "path_relation_neighbor_count": len(neighbors),
                "path_relation_neighbors": neighbors,
            }

        doc_link_payload["global_links"] = global_links[
            : max(1, int(max_neighbors_per_doc))
        ]
        return doc_link_payload

    def __call__(
        self,
        text_doc_ranking: torch.Tensor,
        image_doc_ranking: torch.Tensor,
        top_k: int = 1,
        max_image_docs: int | None = None,
        doc_micro_scores: dict[str, float] | None = None,
        apply_image_fusion: bool = False,
        non_candidate_scale: float = 1.0 / 3.0,
        condition_triples: list[tuple[str, str, str]] | None = None,
        micro_matcher: object | None = None,
        stage1_dir: str | None = None,
        enable_path_relation: bool = False,
        explain_record: dict | None = None,
        path_relation_max_neighbors: int = 5,
        path_relation_max_evidence: int = 3,
        path_relation_score_field: str = "viterbi_score",
        path_relation_bridge_max_window: int = 2,
        path_relation_bridge_decay: float = 0.4,
    ) -> list:
        text_norm_scores = mini_max_scale(text_doc_ranking)
        image_norm_scores = mini_max_scale(image_doc_ranking)
        fused_image_norm_scores = image_norm_scores
        micro_scores = torch.zeros_like(image_doc_ranking)

        candidate_mask = torch.zeros_like(image_doc_ranking, dtype=torch.bool)

        if apply_image_fusion:
            doc_micro_scores = doc_micro_scores or {}
            for doc, doc_id in self.id2doc.items():
                doc_key = str(doc_id)
                if doc_key in doc_micro_scores:
                    micro_scores[doc] = float(doc_micro_scores[doc_key])
                    candidate_mask[doc] = True

            fused_image_norm_scores = image_norm_scores * float(non_candidate_scale)
            if bool(candidate_mask.any()):
                candidate_scores = micro_scores[candidate_mask]
                min_val = candidate_scores.min()
                max_val = candidate_scores.max()
                denom = max_val - min_val
                if torch.isclose(
                    denom,
                    torch.tensor(
                        0.0,
                        device=candidate_scores.device,
                        dtype=candidate_scores.dtype,
                    ),
                ):
                    normalized = torch.ones_like(candidate_scores)
                else:
                    normalized = (candidate_scores - min_val) / denom
                fused_image_norm_scores[candidate_mask] = normalized

        image_norm_scores = fused_image_norm_scores
        merged_norm_scores = torch.maximum(text_norm_scores, image_norm_scores)
        merged_scores = torch.maximum(text_doc_ranking, image_doc_ranking)
        ranked_docs = merged_norm_scores.argsort(descending=True)
        k = min(int(top_k), int(merged_norm_scores.shape[0]))
        image_cap = None if max_image_docs is None else max(0, int(max_image_docs))
        selected_docs: list[int] = []
        image_count = 0

        for doc in ranked_docs.tolist():
            doc_id = str(self.id2doc[doc])
            is_image_candidate = (
                doc_id in self.image_doc_ids
                if self.image_doc_ids
                else (image_doc_ranking[doc].item() > 0)
            )
            if (
                image_cap is not None
                and is_image_candidate
                and image_count >= image_cap
            ):
                continue
            selected_docs.append(doc)
            if is_image_candidate:
                image_count += 1
            if len(selected_docs) >= k:
                break

        if len(selected_docs) < k:
            selected_set = set(selected_docs)
            for doc in ranked_docs.tolist():
                if doc in selected_set:
                    continue
                selected_docs.append(doc)
                if len(selected_docs) >= k:
                    break

        selected_doc_ids = [str(self.id2doc[doc]) for doc in selected_docs]
        doc_path_relations = {
            "per_doc": {},
            "global_links": [],
        }
        if enable_path_relation:
            doc_path_relations = self._build_doc_path_relations(
                selected_doc_ids=selected_doc_ids,
                stage1_dir=stage1_dir,
                explain_record=explain_record,
                max_neighbors_per_doc=path_relation_max_neighbors,
                max_evidence_per_neighbor=path_relation_max_evidence,
                score_field=path_relation_score_field,
                bridge_max_window=path_relation_bridge_max_window,
                bridge_decay=path_relation_bridge_decay,
            )

        return [
            {
                "title": self.id2doc[doc],
                "content": self.docs[self.id2doc[doc]],
                "score": merged_scores[doc].item(),
                "norm_score": merged_norm_scores[doc].item(),
                "text_score": text_doc_ranking[doc].item(),
                "image_score": image_doc_ranking[doc].item(),
                "text_norm_score": text_norm_scores[doc].item(),
                "image_norm_score": image_norm_scores[doc].item(),
                "is_image_candidate": (
                    str(self.id2doc[doc]) in self.image_doc_ids
                    if self.image_doc_ids
                    else (image_doc_ranking[doc].item() > 0)
                ),
                "image_fusion_applied": bool(apply_image_fusion),
                "micro_score": micro_scores[doc].item(),
                "micro_in_candidate_set": bool(candidate_mask[doc].item()),
                "path_relation_score": float(
                    doc_path_relations["per_doc"]
                    .get(str(self.id2doc[doc]), {})
                    .get("path_relation_score", 0.0)
                ),
                "path_relation_neighbor_count": int(
                    doc_path_relations["per_doc"]
                    .get(str(self.id2doc[doc]), {})
                    .get("path_relation_neighbor_count", 0)
                ),
                "path_relation_neighbors": doc_path_relations["per_doc"]
                .get(str(self.id2doc[doc]), {})
                .get("path_relation_neighbors", []),
                "path_relation_enabled": bool(enable_path_relation),
                "path_relation_sample_match": bool(explain_record is not None)
                if enable_path_relation
                else False,
                "path_relation_global_links": doc_path_relations.get(
                    "global_links", []
                ),
            }
            for doc in selected_docs
        ]


def mini_max_scale(tensor):
    if tensor.numel() == 0:
        return tensor
    min_value = tensor.min()
    max_value = tensor.max()
    denom = max_value - min_value
    if torch.isclose(
        denom, torch.tensor(0.0, device=tensor.device, dtype=tensor.dtype)
    ):
        return torch.zeros_like(tensor)
    return (tensor - min_value) / denom


def entities_to_mask(entities, num_nodes):
    mask = torch.zeros(num_nodes)
    mask[entities] = 1
    return mask


def evaluate(pred, target, metrics):
    ranking, num_pred = pred
    answer_ranking, num_hard = target

    metric = {}
    for _metric in metrics:
        if _metric == "mrr":
            answer_score = 1 / ranking.float()
            query_score = variadic.variadic_mean(answer_score, num_hard)
        elif _metric.startswith("recall@"):
            threshold = int(_metric[7:])
            answer_score = (ranking <= threshold).float()
            query_score = (
                variadic.variadic_sum(answer_score, num_hard) / num_hard.float()
            )
        elif _metric.startswith("hits@"):
            threshold = int(_metric[5:])
            answer_score = (ranking <= threshold).float()
            query_score = variadic.variadic_mean(answer_score, num_hard)
        elif _metric == "mape":
            query_score = (num_pred - num_hard).abs() / (num_hard).float()
        else:
            raise ValueError(f"Unknown metric `{_metric}`")

        score = query_score.mean()
        name = _metric
        metric[name] = score.item()

    return metric


def gather_results(pred, target, rank, world_size, device):
    ranking, num_pred = pred
    answer_ranking, num_target = target

    all_size_r = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_ar = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_p = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_r[rank] = len(ranking)
    all_size_ar[rank] = len(answer_ranking)
    all_size_p[rank] = len(num_pred)
    if world_size > 1:
        dist.all_reduce(all_size_r, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_ar, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_p, op=dist.ReduceOp.SUM)

    cum_size_r = all_size_r.cumsum(0)
    cum_size_ar = all_size_ar.cumsum(0)
    cum_size_p = all_size_p.cumsum(0)

    all_ranking = torch.zeros(all_size_r.sum(), dtype=torch.long, device=device)
    all_num_pred = torch.zeros(all_size_p.sum(), dtype=torch.long, device=device)
    all_answer_ranking = torch.zeros(all_size_ar.sum(), dtype=torch.long, device=device)
    all_num_target = torch.zeros(all_size_p.sum(), dtype=torch.long, device=device)

    all_ranking[cum_size_r[rank] - all_size_r[rank] : cum_size_r[rank]] = ranking
    all_num_pred[cum_size_p[rank] - all_size_p[rank] : cum_size_p[rank]] = num_pred
    all_answer_ranking[cum_size_ar[rank] - all_size_ar[rank] : cum_size_ar[rank]] = (
        answer_ranking
    )
    all_num_target[cum_size_p[rank] - all_size_p[rank] : cum_size_p[rank]] = num_target

    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_pred, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_answer_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_target, op=dist.ReduceOp.SUM)

    return (all_ranking.cpu(), all_num_pred.cpu()), (
        all_answer_ranking.cpu(),
        all_num_target.cpu(),
    )


def batch_evaluate(pred, target, limit_nodes=None):
    num_target = target.sum(dim=-1)

    answer2query = torch.repeat_interleave(num_target)

    num_entity = pred.shape[-1]

    # in inductive (e) fb_ datasets, the number of nodes in the graph structure might exceed
    # the actual number of nodes in the graph, so we'll mask unused nodes
    if limit_nodes is not None:
        keep_mask = torch.zeros(num_entity, dtype=torch.bool, device=limit_nodes.device)
        keep_mask[limit_nodes] = 1
        pred[:, ~keep_mask] = float("-inf")

    order = pred.argsort(dim=-1, descending=True)

    range = torch.arange(num_entity, device=pred.device)
    ranking = variadic.native_scatter(
        range.expand_as(order), order, dim=-1, reduce="sum"
    )

    target_ranking = ranking[target]
    # unfiltered rankings of all answers
    order_among_answer = variadic.variadic_sort(target_ranking, num_target)[1]
    order_among_answer = (
        order_among_answer + (num_target.cumsum(0) - num_target)[answer2query]
    )

    ranking_among_answer = variadic.native_scatter(
        variadic.variadic_arange(num_target), order_among_answer, reduce="sum"
    )

    # filtered rankings of all answers
    ranking = target_ranking - ranking_among_answer + 1
    ends = num_target.cumsum(0)
    starts = ends - num_target
    hard_mask = variadic.multi_slice_mask(starts, ends, ends[-1])
    # filtered rankings of hard answers
    ranking = ranking[hard_mask]

    return ranking, target_ranking
