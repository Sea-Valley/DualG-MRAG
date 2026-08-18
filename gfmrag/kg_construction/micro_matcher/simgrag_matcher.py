from __future__ import annotations

import copy
import hashlib
import logging
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import networkx as nx

from .base_matcher import BaseMicroMatcher
from .utils import KSmallest
from .vector_store import FaissVectorStore, MilvusVectorStore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional runtime dependency
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class SimGRAGMicroMatcher(BaseMicroMatcher):
    """SimGRAG-style semantic + topological matcher for stage2 preparation."""

    def __init__(
        self,
        embedding_model_name: str,
        embedding_device: str = "cpu",
        embedding_batch_size: int = 64,
        local_files_only: bool = True,
        vector_backend: str = "faiss",
        vector_dim: int = 768,
        faiss_storage_dir: str = "tmp/micro_matcher_faiss",
        milvus_uri: str = "http://localhost:19530",
        node_collection: str = "gfmrag_micro_nodes",
        relation_collection: str = "gfmrag_micro_relations",
        node_sim_topk: int = 64,
        relation_sim_topk: int = 32,
        final_topk: int = 20,
        timeout_sec: int = 30,
        relation_gate_node_threshold: float = 0.6,
        relation_score_weight: float = 1.0,
        reverse_edge_penalty: float = 0.25,
    ) -> None:
        self.embedding_model_name = embedding_model_name
        self.embedding_device = embedding_device
        self.embedding_batch_size = embedding_batch_size
        self.local_files_only = local_files_only
        self.vector_backend = vector_backend.strip().lower()
        self.vector_dim = int(vector_dim)
        self.faiss_storage_dir = faiss_storage_dir
        self.milvus_uri = milvus_uri
        self.node_collection = node_collection
        self.relation_collection = relation_collection
        self.node_sim_topk = node_sim_topk
        self.relation_sim_topk = relation_sim_topk
        self.final_topk = final_topk
        self.timeout_sec = timeout_sec
        self.relation_gate_node_threshold = float(relation_gate_node_threshold)
        self.relation_score_weight = float(relation_score_weight)
        self.reverse_edge_penalty = max(0.0, float(reverse_edge_penalty))

        self.node_store: MilvusVectorStore | FaissVectorStore
        self.relation_store: MilvusVectorStore | FaissVectorStore
        if self.vector_backend == "milvus":
            self.node_store = MilvusVectorStore(
                name=node_collection, uri=milvus_uri, dim=vector_dim
            )
            self.relation_store = MilvusVectorStore(
                name=relation_collection, uri=milvus_uri, dim=vector_dim
            )
        elif self.vector_backend == "faiss":
            self.node_store = FaissVectorStore(
                name=node_collection, dim=vector_dim, storage_dir=faiss_storage_dir
            )
            self.relation_store = FaissVectorStore(
                name=relation_collection, dim=vector_dim, storage_dir=faiss_storage_dir
            )
        else:
            raise ValueError(
                f"Unsupported vector_backend: {vector_backend}. Use 'faiss' or 'milvus'."
            )

        self._embedding_model: SentenceTransformer | None = None
        self.kg: dict[str, dict[str, set[str]]] = {}
        self.kg_reverse: dict[str, dict[str, set[str]]] = {}
        self.edge2triple_ids: dict[tuple[str, str, str], set[str]] = {}
        self.index_ready = False

    def _get_embedding_model(self) -> SentenceTransformer:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is required for SimGRAGMicroMatcher."
            )
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                trust_remote_code=True,
                local_files_only=self.local_files_only,
                device=self.embedding_device,
            )
        return self._embedding_model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_embedding_model()
        vectors = model.encode(
            texts,
            batch_size=self.embedding_batch_size,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def _triples_to_kg(
        self, micro_triples: list[dict[str, Any]]
    ) -> tuple[list[str], list[str]]:
        nodes: dict[str, int] = {}
        relations: dict[str, int] = {}
        self.kg = {}
        self.kg_reverse = {}
        self.edge2triple_ids = {}

        for item in micro_triples:
            head = str(item.get("head", "")).strip()
            relation = str(item.get("relation", "")).strip()
            tail = str(item.get("tail", "")).strip()
            if not head or not relation or not tail:
                continue
            triple_id = str(item.get("triple_id", "")).strip()
            if not triple_id:
                triple_id = hashlib.md5(
                    f"{head}|{relation}|{tail}".encode()
                ).hexdigest()

            nodes.setdefault(head, len(nodes))
            nodes.setdefault(tail, len(nodes))
            relations.setdefault(relation, len(relations))

            self.kg.setdefault(head, {})
            self.kg[head].setdefault(relation, set()).add(tail)
            self.kg.setdefault(tail, {})
            self.kg_reverse.setdefault(tail, {})
            self.kg_reverse[tail].setdefault(relation, set()).add(head)
            self.kg_reverse.setdefault(head, {})

            edge_key = (head, relation, tail)
            self.edge2triple_ids.setdefault(edge_key, set()).add(triple_id)
            reverse_edge_key = (tail, relation, head)
            self.edge2triple_ids.setdefault(reverse_edge_key, set()).add(triple_id)

        return list(nodes.keys()), list(relations.keys())

    def build_index(self, micro_triples: list[dict[str, Any]]) -> None:
        nodes, relations = self._triples_to_kg(micro_triples)
        if not nodes or not relations:
            self.index_ready = False
            logger.warning("Skip index build because micro triples are empty.")
            return

        node_vectors = self._encode(nodes)
        relation_vectors = self._encode(relations)

        self.node_store.reset()
        self.relation_store.reset()
        self.node_store.insert(
            [
                {"id": idx, "vector": vector, "name": name}
                for idx, (name, vector) in enumerate(zip(nodes, node_vectors))
            ]
        )
        self.relation_store.insert(
            [
                {"id": idx, "vector": vector, "name": name}
                for idx, (name, vector) in enumerate(zip(relations, relation_vectors))
            ]
        )

        self.index_ready = True
        logger.info(
            "Built micro matcher index with %s nodes and %s relations.",
            len(nodes),
            len(relations),
        )

    def _dfs_all_edges(
        self,
        query_graph: nx.Graph,
        node: str,
        visited_edges: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        neighbors_with_degrees = [
            (neighbor, query_graph.degree(neighbor)) for neighbor in query_graph[node]
        ]
        for neighbor, _ in sorted(
            neighbors_with_degrees, key=lambda item: item[1], reverse=True
        ):
            if (node, neighbor) not in visited_edges and (
                neighbor,
                node,
            ) not in visited_edges:
                visited_edges.append((node, neighbor))
                visited_edges = self._dfs_all_edges(
                    query_graph, neighbor, visited_edges
                )
            if len(visited_edges) == query_graph.number_of_edges():
                return visited_edges
        return visited_edges

    def _collect_matched_triple_ids(
        self, matched_edges: list[tuple[str, str, str]]
    ) -> list[str]:
        triple_ids: set[str] = set()
        for edge in matched_edges:
            triple_ids.update(self.edge2triple_ids.get(edge, set()))
        return sorted(triple_ids)

    def _search_similar_nodes(
        self,
        query_nodes: list[str],
        query_node_vectors: list[list[float]],
    ) -> dict[str, dict[str, float] | None]:
        if not query_nodes:
            return {}
        raw = self.node_store.search(query_node_vectors, self.node_sim_topk)
        return {
            query_node: {
                hit["entity"]["name"]: math.sqrt(float(hit["distance"]))
                for hit in hits
                if hit["entity"]["name"] in self.kg
            }
            for query_node, hits in zip(query_nodes, raw)
        }

    def _search_similar_relations(
        self,
        query_graph: nx.Graph,
        query_relations: list[str],
        query_relation_vectors: list[list[float]],
    ) -> dict[str, dict[str, float] | None]:
        similar_relations: dict[str, dict[str, float] | None] = {}
        if query_relations:
            raw = self.relation_store.search(
                query_relation_vectors, self.relation_sim_topk
            )
            for relation, hits in zip(query_relations, raw):
                similar_relations[relation] = {
                    hit["entity"]["name"]: math.sqrt(float(hit["distance"]))
                    for hit in hits
                }
        for _, _, data in query_graph.edges(data=True):
            relation = data["relation"]
            if "UNKNOWN" in relation:
                similar_relations[relation] = None
        return similar_relations

    def _relation_enabled(
        self,
        node_candidates: dict[str, float] | None,
        kg_neighbor: str,
    ) -> bool:
        # If the query node is a wildcard (UNKNOWN), relation similarity can still help.
        if node_candidates is None:
            return True
        node_distance = node_candidates.get(kg_neighbor)
        if node_distance is None:
            return False
        return node_distance <= self.relation_gate_node_threshold

    def _match_pattern_impl(
        self,
        pattern_triples: list[tuple[str, str, str]],
        topk: int = 20,
        mode: str = "greedy",
        allow_reverse_traversal: bool = False,
    ) -> list[dict[str, Any]]:
        if not self.index_ready:
            raise RuntimeError(
                "Micro matcher index is not ready. Call build_index() first."
            )

        query_graph = nx.Graph()
        for head, relation, tail in pattern_triples:
            query_graph.add_edge(head, tail, relation=relation, matched=False)
        if query_graph.number_of_edges() == 0:
            return []

        # Handle disconnected condition graph component by component.
        if not nx.is_connected(query_graph):
            merged_results: list[dict[str, Any]] = []
            for component in nx.connected_components(query_graph):
                sub_pattern = [
                    triple
                    for triple in pattern_triples
                    if triple[0] in component and triple[2] in component
                ]
                merged_results.extend(
                    self._match_pattern_impl(
                        sub_pattern,
                        topk=topk,
                        mode=mode,
                        allow_reverse_traversal=allow_reverse_traversal,
                    )
                )
            merged_results.sort(key=lambda item: item["score"])
            return merged_results[:topk]

        query_nodes = [node for node in query_graph.nodes() if "UNKNOWN" not in node]
        unknown_nodes = [node for node in query_graph.nodes() if "UNKNOWN" in node]
        query_relations = list(
            {
                data["relation"]
                for _, _, data in query_graph.edges(data=True)
                if "UNKNOWN" not in data["relation"]
            }
        )

        start_time = time.monotonic()
        query_texts = query_nodes + query_relations
        query_vectors = self._encode(query_texts)
        query_node_vectors = query_vectors[: len(query_nodes)]
        query_relation_vectors = query_vectors[len(query_nodes) :]

        with ThreadPoolExecutor() as executor:
            node_future = executor.submit(
                self._search_similar_nodes, query_nodes, query_node_vectors
            )
            relation_future = executor.submit(
                self._search_similar_relations,
                query_graph,
                query_relations,
                query_relation_vectors,
            )
            similar_nodes = node_future.result()
            similar_relations = relation_future.result()

        for node in unknown_nodes:
            similar_nodes[node] = None  # Unknown variables are wildcard placeholders.

        def _candidate_count(node: str) -> int:
            candidates = similar_nodes[node]
            return len(candidates) if candidates is not None else len(self.kg)

        root = sorted(similar_nodes.keys(), key=_candidate_count)[0]
        root_candidates = similar_nodes[root]
        if not root_candidates:
            raise RuntimeError("No similar nodes found for root query node.")

        query_edge_sequence = self._dfs_all_edges(query_graph, root, [])
        final_best_score = [0.0] * query_graph.number_of_edges()
        if mode == "greedy":
            for idx in range(query_graph.number_of_edges() - 1, 0, -1):
                query_node, query_neighbor = query_edge_sequence[idx]
                query_relation = query_graph[query_node][query_neighbor]["relation"]
                final_best_score[idx - 1] = final_best_score[idx]
                node_candidates = similar_nodes.get(query_neighbor)
                rel_scores = similar_relations.get(query_relation)
                if node_candidates:
                    min_node_score = min(node_candidates.values())
                    final_best_score[idx - 1] += min_node_score
                    if (
                        rel_scores
                        and min_node_score <= self.relation_gate_node_threshold
                    ):
                        final_best_score[idx - 1] += self.relation_score_weight * min(
                            rel_scores.values()
                        )
                elif rel_scores:
                    final_best_score[idx - 1] += self.relation_score_weight * min(
                        rel_scores.values()
                    )

        results = KSmallest(min(topk, self.final_topk))

        def check_timeout() -> None:
            if time.monotonic() - start_time > self.timeout_sec:
                raise TimeoutError("Pattern matching timeout.")

        def match(
            cur_query_idx: int,
            node_matching: dict[str, str],
            matched_kg_edges: list[tuple[str, str, str]],
            cur_score: float,
            reuse_nodes: bool,
        ) -> None:
            check_timeout()
            if cur_query_idx == query_graph.number_of_edges():
                results.add(cur_score, copy.deepcopy(matched_kg_edges), reuse_nodes)
                return

            cur_query_node, query_neighbor = query_edge_sequence[cur_query_idx]
            query_relation = query_graph[cur_query_node][query_neighbor]["relation"]
            cur_kg_node = node_matching[cur_query_node]

            to_expand: list[tuple[str, str, float, bool]] = []
            candidate_edges: list[tuple[str, str, bool]] = []
            for kg_relation, kg_neighbors in self.kg[cur_kg_node].items():
                for kg_neighbor in kg_neighbors:
                    candidate_edges.append((kg_relation, kg_neighbor, False))
            if allow_reverse_traversal:
                for kg_relation, kg_neighbors in self.kg_reverse[cur_kg_node].items():
                    for kg_neighbor in kg_neighbors:
                        candidate_edges.append((kg_relation, kg_neighbor, True))

            for kg_relation, kg_neighbor, is_reverse_edge in candidate_edges:
                node_candidates = similar_nodes.get(query_neighbor)
                if node_candidates is not None and kg_neighbor not in node_candidates:
                    continue

                next_reuse_nodes = reuse_nodes
                if query_neighbor in node_matching:
                    if node_matching[query_neighbor] != kg_neighbor:
                        continue
                    node_score = 0.0
                else:
                    if kg_neighbor in node_matching.values():
                        next_reuse_nodes = True
                    node_score = (
                        0.0
                        if node_candidates is None
                        else float(node_candidates[kg_neighbor])
                    )
                relation_candidates = similar_relations.get(query_relation)
                relation_enabled = self._relation_enabled(node_candidates, kg_neighbor)
                if (
                    relation_enabled
                    and relation_candidates is not None
                    and kg_relation not in relation_candidates
                ):
                    continue
                relation_score = 0.0
                if relation_enabled and relation_candidates is not None:
                    relation_score = self.relation_score_weight * float(
                        relation_candidates[kg_relation]
                    )
                if is_reverse_edge:
                    relation_score += self.reverse_edge_penalty
                next_score = cur_score + node_score + relation_score
                to_expand.append(
                    (
                        kg_relation,
                        kg_neighbor,
                        next_score,
                        next_reuse_nodes,
                    )
                )

            if mode == "greedy":
                to_expand.sort(key=lambda item: item[2])
                for kg_relation, kg_neighbor, next_score, next_reuse_nodes in to_expand:
                    if (
                        next_score + final_best_score[cur_query_idx]
                        > results.max_score()
                    ):
                        break
                    next_node_matching = copy.deepcopy(node_matching)
                    next_node_matching[query_neighbor] = kg_neighbor
                    next_edges = matched_kg_edges + [
                        (cur_kg_node, kg_relation, kg_neighbor)
                    ]
                    match(
                        cur_query_idx + 1,
                        next_node_matching,
                        next_edges,
                        next_score,
                        next_reuse_nodes,
                    )
            else:
                for kg_relation, kg_neighbor, next_score, next_reuse_nodes in to_expand:
                    next_node_matching = copy.deepcopy(node_matching)
                    next_node_matching[query_neighbor] = kg_neighbor
                    next_edges = matched_kg_edges + [
                        (cur_kg_node, kg_relation, kg_neighbor)
                    ]
                    match(
                        cur_query_idx + 1,
                        next_node_matching,
                        next_edges,
                        next_score,
                        next_reuse_nodes,
                    )

        sorted_root_candidates = sorted(
            root_candidates.items(), key=lambda item: item[1]
        )
        for kg_node, distance in sorted_root_candidates:
            match(0, {root: kg_node}, [], float(distance), False)

        payload: list[dict[str, Any]] = []
        for score, matched_edges, reuse_nodes in results.get():
            payload.append(
                {
                    "score": float(score),
                    "matched_edges": matched_edges,
                    "reuse_nodes": bool(reuse_nodes),
                    "matched_triple_ids": self._collect_matched_triple_ids(
                        matched_edges
                    ),
                }
            )
        payload.sort(key=lambda item: item["score"])
        return payload[:topk]

    def match_pattern(
        self,
        pattern_triples: list[tuple[str, str, str]],
        topk: int = 20,
        mode: str = "greedy",
    ) -> list[dict[str, Any]]:
        hits = self._match_pattern_impl(
            pattern_triples,
            topk=topk,
            mode=mode,
            allow_reverse_traversal=False,
        )
        if hits:
            return hits
        logger.info(
            "No forward-only match found; fallback to reverse traversal with penalty=%.3f",
            self.reverse_edge_penalty,
        )
        return self._match_pattern_impl(
            pattern_triples,
            topk=topk,
            mode=mode,
            allow_reverse_traversal=True,
        )

    def _normalize_candidate_triples(
        self, candidate_triples: list[tuple[str, str, str] | dict[str, Any]]
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, item in enumerate(candidate_triples):
            if isinstance(item, dict):
                head = str(item.get("head", "")).strip()
                relation = str(item.get("relation", "")).strip()
                tail = str(item.get("tail", "")).strip()
                triple_id = str(item.get("triple_id", f"score_only_{idx}")).strip()
            else:
                try:
                    head, relation, tail = item
                except Exception as exc:  # pragma: no cover - defensive branch
                    raise ValueError(
                        "Each candidate triple must be a dict with head/relation/tail "
                        "or a tuple/list of (head, relation, tail)."
                    ) from exc
                head = str(head).strip()
                relation = str(relation).strip()
                tail = str(tail).strip()
                triple_id = f"score_only_{idx}"

            if not head or not relation or not tail:
                continue
            rows.append(
                {
                    "triple_id": triple_id,
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                }
            )
        return rows

    def score_only(
        self,
        pattern_triples: list[tuple[str, str, str]],
        candidate_triples: list[tuple[str, str, str] | dict[str, Any]],
        mode: str = "greedy",
    ) -> float:
        """
        Return only the best SimGRAG matching score for given candidate triples.

        This method intentionally runs the same build_index + match_pattern pipeline
        on a temporary matcher, so the scoring behavior is identical to main matching.
        """
        if not pattern_triples:
            return 0.0
        if not candidate_triples:
            return math.inf

        normalized_candidates = self._normalize_candidate_triples(candidate_triples)
        if not normalized_candidates:
            return math.inf

        suffix = uuid.uuid4().hex[:10]
        tmp_matcher = SimGRAGMicroMatcher(
            embedding_model_name=self.embedding_model_name,
            embedding_device=self.embedding_device,
            embedding_batch_size=self.embedding_batch_size,
            local_files_only=self.local_files_only,
            vector_backend=self.vector_backend,
            vector_dim=self.vector_dim,
            faiss_storage_dir=self.faiss_storage_dir,
            milvus_uri=self.milvus_uri,
            node_collection=f"{self.node_collection}_score_only_{suffix}",
            relation_collection=f"{self.relation_collection}_score_only_{suffix}",
            node_sim_topk=self.node_sim_topk,
            relation_sim_topk=self.relation_sim_topk,
            final_topk=max(1, self.final_topk),
            timeout_sec=self.timeout_sec,
            relation_gate_node_threshold=self.relation_gate_node_threshold,
            relation_score_weight=self.relation_score_weight,
            reverse_edge_penalty=self.reverse_edge_penalty,
        )
        tmp_matcher._embedding_model = self._get_embedding_model()

        try:
            tmp_matcher.build_index(normalized_candidates)
            hits = tmp_matcher.match_pattern(pattern_triples, topk=1, mode=mode)
            if not hits:
                return math.inf
            return float(hits[0].get("score", math.inf))
        finally:
            if self.vector_backend == "milvus":
                for store in (tmp_matcher.node_store, tmp_matcher.relation_store):
                    try:
                        store.client.drop_collection(collection_name=store.name)  # type: ignore[union-attr]
                    except Exception:
                        pass
            else:
                for store in (tmp_matcher.node_store, tmp_matcher.relation_store):
                    try:
                        store.reset()
                    except Exception:
                        pass
