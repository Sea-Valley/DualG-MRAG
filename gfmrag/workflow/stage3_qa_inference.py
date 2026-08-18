import json
import logging
import math
import os
import time
from multiprocessing.dummy import Pool as ThreadPool
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils import data as torch_data
from torch.utils.data import Dataset
from tqdm import tqdm

from gfmrag import utils
from gfmrag.datasets import QADataset
from gfmrag.evaluation import RetrievalEvaluator
from gfmrag.kg_construction.micro_matcher import SimGRAGMicroMatcher
from gfmrag.llms import BaseLanguageModel
from gfmrag.llms.qwenvl_vllm import get_or_create_qwen_vl_client
from gfmrag.prompt_builder import QAPromptBuilder
from gfmrag.query_pattern import (
    MicroMacroActivator,
    SimGRAGQueryParser,
    run_visual_budget_if_enabled,
)
from gfmrag.ultra import query_utils
from gfmrag.utils.qa_utils import entities_to_mask
from gfmrag.workflow.qa_final_eval import run_final_eval

logger = logging.getLogger(__name__)


def _maybe_cuda_sync(device: torch.device) -> None:
    """Synchronize CUDA device for accurate stage timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _normalize_text(value: Any) -> str:
    return str(value).strip().lower()


def _to_nonnegative_int(value: Any, default: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number):
        return number
    return None


def _serialize_path_triplets(
    path_triplets: list[tuple[int, int, int]],
    id2ent: dict[int, str],
    id2rel: dict[int, str],
) -> list[dict[str, Any]]:
    return [
        {
            "head_id": int(head),
            "head": id2ent[int(head)],
            "relation_id": int(relation),
            "relation": id2rel[int(relation)],
            "tail_id": int(tail),
            "tail": id2ent[int(tail)],
        }
        for head, tail, relation in path_triplets
    ]


def _serialize_path_targets(
    paths_results: dict[int, dict[str, Any]],
    id2ent: dict[int, str],
    id2rel: dict[int, str],
) -> list[dict[str, Any]]:
    serialized_targets: list[dict[str, Any]] = []
    for target_id, result in paths_results.items():
        path = _serialize_path_triplets(result["path"], id2ent, id2rel)
        serialized_paths: list[dict[str, Any]] = []
        for path_item in result.get("paths", []):
            if not isinstance(path_item, dict):
                continue
            serialized_paths.append(
                {
                    "rank": int(path_item.get("rank", len(serialized_paths))),
                    "score": _safe_float(path_item.get("score", 0.0)),
                    "score_log": _safe_float(path_item.get("score_log", float("-inf"))),
                    "path": _serialize_path_triplets(
                        path_item.get("path", []), id2ent, id2rel
                    ),
                }
            )
        serialized_targets.append(
            {
                "target_id": int(target_id),
                "target_entity": id2ent[int(target_id)],
                "path": path,
                "paths": serialized_paths,
                "viterbi_score": _safe_float(result["viterbi_score"]),
                "viterbi_score_log": _safe_float(result["viterbi_score_log"]),
                "path_mass": _safe_float(result["path_mass"]),
                "path_mass_log": _safe_float(result["path_mass_log"]),
                "target_score": _safe_float(result["target_score"]),
                "best_hop": int(result["best_hop"]),
            }
        )
    return serialized_targets


def _slice_batch_sample(batch: dict, row_idx: int, batch_size: int) -> dict:
    sample: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.shape[:1] == (batch_size,):
            sample[key] = value[row_idx : row_idx + 1]
        else:
            sample[key] = value
    return sample


@torch.inference_mode()
def _decode_sample_paths(
    cfg: DictConfig,
    sample: dict,
    model: nn.Module,
    graph: Any,
    entities_weight: torch.Tensor | None,
) -> dict[int, dict[str, Any]]:
    decode_mode = str(cfg.path_relation.get("decode", "viterbi"))
    if decode_mode != "viterbi":
        raise ValueError(
            f"Unsupported path_relation.decode `{decode_mode}`; only `viterbi` is implemented."
        )
    model.eval()
    return model.explain_forward_paths(
        graph,
        sample,
        entities_weight=entities_weight,
        target_topk=int(cfg.path_relation.get("target_topk", 20)),
        target_selection_mode=str(
            cfg.path_relation.get("target_selection_mode", "score")
        ),
        target_score_topk=(
            int(cfg.path_relation.target_score_topk)
            if cfg.path_relation.get("target_score_topk", None) is not None
            else None
        ),
        target_mass_topk=(
            int(cfg.path_relation.target_mass_topk)
            if cfg.path_relation.get("target_mass_topk", None) is not None
            else None
        ),
        paths_per_target=int(cfg.path_relation.get("paths_per_target", 2)),
        flow_temperature=float(cfg.path_relation.get("tau_flow", 1.0)),
        flow_eps=float(cfg.path_relation.get("flow_eps", 1.0e-12)),
    )


def _resolve_max_images_score_file(cfg: DictConfig) -> str | None:
    configured = str(cfg.final_generation.get("max_images_score_file") or "").strip()
    if configured:
        return configured
    candidate = os.path.join(
        str(cfg.dataset.root),
        str(cfg.dataset.data_name),
        "webqa_test_qwen3vl_scores.jsonl",
    )
    if os.path.exists(candidate):
        return candidate
    return None


def _load_dynamic_max_images(
    score_file: str | None, default_max_images: int
) -> tuple[dict[str, int], dict[str, int]]:
    by_qid: dict[str, int] = {}
    by_query: dict[str, int] = {}
    if not score_file:
        return by_qid, by_query
    if not os.path.exists(score_file):
        logger.warning("max_images score file not found: %s", score_file)
        return by_qid, by_query

    total = 0
    with open(score_file, encoding="utf-8") as fin:
        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            mapped = _to_nonnegative_int(payload.get("score"), default_max_images)
            qid = str(payload.get("qid", "")).strip()
            if qid:
                by_qid[qid] = mapped
            query_key = _normalize_text(payload.get("query", ""))
            if query_key:
                by_query[query_key] = mapped
    logger.info(
        "Loaded dynamic max_images mapping from %s (rows=%s, qids=%s, queries=%s).",
        score_file,
        total,
        len(by_qid),
        len(by_query),
    )
    return by_qid, by_query


def _resolve_sample_max_images(
    sample: dict,
    default_max_images: int,
    by_qid: dict[str, int],
    by_query: dict[str, int],
) -> int:
    sample_id = str(sample.get("id", "")).strip()
    if sample_id and sample_id in by_qid:
        return by_qid[sample_id]
    query_key = _normalize_text(sample.get("question", ""))
    if query_key and query_key in by_query:
        return by_query[query_key]
    return default_max_images


def _merge_doc_micro_scores(
    graph_scores: dict[str, float] | None,
    node_scores: dict[str, float] | None,
    node_weight: float,
) -> dict[str, float]:
    merged: dict[str, float] = {}
    for doc_id, score in (graph_scores or {}).items():
        merged[str(doc_id)] = max(merged.get(str(doc_id), 0.0), float(score))
    for doc_id, score in (node_scores or {}).items():
        weighted_score = float(score) * float(node_weight)
        merged[str(doc_id)] = max(merged.get(str(doc_id), 0.0), weighted_score)
    return merged


def _align_qa_inputs(
    test_data: list[dict], retrieval_result: list[dict]
) -> tuple[list[tuple[dict, dict]], list[str]]:
    sample_by_raw_id = {str(sample.get("id")): sample for sample in test_data}
    qa_inputs: list[tuple[dict, dict]] = []
    unmatched_retrieval_ids: list[str] = []
    for item in retrieval_result:
        rid = item.get("id")
        sample = None
        try:
            rid_int = int(str(rid))
            if 0 <= rid_int < len(test_data):
                sample = test_data[rid_int]
        except (TypeError, ValueError):
            sample = None

        if sample is None:
            sample = sample_by_raw_id.get(str(rid))

        if sample is None:
            unmatched_retrieval_ids.append(str(rid))
            continue
        qa_inputs.append((sample, item))
    return qa_inputs, unmatched_retrieval_ids


def _export_macro_ranking(
    output_dir: str,
    qa_data: Dataset,
    retrieval_result: list[dict],
    top_k: int,
    output_path: str | None = None,
) -> str:
    test_data = qa_data.raw_test_data
    qa_inputs, unmatched_retrieval_ids = _align_qa_inputs(test_data, retrieval_result)
    if len(qa_inputs) != len(test_data) or len(retrieval_result) != len(test_data):
        logger.warning(
            "Sample/retrieval mismatch in macro export: test=%s, retrieval=%s, aligned=%s, unmatched=%s.",
            len(test_data),
            len(retrieval_result),
            len(qa_inputs),
            len(unmatched_retrieval_ids),
        )

    n_docs = len(qa_data.id2doc)
    top_k = max(1, min(int(top_k), max(1, n_docs)))
    destination = output_path or os.path.join(
        output_dir, "scienceqa_macro_scores.jsonl"
    )
    with open(destination, "w", encoding="utf-8") as fout:
        for sample, retrieval_doc in qa_inputs:
            text_doc_pred = (
                retrieval_doc.get("text_doc_pred", retrieval_doc["doc_pred"])
                .detach()
                .cpu()
            )
            values, indices = torch.topk(text_doc_pred, k=top_k, dim=-1)
            macro_ranking: list[dict[str, Any]] = []
            for idx, score in zip(indices.tolist(), values.tolist()):
                doc_id = str(qa_data.id2doc[int(idx)])
                macro_ranking.append(
                    {
                        "doc_id": doc_id,
                        "score": float(score),
                    }
                )
            top1_macro_doc = {}
            if macro_ranking:
                top1_doc_id = str(macro_ranking[0]["doc_id"])
                top1_macro_doc = {
                    "doc_id": top1_doc_id,
                    "score": float(macro_ranking[0]["score"]),
                    "content": str(qa_data.doc.get(top1_doc_id, "")),
                }
            row = {
                "id": sample.get("id"),
                "question": sample.get("question", ""),
                "choices": sample.get("choices", []),
                "answer": sample.get("answer"),
                "answer_letter": sample.get("answer_letter", ""),
                "question_images": sample.get("question_images", []),
                "question_image_captions": sample.get("question_image_captions", []),
                "macro_ranking": macro_ranking,
                "top1_macro_doc": top1_macro_doc,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Exported ScienceQA macro ranking to %s", destination)
    return destination


class QwenVLTextOnlyAdapter(BaseLanguageModel):
    """Adapter that exposes Qwen-VL client with BaseLanguageModel interface."""

    def __init__(self, qwen_vl_client: Any, max_new_tokens: int = 512):
        self.qwen_vl_client = qwen_vl_client
        self.max_new_tokens = int(max_new_tokens)

    def token_len(self, text: str) -> int:
        return len(text.split())

    def generate_sentence(
        self, llm_input: str | list, system_input: str = ""
    ) -> str | Exception:
        if isinstance(llm_input, list):
            chunks: list[str] = []
            for msg in llm_input:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "user"))
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_content = " ".join(
                        str(item.get("text", ""))
                        for item in content
                        if isinstance(item, dict)
                    )
                else:
                    text_content = str(content)
                chunks.append(f"[{role}] {text_content}")
            prompt = "\n".join(chunks).strip()
        else:
            prompt = str(llm_input).strip()

        if system_input:
            prompt = f"[system] {system_input}\n[user] {prompt}".strip()

        try:
            return self.qwen_vl_client.chat(
                prompt=prompt,
                images=None,
                sampling_overrides={"max_tokens": self.max_new_tokens},
            )
        except Exception as e:
            return e


def _build_text_llm(cfg: DictConfig) -> BaseLanguageModel:
    try:
        return instantiate(cfg.llm)
    except Exception as exc:
        logger.warning(
            "Instantiate cfg.llm failed, fallback to local Qwen-VL text adapter: %s",
            exc,
        )
        qwen_vl_client = get_or_create_qwen_vl_client(
            model_name_or_path=cfg.final_generation.qwen_vl.model_name_or_path,
            tensor_parallel_size=cfg.final_generation.qwen_vl.tensor_parallel_size,
            gpu_memory_utilization=cfg.final_generation.qwen_vl.gpu_memory_utilization,
            max_num_batched_tokens=cfg.final_generation.qwen_vl.max_num_batched_tokens,
            max_model_len=cfg.final_generation.qwen_vl.max_model_len,
            allowed_local_media_path=cfg.final_generation.qwen_vl.allowed_local_media_path,
        )
        return QwenVLTextOnlyAdapter(
            qwen_vl_client=qwen_vl_client,
            max_new_tokens=cfg.final_generation.qwen_vl.max_tokens,
        )


@torch.no_grad()
def doc_retrieval(
    cfg: DictConfig,
    model: nn.Module,
    qa_data: Dataset,
    device: torch.device,
    pattern_parser: SimGRAGQueryParser | None = None,
    micro_macro_activator: MicroMacroActivator | None = None,
) -> list[dict]:
    retrieval_wall_start = time.perf_counter()
    world_size = utils.get_world_size()
    rank = utils.get_rank()

    _, test_data = qa_data._data
    graph = qa_data.kg
    ent2docs_text = qa_data.ent2docs_text
    ent2docs_image = qa_data.ent2docs_image

    sampler = torch_data.DistributedSampler(test_data, world_size, rank, shuffle=False)
    test_loader = torch_data.DataLoader(
        test_data, cfg.test.retrieval_batch_size, sampler=sampler
    )

    doc_ranker_text = instantiate(cfg.doc_ranker, ent2doc=ent2docs_text)
    doc_ranker_image = instantiate(cfg.doc_ranker, ent2doc=ent2docs_image)

    if cfg.test.init_entities_weight:
        entities_weight = utils.get_entities_weight(ent2docs_text)
    else:
        entities_weight = None

    model.eval()
    all_predictions: list[dict] = []
    pattern_stats = {
        "total_samples": 0,
        "parse_fail": 0,
        "parse_success": 0,
        "condition_nonempty": 0,
        "condition_empty": 0,
        "activation_fail": 0,
        "extra_seed_nonempty": 0,
        "extra_seed_empty": 0,
        "seed_total": 0,
        "seed_in_graph_total": 0,
        "seed_oov_total": 0,
        "mask_expanded": 0,
        "mask_unchanged": 0,
        "mask_delta_total": 0,
        "mask_delta_max": 0,
    }
    debug_first_n = int(cfg.pattern_activation.get("debug_first_n", 0))
    debug_include_query = bool(cfg.pattern_activation.get("debug_include_query", False))
    image_fusion_cfg = cfg.get("image_fusion", None)
    use_micro_score = bool(
        image_fusion_cfg is not None
        and image_fusion_cfg.enable
        and image_fusion_cfg.get("enable_micro_score", True)
    )
    node_doc_score_enable = bool(
        use_micro_score and cfg.pattern_activation.get("node_doc_score_enable", False)
    )
    node_doc_weight = float(cfg.pattern_activation.get("node_doc_weight", 1.0))
    node_anchor_source = str(
        cfg.pattern_activation.get("node_anchor_source", "question_entities")
    ).strip()
    path_relation_cfg = cfg.get("path_relation", None)
    path_decode_enable = bool(
        path_relation_cfg is not None and path_relation_cfg.get("enable", False)
    )
    id2ent: dict[int, str] = {}
    id2rel: dict[int, str] = {}
    if path_decode_enable:
        id2ent = {v: k for k, v in qa_data.ent2id.items()}
        id2rel = {v: k for k, v in qa_data.rel2id.items()}
    sample_debug_logs: list[str] = []
    timing_stats = {
        "num_batches": 0,
        "num_samples": 0,
        "pattern_total_s": 0.0,
        "pattern_llm_parse_s": 0.0,
        "to_device_s": 0.0,
        "model_forward_s": 0.0,
        "doc_ranker_s": 0.0,
        "cpu_transfer_s": 0.0,
        "pack_output_s": 0.0,
        "batch_total_s": 0.0,
    }
    for batch in tqdm(test_loader):
        batch_wall_start = time.perf_counter()
        batch_size = int(batch["sample_id"].shape[0])
        timing_stats["num_batches"] += 1
        timing_stats["num_samples"] += batch_size
        batch_doc_micro_scores: list[dict[str, float]] = [{} for _ in range(batch_size)]
        batch_condition_triples: list[list[tuple[str, str, str]]] = [
            [] for _ in range(batch_size)
        ]
        pattern_stage_start = time.perf_counter()
        if (
            cfg.pattern_activation.enable
            and pattern_parser is not None
            and micro_macro_activator is not None
        ):
            sample_ids = batch["sample_id"].tolist()
            enhanced_masks = []
            for idx, sample_id in enumerate(sample_ids):
                pattern_stats["total_samples"] += 1
                merged_mask = batch["question_entities_masks"][idx].clone()
                before_nnz = int(merged_mask.sum().item())
                query = qa_data.raw_test_data[sample_id]["question"]
                question_entities = [
                    str(x).strip()
                    for x in qa_data.raw_test_data[sample_id].get(
                        "question_entities", []
                    )
                    if str(x).strip()
                ]
                condition_triples = []
                extra_seed_entities: list[str] = []
                activation_trace: dict[str, Any] = {
                    "activated_entities": [],
                    "matched_hits": [],
                }
                try:
                    parse_llm_start = time.perf_counter()
                    pattern_payload = pattern_parser.parse(query)
                    timing_stats["pattern_llm_parse_s"] += (
                        time.perf_counter() - parse_llm_start
                    )
                    pattern_stats["parse_success"] += 1
                    condition_triples = pattern_payload.get("condition_triples", [])
                    if condition_triples:
                        pattern_stats["condition_nonempty"] += 1
                    else:
                        pattern_stats["condition_empty"] += 1
                    activation_trace = micro_macro_activator.activate_with_trace(
                        condition_triples
                    )
                    extra_seed_entities = activation_trace["activated_entities"]
                except Exception as exc:
                    logger.warning(
                        "Pattern activation failed in single-round retrieval: %s", exc
                    )
                    pattern_stats["parse_fail"] += 1
                    pattern_stats["activation_fail"] += 1
                    condition_triples = []
                    extra_seed_entities = []
                    activation_trace = {
                        "activated_entities": [],
                        "matched_hits": [],
                        "doc_micro_scores": {},
                    }

                graph_doc_micro_scores = (
                    activation_trace.get("doc_micro_scores", {})
                    if use_micro_score
                    else {}
                )
                node_doc_micro_scores: dict[str, float] = {}
                if node_doc_score_enable and (condition_triples or question_entities):
                    try:
                        anchor_texts = (
                            question_entities
                            if node_anchor_source == "question_entities"
                            else None
                        )
                        node_doc_micro_scores = (
                            micro_macro_activator.score_docs_with_node_anchors(
                                condition_triples,
                                anchor_texts=anchor_texts,
                            )
                        )
                    except Exception as exc:
                        logger.warning(
                            "Node-level doc scoring failed in single-round retrieval: %s",
                            exc,
                        )
                        node_doc_micro_scores = {}
                merged_doc_micro_scores = _merge_doc_micro_scores(
                    graph_scores=graph_doc_micro_scores,
                    node_scores=node_doc_micro_scores,
                    node_weight=node_doc_weight,
                )

                if extra_seed_entities:
                    pattern_stats["extra_seed_nonempty"] += 1
                else:
                    pattern_stats["extra_seed_empty"] += 1
                pattern_stats["seed_total"] += len(extra_seed_entities)

                mapped_entities = [
                    entity for entity in extra_seed_entities if entity in qa_data.ent2id
                ]
                oov_entities = [
                    entity
                    for entity in extra_seed_entities
                    if entity not in qa_data.ent2id
                ]
                pattern_stats["seed_in_graph_total"] += len(mapped_entities)
                pattern_stats["seed_oov_total"] += len(oov_entities)

                if extra_seed_entities:
                    entity_ids = [qa_data.ent2id[entity] for entity in mapped_entities]
                    if entity_ids:
                        extra_mask = entities_to_mask(entity_ids, qa_data.kg.num_nodes)
                        merged_mask = torch.maximum(merged_mask, extra_mask)
                after_nnz = int(merged_mask.sum().item())
                delta_nnz = max(0, after_nnz - before_nnz)
                pattern_stats["mask_delta_total"] += delta_nnz
                pattern_stats["mask_delta_max"] = max(
                    pattern_stats["mask_delta_max"], delta_nnz
                )
                if delta_nnz > 0:
                    pattern_stats["mask_expanded"] += 1
                else:
                    pattern_stats["mask_unchanged"] += 1

                if len(sample_debug_logs) < debug_first_n:
                    reason = []
                    if not condition_triples:
                        reason.append("no_condition_triples")
                    if extra_seed_entities and not mapped_entities:
                        reason.append("all_extra_seeds_oov")
                    if mapped_entities and delta_nnz == 0:
                        reason.append("extra_seeds_already_activated")
                    if not extra_seed_entities:
                        reason.append("no_extra_seeds")
                    reason_text = ",".join(reason) if reason else "expanded"
                    q_text = f" | query={query}" if debug_include_query else ""
                    trace_lines: list[str] = []
                    for hit_idx, hit in enumerate(
                        activation_trace.get("matched_hits", [])[:3], start=1
                    ):
                        triples = hit.get("matched_triples", [])[:5]
                        for triple in triples:
                            trace_lines.append(
                                "hit{}(score={:.4f}): triple_id={} doc_id={} modality={} source_ref={} triple=({},{},{})".format(
                                    hit_idx,
                                    float(hit.get("score", 0.0)),
                                    triple.get("triple_id", ""),
                                    triple.get("doc_id", ""),
                                    triple.get("modality", ""),
                                    triple.get("source_ref", ""),
                                    triple.get("head", ""),
                                    triple.get("relation", ""),
                                    triple.get("tail", ""),
                                )
                            )
                    trace_text = (
                        " | matched_subgraphs=[" + " ; ".join(trace_lines) + "]"
                        if trace_lines
                        else ""
                    )
                    sample_debug_logs.append(
                        f"sample_id={sample_id} cond={len(condition_triples)} extra_seed={len(extra_seed_entities)} mapped={len(mapped_entities)} oov={len(oov_entities)} mask_before={before_nnz} mask_after={after_nnz} delta={delta_nnz} reason={reason_text}{q_text}{trace_text}"
                    )
                batch_doc_micro_scores[idx] = merged_doc_micro_scores
                batch_condition_triples[idx] = list(condition_triples)
                enhanced_masks.append(merged_mask)
            batch["question_entities_masks"] = torch.stack(enhanced_masks, dim=0)
        timing_stats["pattern_total_s"] += time.perf_counter() - pattern_stage_start

        _maybe_cuda_sync(device)
        to_device_start = time.perf_counter()
        batch = query_utils.cuda(batch, device=device)
        _maybe_cuda_sync(device)
        timing_stats["to_device_s"] += time.perf_counter() - to_device_start

        _maybe_cuda_sync(device)
        model_forward_start = time.perf_counter()
        ent_pred = model(graph, batch, entities_weight=entities_weight)
        _maybe_cuda_sync(device)
        timing_stats["model_forward_s"] += time.perf_counter() - model_forward_start

        _maybe_cuda_sync(device)
        doc_ranker_start = time.perf_counter()
        text_doc_pred = doc_ranker_text(ent_pred)
        image_doc_pred = doc_ranker_image(ent_pred)
        _maybe_cuda_sync(device)
        timing_stats["doc_ranker_s"] += time.perf_counter() - doc_ranker_start

        _maybe_cuda_sync(device)
        cpu_transfer_start = time.perf_counter()
        sample_id_batch = batch["sample_id"]
        ent_pred_cpu = ent_pred.cpu()
        text_doc_pred_cpu = text_doc_pred.cpu()
        image_doc_pred_cpu = image_doc_pred.cpu()
        _maybe_cuda_sync(device)
        timing_stats["cpu_transfer_s"] += time.perf_counter() - cpu_transfer_start

        pack_output_start = time.perf_counter()
        for row_idx, sample_id in enumerate(sample_id_batch.cpu().tolist()):
            path_targets: list[dict[str, Any]] = []
            if path_decode_enable:
                single_sample = _slice_batch_sample(batch, row_idx, batch_size)
                try:
                    paths_results = _decode_sample_paths(
                        cfg, single_sample, model, graph, entities_weight
                    )
                    path_targets = _serialize_path_targets(
                        paths_results, id2ent, id2rel
                    )
                except ValueError as exc:
                    if "source_weights must contain at least one positive entry" in str(
                        exc
                    ):
                        logger.warning(
                            "Skip path decode for sample %s: empty source_weights after entity mapping/weighting.",
                            sample_id,
                        )
                    else:
                        raise
            all_predictions.append(
                {
                    "id": sample_id,
                    "ent_pred": ent_pred_cpu[row_idx],
                    "text_doc_pred": text_doc_pred_cpu[row_idx],
                    "image_doc_pred": image_doc_pred_cpu[row_idx],
                    # Backward-compatible alias for old downstream consumers.
                    "doc_pred": text_doc_pred_cpu[row_idx],
                    "doc_micro_scores": batch_doc_micro_scores[row_idx],
                    "condition_triples": batch_condition_triples[row_idx],
                    "path_targets": path_targets,
                }
            )
        timing_stats["pack_output_s"] += time.perf_counter() - pack_output_start
        timing_stats["batch_total_s"] += time.perf_counter() - batch_wall_start

    if utils.get_world_size() > 1:
        gathered_predictions = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered_predictions, all_predictions)
    else:
        gathered_predictions = [all_predictions]  # type: ignore

    sorted_predictions = sorted(
        [item for sublist in gathered_predictions for item in sublist],  # type: ignore
        key=lambda x: x["id"],
    )
    if cfg.pattern_activation.enable and utils.is_main_process():
        logger.info(
            "Pattern activation stats (single-round): total=%s, parse_success=%s, parse_fail=%s, condition_nonempty=%s, condition_empty=%s, extra_seed_nonempty=%s, extra_seed_empty=%s, seed_total=%s, seed_in_graph_total=%s, seed_oov_total=%s, mask_expanded=%s, mask_unchanged=%s, mask_delta_total=%s, mask_delta_max=%s",
            pattern_stats["total_samples"],
            pattern_stats["parse_success"],
            pattern_stats["parse_fail"],
            pattern_stats["condition_nonempty"],
            pattern_stats["condition_empty"],
            pattern_stats["extra_seed_nonempty"],
            pattern_stats["extra_seed_empty"],
            pattern_stats["seed_total"],
            pattern_stats["seed_in_graph_total"],
            pattern_stats["seed_oov_total"],
            pattern_stats["mask_expanded"],
            pattern_stats["mask_unchanged"],
            pattern_stats["mask_delta_total"],
            pattern_stats["mask_delta_max"],
        )
        for line in sample_debug_logs:
            logger.info("Pattern activation detail: %s", line)
    retrieval_wall_total = time.perf_counter() - retrieval_wall_start
    if utils.is_main_process():
        num_batches = max(1, int(timing_stats["num_batches"]))
        num_samples = max(1, int(timing_stats["num_samples"]))
        pattern_total_s = float(timing_stats["pattern_total_s"])
        pattern_llm_parse_s = float(timing_stats["pattern_llm_parse_s"])
        pattern_non_llm_s = max(0.0, pattern_total_s - pattern_llm_parse_s)
        effective_retrieval_s = max(0.0, retrieval_wall_total - pattern_llm_parse_s)
        logger.info(
            "Retrieval latency stats: wall_total=%.3fs, effective_no_pattern_llm=%.3fs, batches=%s, samples=%s, "
            "batch_avg=%.3fs, sample_avg=%.6fs, pattern_total=%.3fs, pattern_llm_parse=%.3fs, pattern_non_llm=%.3fs, "
            "to_device=%.3fs, model_forward=%.3fs, doc_ranker=%.3fs, cpu_transfer=%.3fs, pack_output=%.3fs",
            retrieval_wall_total,
            effective_retrieval_s,
            int(timing_stats["num_batches"]),
            int(timing_stats["num_samples"]),
            float(timing_stats["batch_total_s"]) / num_batches,
            effective_retrieval_s / num_samples,
            pattern_total_s,
            pattern_llm_parse_s,
            pattern_non_llm_s,
            float(timing_stats["to_device_s"]),
            float(timing_stats["model_forward_s"]),
            float(timing_stats["doc_ranker_s"]),
            float(timing_stats["cpu_transfer_s"]),
            float(timing_stats["pack_output_s"]),
        )
    utils.synchronize()
    return sorted_predictions


def ans_prediction(
    cfg: DictConfig,
    output_dir: str,
    qa_data: Dataset,
    retrieval_result: list[dict],
    retrieval_only: bool = False,
) -> str:
    llm = None
    if not retrieval_only:
        llm = _build_text_llm(cfg)
    doc_retriever = utils.DualBranchDocumentRetriever(
        qa_data.doc,
        qa_data.id2doc,
        image_doc_ids=set(getattr(qa_data, "image_doc_ids", set())),
    )
    test_data = qa_data.raw_test_data
    id2ent = {v: k for k, v in qa_data.ent2id.items()}
    default_max_images = _to_nonnegative_int(cfg.final_generation.max_images, 0)
    score_file = _resolve_max_images_score_file(cfg)
    if score_file is None:
        # Fall back to the inline visual-budget classifier's cache when enabled.
        score_file = run_visual_budget_if_enabled(
            cfg.final_generation, output_dir, test_data, str(cfg.dataset.data_name)
        )
    max_images_by_qid, max_images_by_query = _load_dynamic_max_images(
        score_file, default_max_images
    )

    prompt_builder = QAPromptBuilder(cfg.qa_prompt)
    qa_inputs, unmatched_retrieval_ids = _align_qa_inputs(test_data, retrieval_result)

    if len(qa_inputs) != len(test_data) or len(retrieval_result) != len(test_data):
        logger.warning(
            "Sample/retrieval mismatch: test=%s, retrieval=%s, aligned=%s, unmatched_retrieval=%s.",
            len(test_data),
            len(retrieval_result),
            len(qa_inputs),
            len(unmatched_retrieval_ids),
        )
        if unmatched_retrieval_ids:
            logger.warning(
                "Unmatched retrieval ids (first 20): %s", unmatched_retrieval_ids[:20]
            )
    image_fusion_cfg = cfg.get("image_fusion", None)
    micro_score_matcher: SimGRAGMicroMatcher | None = None
    image_fusion_enable = bool(image_fusion_cfg is not None and image_fusion_cfg.enable)
    use_micro_score = bool(
        image_fusion_enable and image_fusion_cfg.get("enable_micro_score", True)
    )
    if use_micro_score:
        micro_score_matcher = instantiate(cfg.micro_matcher)

    def predict(qa_input: tuple[dict, torch.Tensor]) -> dict | Exception:
        data, retrieval_doc = qa_input
        sample_max_images = _resolve_sample_max_images(
            data, default_max_images, max_images_by_qid, max_images_by_query
        )
        retrieved_ent_idx = torch.topk(
            retrieval_doc["ent_pred"], cfg.test.save_top_k_entity, dim=-1
        ).indices
        retrieved_ent = [id2ent[i.item()] for i in retrieved_ent_idx]
        text_doc_pred = retrieval_doc.get("text_doc_pred", retrieval_doc["doc_pred"])
        image_doc_pred = retrieval_doc.get(
            "image_doc_pred", torch.zeros_like(text_doc_pred)
        )
        doc_micro_scores = (
            retrieval_doc.get("doc_micro_scores", {}) if use_micro_score else {}
        )
        retrieved_docs = doc_retriever(
            text_doc_pred,
            image_doc_pred,
            top_k=cfg.test.top_k,
            max_image_docs=sample_max_images,
            doc_micro_scores=doc_micro_scores,
            apply_image_fusion=image_fusion_enable,
            non_candidate_scale=float(
                image_fusion_cfg.get("non_candidate_scale", 1.0 / 3.0)
            )
            if image_fusion_cfg is not None
            else (1.0 / 3.0),
            condition_triples=retrieval_doc.get("condition_triples", []),
            micro_matcher=micro_score_matcher,
            stage1_dir=os.path.join(
                cfg.dataset.root, cfg.dataset.data_name, "processed", "stage1"
            ),
            enable_path_relation=bool(cfg.path_relation.enable),
            explain_record=(
                {"targets": retrieval_doc.get("path_targets", [])}
                if cfg.path_relation.enable
                else None
            ),
            path_relation_max_neighbors=int(cfg.path_relation.max_neighbors_per_doc),
            path_relation_max_evidence=int(cfg.path_relation.max_evidence_per_neighbor),
            path_relation_score_field=str(cfg.path_relation.score_field),
            path_relation_bridge_max_window=int(cfg.path_relation.bridge_max_window),
            path_relation_bridge_decay=float(cfg.path_relation.bridge_decay),
        )
        final_context_images: list[str] = []
        final_context_media_metadata: list[dict] = []
        if retrieval_only:
            response: str | Exception = ""
        else:
            message = prompt_builder.build_input_prompt(
                data["question"], retrieved_docs
            )
            if llm is None:
                raise RuntimeError(
                    "Text-only generation requires cfg.llm, but no llm is initialized."
                )
            response = llm.generate_sentence(message)
        if isinstance(response, Exception):
            return response
        else:
            return {
                "id": data["id"],
                "question": data["question"],
                "answer": data["answer"],
                "answer_aliases": data.get(
                    "answer_aliases", []
                ),  # Some datasets have answer aliases
                "supporting_facts": data.get("supporting_facts", []),
                "response": response,
                "retrieved_ent": retrieved_ent,
                "retrieved_docs": retrieved_docs,
                "final_context_images": final_context_images,
                "final_context_media_metadata": final_context_media_metadata,
            }

    with open(os.path.join(output_dir, "prediction.jsonl"), "w") as f:
        if retrieval_only:
            for qa_input in tqdm(qa_inputs, total=len(qa_inputs)):
                results = predict(qa_input)
                if isinstance(results, Exception):
                    logger.error(f"Error: {results}")
                    continue
                f.write(json.dumps(results) + "\n")
                f.flush()
        else:
            with ThreadPool(cfg.test.n_threads) as pool:
                for results in tqdm(
                    pool.imap(predict, qa_inputs),
                    total=len(qa_inputs),
                ):
                    if isinstance(results, Exception):
                        logger.error(f"Error: {results}")
                        continue

                    f.write(json.dumps(results) + "\n")
                    f.flush()

    return os.path.join(output_dir, "prediction.jsonl")


@hydra.main(config_path="config", config_name="stage3_qa_inference", version_base=None)
def main(cfg: DictConfig) -> dict[str, Any] | None:
    output_dir = HydraConfig.get().runtime.output_dir
    scienceqa_export_macro_only = bool(
        cfg.test.get("scienceqa_export_macro_only", False)
    )
    utils.init_distributed_mode()
    torch.manual_seed(cfg.seed + utils.get_rank())
    if utils.get_rank() == 0:
        logger.info(f"Config:\n {OmegaConf.to_yaml(cfg)}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Output directory: {output_dir}")

    model, model_config = utils.load_model_from_pretrained(
        cfg.graph_retriever.model_path
    )
    if "text_emb_model" in cfg and cfg.text_emb_model is not None:
        text_emb_model_cfgs = OmegaConf.create(
            OmegaConf.to_container(cfg.text_emb_model, resolve=True)
        )
    else:
        text_emb_model_cfgs = OmegaConf.create(model_config["text_emb_model_config"])
    qa_data = QADataset(
        **cfg.dataset,
        text_emb_model_cfgs=text_emb_model_cfgs,
    )
    pattern_parser: SimGRAGQueryParser | None = None
    micro_macro_activator: MicroMacroActivator | None = None
    if (not scienceqa_export_macro_only) and cfg.pattern_activation.enable:
        llm_for_pattern = _build_text_llm(cfg)
        stage1_dir = os.path.join(
            cfg.dataset.root, cfg.dataset.data_name, "processed", "stage1"
        )
        cross_tier_path = cfg.pattern_activation.cross_tier_path or os.path.join(
            stage1_dir, "cross_tier_index.json"
        )
        micro_kg_path = cfg.pattern_activation.micro_kg_path or os.path.join(
            stage1_dir, "micro_kg.jsonl"
        )
        micro_matcher: SimGRAGMicroMatcher = instantiate(cfg.micro_matcher)
        pattern_parser = SimGRAGQueryParser(llm_for_pattern)
        micro_macro_activator = MicroMacroActivator(
            micro_matcher=micro_matcher,
            cross_tier_index_path=cross_tier_path,
            micro_kg_path=micro_kg_path,
            micro_match_topk=cfg.pattern_activation.micro_match_topk,
            max_match_score=cfg.pattern_activation.max_match_score,
            min_match_score=cfg.pattern_activation.min_match_score,
            node_doc_match_topk=cfg.pattern_activation.get("node_doc_match_topk", 30),
            node_doc_max_distance=cfg.pattern_activation.get(
                "node_doc_max_distance", 1.0
            ),
            node_doc_anchor_min_tokens=cfg.pattern_activation.get(
                "node_doc_anchor_min_tokens", 1
            ),
            node_doc_anchor_min_chars=cfg.pattern_activation.get(
                "node_doc_anchor_min_chars", 4
            ),
            node_doc_include_tail_nodes=cfg.pattern_activation.get(
                "node_doc_include_tail_nodes", False
            ),
        )
    device = utils.get_device()
    model = model.to(device)

    qa_data.kg = qa_data.kg.to(device)
    qa_data.ent2docs_text = qa_data.ent2docs_text.to(device)
    qa_data.ent2docs_image = qa_data.ent2docs_image.to(device)
    qa_data.ent2docs = qa_data.ent2docs_text

    if cfg.test.retrieved_result_path:
        retrieval_result = torch.load(cfg.test.retrieved_result_path, weights_only=True)
    else:
        if cfg.test.prediction_result_path:
            retrieval_result = None
        else:
            retrieval_result = doc_retrieval(
                cfg,
                model,
                qa_data,
                device=device,
                pattern_parser=pattern_parser,
                micro_macro_activator=micro_macro_activator,
            )
    if utils.is_main_process():
        retrieval_only = bool(cfg.test.get("retrieval_only", False))
        if cfg.test.save_retrieval and retrieval_result is not None:
            logger.info(
                f"Ranking saved to disk: {os.path.join(output_dir, 'retrieval_result.pt')}"
            )
            torch.save(
                retrieval_result, os.path.join(output_dir, "retrieval_result.pt")
            )
        if scienceqa_export_macro_only:
            if retrieval_result is None:
                raise RuntimeError(
                    "ScienceQA macro export requires retrieval_result, but none was generated or loaded."
                )
            output_path = _export_macro_ranking(
                output_dir=output_dir,
                qa_data=qa_data,
                retrieval_result=retrieval_result,
                top_k=int(cfg.test.get("scienceqa_macro_top_k", 200)),
                output_path=cfg.test.prediction_result_path,
            )
            logger.info("ScienceQA macro export finished: %s", output_path)
            return {"macro_output": output_path}
        elif cfg.test.prediction_result_path:
            output_path = cfg.test.prediction_result_path
        else:
            output_path = ans_prediction(
                cfg,
                output_dir,
                qa_data,
                retrieval_result,
                retrieval_only=retrieval_only,
            )

        if bool(cfg.final_eval.get("enable", False)):
            output_path = run_final_eval(cfg, output_path, output_dir)
            evaluator = instantiate(cfg.qa_evaluator, prediction_file=output_path)
        elif retrieval_only:
            evaluator = RetrievalEvaluator(prediction_file=output_path)
        else:
            evaluator = instantiate(cfg.qa_evaluator, prediction_file=output_path)
        metrics = evaluator.evaluate()
        query_utils.print_metrics(metrics, logger)
        return metrics

    return None


if __name__ == "__main__":
    main()
