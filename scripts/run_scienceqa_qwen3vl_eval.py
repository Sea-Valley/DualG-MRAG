"""ScienceQA post-processing pipeline: micro scoring, score merging, VLM
answering and direct-vs-RAG judge arbitration, all behind one entry point.

Subcommands:
    evaluate      Run the full direct-vs-RAG chain (micro-score -> merge ->
                  answer RAG + direct -> judge) with one shared vLLM client;
                  intermediates are written to ``--output-dir``.
    micro-score   Score question anchors against the micro matching graph
                  (SimGRAG pattern match + node-level anchors) and write
                  ``scienceqa_micro_scores.jsonl``.
    merge         Additively merge macro and micro rankings (micro only for
                  questions with images) into ``final_ranking`` /
                  ``top1_doc_id`` / ``top1_doc_content``.
    answer        Answer questions with a local Qwen3-VL model over the merged
                  top-1 document (``--zeroshot`` produces the no-retrieval
                  direct run used by ``judge``).
    judge         Arbitrate samples where the direct and RAG runs disagree in
                  correctness with a Qwen3-VL judge, then report bucketed
                  accuracies.

Typical flow (after stage3 ``test.scienceqa_export_macro_only=true``):

    python scripts/run_scienceqa_qwen3vl_eval.py evaluate \
        --macro-jsonl <stage3_export.jsonl> --output-dir outputs/scienceqa_eval

The granular subcommands remain available for debugging individual steps.
Invoking the script without a subcommand defaults to ``answer`` for backward
compatibility.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import threading
import time
from collections.abc import Sequence
from math import sqrt
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from gfmrag.kg_construction.micro_matcher import SimGRAGMicroMatcher
from gfmrag.kg_construction.utils import processing_phrases
from gfmrag.llms.qwenvl_vllm import get_or_create_qwen_vl_client
from gfmrag.query_pattern import SimGRAGQueryParser
from gfmrag.workflow.scienceqa_utils import (
    answer_index_to_letter,
    dump_jsonl,
    load_json,
    parse_answer_letter,
    render_choices,
    render_question_with_choices,
)

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_MICRO_MATCHER_CONFIG = Path(
    "gfmrag/workflow/config/micro_matcher/simgrag_matcher.yaml"
)
DEFAULT_ANSWER_PROMPT = Path("gfmrag/workflow/config/qa_prompt/scienceqa.yaml")

SUBCOMMANDS = ("micro-score", "merge", "answer", "judge", "evaluate")


# ---------------------------------------------------------------------------
# Shared IO / bucket helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.flush()
        os.fsync(fout.fileno())


def _load_problems(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"ScienceQA problems must be dict: {path}")
    return {str(pid): item for pid, item in payload.items() if isinstance(item, dict)}


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def _subject_bucket(subject: str) -> str:
    value = str(subject or "").strip().lower()
    if value == "natural science":
        return "NAT"
    if value == "social science":
        return "Soc"
    if value == "language science":
        return "LAN"
    return "OTHER"


def _modality_bucket(problem: dict[str, Any]) -> str:
    has_image = str(problem.get("image", "")).strip() not in {"", "null", "None"}
    has_text = bool(str(problem.get("hint", "")).strip())
    if has_image:
        return "IMG"
    if has_text:
        return "TXT"
    return "NO"


def _grade_bucket(grade: Any) -> str:
    value = str(grade or "").strip().lower().replace("grade", "")
    try:
        num = int(value)
    except ValueError:
        return "UNKNOWN"
    return "G1-6" if num <= 6 else "G7-12"


def _update_bucket(
    stats: dict[str, dict[str, int]], key: str, is_correct: bool
) -> None:
    bucket = stats.setdefault(key, {"total": 0, "correct": 0})
    bucket["total"] += 1
    bucket["correct"] += int(is_correct)


def _summarize_bucket(
    stats: dict[str, dict[str, int]],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for key in sorted(stats.keys()):
        total = int(stats[key].get("total", 0))
        correct = int(stats[key].get("correct", 0))
        summary[key] = {
            "total": total,
            "correct": correct,
            "accuracy": (correct / total) if total else 0.0,
        }
    return summary


def _create_client(args: argparse.Namespace) -> Any:
    attempts = [
        (
            int(args.tensor_parallel_size),
            float(args.gpu_memory_utilization),
            int(args.max_num_batched_tokens),
            int(args.max_model_len),
        ),
        (
            1,
            min(float(args.gpu_memory_utilization), 0.9),
            min(int(args.max_num_batched_tokens), 8192),
            min(int(args.max_model_len), 8192),
        ),
        (1, 0.85, 4096, 4096),
    ]
    errors: list[str] = []
    dedup_attempts: list[tuple[int, float, int, int]] = []
    for item in attempts:
        if item not in dedup_attempts:
            dedup_attempts.append(item)
    for tp, gpu, max_bt, max_len in dedup_attempts:
        try:
            return get_or_create_qwen_vl_client(
                model_name_or_path=args.model_name_or_path,
                tensor_parallel_size=tp,
                gpu_memory_utilization=gpu,
                max_num_batched_tokens=max_bt,
                max_model_len=max_len,
                allowed_local_media_path=args.allowed_local_media_path or None,
            )
        except Exception as exc:
            errors.append(
                f"init_failed(tp={tp}, gpu={gpu}, max_bt={max_bt}, max_len={max_len}): {exc}"
            )
            continue
    raise RuntimeError(
        "Failed to initialize Qwen-VL client after retries. "
        "Please verify GPU resources/model path. Details:\n" + "\n".join(errors)
    )


def _add_vllm_args(parser: argparse.ArgumentParser, default_tp: int = 2) -> None:
    parser.add_argument("--model-name-or-path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--tensor-parallel-size", type=int, default=default_tp)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--allowed-local-media-path", type=str, default="")
    parser.add_argument("--max-tokens", type=int, default=256)


# ---------------------------------------------------------------------------
# micro-score
# ---------------------------------------------------------------------------


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    kept: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        kept.append(normalized)
    return kept


def _select_anchor_entities(
    sample: dict[str, Any], condition_triples: list[tuple[str, str, str]]
) -> list[str]:
    fixed_captions = _dedupe_keep_order(
        [str(item).strip() for item in sample.get("question_image_captions", [])]
    )
    extra_candidates: list[str] = []
    for entity in sample.get("question_entities", []):
        text = str(entity).strip()
        if text:
            extra_candidates.append(text)
    for head, _, tail in condition_triples:
        if "UNKNOWN" not in head:
            extra_candidates.append(str(head).strip())
        if "UNKNOWN" not in tail:
            extra_candidates.append(str(tail).strip())
    extra_entities = [
        entity
        for entity in _dedupe_keep_order(extra_candidates)
        if entity not in fixed_captions
    ][:2]
    return fixed_captions + extra_entities


def _sort_doc_scores(score_map: dict[str, float], top_k: int) -> list[dict[str, Any]]:
    ranked = sorted(
        score_map.items(),
        key=lambda item: (-float(item[1]), str(item[0])),
    )
    return [
        {"doc_id": str(doc_id), "score": float(score)}
        for doc_id, score in ranked[: max(1, int(top_k))]
    ]


def _safe_match_pattern(
    matcher: SimGRAGMicroMatcher,
    condition_triples: list[tuple[str, str, str]],
    topk: int,
) -> tuple[list[dict[str, Any]], str | None]:
    if not condition_triples:
        return [], "empty_condition_triples"
    try:
        return matcher.match_pattern(condition_triples, topk=topk, mode="greedy"), None
    except RuntimeError as exc:
        message = str(exc)
        if "No similar nodes found for root query node." in message:
            return [], "no_similar_root_node"
        return [], message


def _build_micro_metadata(
    micro_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, set[str]]]:
    triple_meta: dict[str, dict[str, Any]] = {}
    node_to_triple_ids: dict[str, set[str]] = {}
    for row in micro_rows:
        triple_id = str(row.get("triple_id", "")).strip()
        if not triple_id:
            continue
        triple_meta[triple_id] = {
            "triple_id": triple_id,
            "doc_id": str(row.get("doc_id", "")).strip(),
            "modality": str(row.get("modality", "")).strip(),
            "source_ref": str(row.get("source_ref", "")).strip(),
            "head": str(row.get("head", "")).strip(),
            "relation": str(row.get("relation", "")).strip(),
            "tail": str(row.get("tail", "")).strip(),
        }
        for raw_node in (row.get("head", ""), row.get("tail", "")):
            node = processing_phrases(raw_node)
            if not node:
                continue
            node_to_triple_ids.setdefault(node, set()).add(triple_id)
    return triple_meta, node_to_triple_ids


def _collect_graph_doc_scores(
    graph_hits: list[dict[str, Any]],
    triple_meta: dict[str, dict[str, Any]],
    max_match_score: float | None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    doc_scores: dict[str, float] = {}
    traced_hits: list[dict[str, Any]] = []
    for hit in graph_hits:
        score = float(hit.get("score", 0.0))
        if max_match_score is not None and score > max_match_score:
            continue
        matched_triple_ids = [str(item) for item in hit.get("matched_triple_ids", [])]
        matched_triples = []
        hit_micro_score = 1.0 / (1.0 + max(0.0, score))
        for triple_id in matched_triple_ids:
            triple_info = dict(triple_meta.get(triple_id, {"triple_id": triple_id}))
            matched_triples.append(triple_info)
            doc_id = str(triple_info.get("doc_id", "")).strip()
            if not doc_id:
                continue
            doc_scores[doc_id] = max(doc_scores.get(doc_id, 0.0), hit_micro_score)
        traced_hits.append(
            {
                "score": score,
                "doc_micro_score": hit_micro_score,
                "matched_triple_ids": matched_triple_ids,
                "matched_triples": matched_triples,
            }
        )
    return doc_scores, traced_hits


def _anchor_weight(anchor: str) -> float:
    token_count = len(anchor.split())
    if token_count >= 3:
        return 1.0
    if token_count == 2:
        return 0.9
    if len(anchor) >= 10:
        return 0.6
    return 0.25


def _collect_node_doc_scores(
    matcher: SimGRAGMicroMatcher,
    anchor_entities: list[str],
    node_to_triple_ids: dict[str, set[str]],
    triple_meta: dict[str, dict[str, Any]],
    node_doc_match_topk: int,
    node_doc_max_distance: float | None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not anchor_entities:
        return {}, []
    anchor_vectors = matcher._encode(anchor_entities)
    raw_hits = matcher.node_store.search(anchor_vectors, node_doc_match_topk)
    doc_anchor_scores: dict[str, dict[str, float]] = {}
    node_hits: list[dict[str, Any]] = []

    for anchor, hits in zip(anchor_entities, raw_hits):
        exact_added = False
        if anchor in node_to_triple_ids:
            hits = [{"distance": 0.0, "entity": {"name": anchor}}] + hits
            exact_added = True
        seen_nodes: set[str] = set()
        weight = _anchor_weight(anchor)
        for hit in hits:
            matched_node = processing_phrases(hit.get("entity", {}).get("name", ""))
            if not matched_node or matched_node in seen_nodes:
                continue
            seen_nodes.add(matched_node)
            raw_distance = float(hit.get("distance", 0.0))
            distance = (
                0.0
                if matched_node == anchor and exact_added
                else sqrt(max(0.0, raw_distance))
            )
            if node_doc_max_distance is not None and distance > node_doc_max_distance:
                continue
            triple_ids = sorted(node_to_triple_ids.get(matched_node, set()))
            if not triple_ids:
                continue
            hit_score = weight * (1.0 / (1.0 + distance))
            matched_triples = []
            for triple_id in triple_ids:
                triple_info = dict(triple_meta.get(triple_id, {"triple_id": triple_id}))
                matched_triples.append(triple_info)
                doc_id = str(triple_info.get("doc_id", "")).strip()
                if not doc_id:
                    continue
                doc_anchor_scores.setdefault(doc_id, {})
                doc_anchor_scores[doc_id][anchor] = max(
                    doc_anchor_scores[doc_id].get(anchor, 0.0), hit_score
                )
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

    doc_scores = {
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
    return doc_scores, node_hits


def _run_micro_score(args: argparse.Namespace) -> None:
    stage1_dir = Path(args.dataset_root) / args.data_name / "processed" / "stage1"
    test_path = Path(args.test_file) if args.test_file else stage1_dir / "test.json"
    micro_kg_path = (
        Path(args.micro_kg) if args.micro_kg else stage1_dir / "micro_kg.jsonl"
    )
    if not test_path.exists():
        raise FileNotFoundError(f"ScienceQA processed test file not found: {test_path}")
    if not micro_kg_path.exists():
        raise FileNotFoundError(f"ScienceQA micro kg not found: {micro_kg_path}")

    test_samples = load_json(test_path)
    if not isinstance(test_samples, list):
        raise TypeError(f"Expected list test file: {test_path}")

    if args.llm_config:
        llm_cfg = OmegaConf.load(args.llm_config)
    else:
        llm_cfg = OmegaConf.create(
            {
                "_target_": "gfmrag.llms.ChatGPT",
                "model_name_or_path": args.llm_model,
                "retry": 5,
            }
        )
    llm = instantiate(llm_cfg)
    parser = SimGRAGQueryParser(llm)

    matcher_cfg = OmegaConf.load(args.micro_matcher_config)
    micro_matcher: SimGRAGMicroMatcher = instantiate(matcher_cfg)
    micro_rows = _read_jsonl(micro_kg_path)
    if not micro_rows:
        raise RuntimeError(f"ScienceQA micro kg is empty: {micro_kg_path}")
    micro_matcher.build_index(micro_rows)
    triple_meta, node_to_triple_ids = _build_micro_metadata(micro_rows)

    query_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for sample in tqdm(test_samples, total=len(test_samples), desc="Micro score"):
        query_text = render_question_with_choices(
            str(sample.get("question", "")),
            [str(choice) for choice in sample.get("choices", [])],
        )
        parsed = parser.parse(query_text)
        condition_triples = parsed.get("condition_triples", [])
        target_triples = parsed.get("target_triples", [])
        anchor_entities = _select_anchor_entities(sample, condition_triples)

        graph_hits, graph_skip_reason = _safe_match_pattern(
            matcher=micro_matcher,
            condition_triples=condition_triples,
            topk=args.graph_topk,
        )
        graph_doc_scores, graph_hit_trace = _collect_graph_doc_scores(
            graph_hits,
            triple_meta=triple_meta,
            max_match_score=args.max_match_score,
        )
        node_doc_scores, node_hit_trace = _collect_node_doc_scores(
            matcher=micro_matcher,
            anchor_entities=anchor_entities,
            node_to_triple_ids=node_to_triple_ids,
            triple_meta=triple_meta,
            node_doc_match_topk=args.node_doc_match_topk,
            node_doc_max_distance=args.node_doc_max_distance,
        )
        doc_micro_scores = dict(graph_doc_scores)
        for doc_id, score in node_doc_scores.items():
            weighted_score = float(score) * args.node_doc_weight
            doc_micro_scores[doc_id] = max(
                doc_micro_scores.get(doc_id, 0.0), weighted_score
            )
        ranked_scores = _sort_doc_scores(doc_micro_scores, top_k=args.top_k)

        query_rows.append(
            {
                "id": sample.get("id"),
                "question": sample.get("question", ""),
                "choices": sample.get("choices", []),
                "condition_triples": condition_triples,
                "target_triples": target_triples,
                "anchor_entities": anchor_entities,
                "graph_skip_reason": graph_skip_reason,
            }
        )
        score_rows.append(
            {
                "id": sample.get("id"),
                "question": sample.get("question", ""),
                "choices": sample.get("choices", []),
                "answer": sample.get("answer"),
                "answer_letter": sample.get("answer_letter", ""),
                "question_images": sample.get("question_images", []),
                "question_image_captions": sample.get("question_image_captions", []),
                "micro_ranking": ranked_scores,
                "graph_skip_reason": graph_skip_reason,
                "graph_hits": graph_hit_trace,
                "node_hits": node_hit_trace,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    query_output = output_dir / "scienceqa_micro_queries.jsonl"
    score_output = output_dir / "scienceqa_micro_scores.jsonl"
    dump_jsonl(query_output, query_rows)
    dump_jsonl(score_output, score_rows)
    print(
        json.dumps(
            {
                "test_samples": len(test_samples),
                "query_output": str(query_output),
                "score_output": str(score_output),
            },
            ensure_ascii=False,
        )
    )


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


def _to_score_map(ranking: list[dict[str, Any]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in ranking:
        doc_id = str(item.get("doc_id", "")).strip()
        if not doc_id:
            continue
        scores[doc_id] = float(item.get("score", 0.0))
    return scores


def _sorted_ranking(
    score_map: dict[str, float], doc_corpus: dict[str, str], top_k: int
) -> list[dict[str, Any]]:
    ranked = sorted(
        score_map.items(),
        key=lambda item: (-float(item[1]), str(item[0])),
    )
    rows = []
    for doc_id, score in ranked[: max(1, int(top_k))]:
        rows.append(
            {
                "doc_id": str(doc_id),
                "score": float(score),
                "content": str(doc_corpus.get(str(doc_id), "")),
            }
        )
    return rows


def _run_merge(args: argparse.Namespace) -> None:
    macro_rows = _read_jsonl(args.macro_jsonl)
    micro_rows = _read_jsonl(args.micro_jsonl)
    micro_by_id = {str(row.get("id")): row for row in micro_rows}
    doc_corpus = load_json(
        Path(args.dataset_root) / args.data_name / "raw" / "dataset_corpus.json"
    )
    if not isinstance(doc_corpus, dict):
        raise TypeError("dataset_corpus.json must be a dict.")

    merged_rows: list[dict[str, Any]] = []
    for macro_row in macro_rows:
        sample_id = str(macro_row.get("id"))
        micro_row = micro_by_id.get(sample_id, {})
        macro_scores = _to_score_map(macro_row.get("macro_ranking", []))
        has_question_image = bool(
            macro_row.get("has_image")
            or macro_row.get("question_images")
            or macro_row.get("question_image_captions")
        )
        micro_scores = (
            _to_score_map(micro_row.get("micro_ranking", []))
            if has_question_image
            else {}
        )

        final_scores: dict[str, float] = {}
        for doc_id, score in macro_scores.items():
            final_scores[doc_id] = final_scores.get(doc_id, 0.0) + float(score)
        for doc_id, score in micro_scores.items():
            final_scores[doc_id] = final_scores.get(doc_id, 0.0) + float(score)

        final_ranking = _sorted_ranking(
            final_scores, doc_corpus=doc_corpus, top_k=args.top_k
        )
        top1_doc = final_ranking[0] if final_ranking else {}
        merged_rows.append(
            {
                "id": macro_row.get("id"),
                "question": macro_row.get("question", ""),
                "choices": macro_row.get("choices", []),
                "answer": macro_row.get("answer"),
                "answer_letter": macro_row.get("answer_letter", ""),
                "has_image": has_question_image,
                "question_images": macro_row.get("question_images", []),
                "question_image_captions": macro_row.get("question_image_captions", []),
                "top1_doc_id": top1_doc.get("doc_id", ""),
                "top1_doc_content": top1_doc.get("content", ""),
                "macro_ranking": macro_row.get("macro_ranking", []),
                "micro_ranking": micro_row.get("micro_ranking", []),
                "final_ranking": final_ranking,
            }
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(args.output_jsonl, merged_rows)
    print(
        json.dumps(
            {
                "macro_rows": len(macro_rows),
                "micro_rows": len(micro_rows),
                "output_jsonl": str(args.output_jsonl),
            },
            ensure_ascii=False,
        )
    )


# ---------------------------------------------------------------------------
# answer
# ---------------------------------------------------------------------------


def _resolve_problems_path(args: argparse.Namespace) -> Path | None:
    if args.problems_json:
        path = Path(args.problems_json)
        return path if path.exists() else None
    if args.allowed_local_media_path:
        candidate = Path(args.allowed_local_media_path) / "problems.json"
        if candidate.exists():
            return candidate
    return None


def _get_top1_doc(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("top1_doc_content"):
        return str(row.get("top1_doc_id", "")), str(row["top1_doc_content"])
    top1_macro = row.get("top1_macro_doc") or {}
    return str(top1_macro.get("doc_id", "")), str(top1_macro.get("content", ""))


def _build_prompt(
    row: dict[str, Any], user_prompt_template: str, zeroshot: bool
) -> str:
    image_names = [
        str(item.get("filename", "")).strip() for item in row.get("question_images", [])
    ]
    image_names = [name for name in image_names if name]
    image_block = ", ".join(image_names) if image_names else "None"
    _, top1_doc_content = _get_top1_doc(row)
    if zeroshot:
        top1_doc_content = ""
    return user_prompt_template.format(
        question=row.get("question", ""),
        choices=render_choices([str(choice) for choice in row.get("choices", [])]),
        image_names=image_block,
        top1_doc_content=top1_doc_content,
    )


def _resolve_image_paths(row: dict[str, Any], image_root: Path | None) -> list[str]:
    paths: list[str] = []
    for item in row.get("question_images", []):
        if image_root is not None:
            relative = str(item.get("relative_path", "")).strip()
            if relative:
                candidate = image_root / relative
                if candidate.exists():
                    paths.append(str(candidate.resolve()))
                    continue
        absolute = str(item.get("absolute_path", "")).strip()
        if absolute:
            paths.append(absolute)
    return paths


def _run_answer(args: argparse.Namespace, client: Any = None) -> None:
    rows = _read_jsonl(args.input_jsonl)
    if client is None:
        client = _create_client(args)
    problems_path = _resolve_problems_path(args)
    problems = _load_problems(problems_path) if problems_path is not None else {}

    prompt_cfg = OmegaConf.to_container(OmegaConf.load(args.prompt_file), resolve=True)
    if not isinstance(prompt_cfg, dict):
        raise ValueError(f"Invalid prompt config: {args.prompt_file}")
    system_prompt = str(prompt_cfg.get("system_prompt", "")).strip()
    user_prompt_template = str(prompt_cfg.get("user_prompt", "")).strip()
    if not user_prompt_template:
        raise ValueError(f"Missing user_prompt in prompt config: {args.prompt_file}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    total = 0
    subject_stats: dict[str, dict[str, int]] = {}
    modality_stats: dict[str, dict[str, int]] = {}
    grade_stats: dict[str, dict[str, int]] = {}
    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for row in tqdm(
            rows, total=len(rows), desc="ScienceQA Eval", dynamic_ncols=True
        ):
            image_paths = _resolve_image_paths(row, args.image_root)
            user_prompt = _build_prompt(row, user_prompt_template, args.zeroshot)
            prompt_text = (
                f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
            )
            response = client.chat(
                prompt=prompt_text,
                images=image_paths or None,
                sampling_overrides={"max_tokens": args.max_tokens},
            )
            parsed_letter = parse_answer_letter(str(response))
            parsed_index = (
                "ABCDE".index(parsed_letter)
                if parsed_letter and parsed_letter in "ABCDE"
                else -1
            )
            gold_index = int(row.get("answer", -1))
            gold_letter = str(
                row.get("answer_letter") or answer_index_to_letter(gold_index)
            )
            is_correct = parsed_index == gold_index and parsed_index >= 0
            total += 1
            correct += int(is_correct)
            sample_id = str(row.get("id", ""))
            problem = problems.get(sample_id, {})
            subject_bucket = _subject_bucket(problem.get("subject", ""))
            modality_bucket = _modality_bucket(problem) if problem else "UNKNOWN"
            grade_bucket = (
                _grade_bucket(problem.get("grade", "")) if problem else "UNKNOWN"
            )
            _update_bucket(subject_stats, subject_bucket, is_correct)
            _update_bucket(modality_stats, modality_bucket, is_correct)
            _update_bucket(grade_stats, grade_bucket, is_correct)
            top1_doc_id, top1_doc_content = _get_top1_doc(row)
            result = {
                "id": sample_id,
                "question": row.get("question", ""),
                "choices": row.get("choices", []),
                "answer": gold_index,
                "answer_letter": gold_letter,
                "response": response,
                "parsed_letter": parsed_letter,
                "parsed_index": parsed_index,
                "is_correct": is_correct,
                "top1_doc_id": top1_doc_id,
                "top1_doc_content": top1_doc_content,
                "subject_bucket": subject_bucket,
                "modality_bucket": modality_bucket,
                "grade_bucket": grade_bucket,
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()

    accuracy = (correct / total) if total else 0.0
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "zeroshot": bool(args.zeroshot),
        "subject_accuracy": _summarize_bucket(subject_stats),
        "modality_accuracy": _summarize_bucket(modality_stats),
        "grade_accuracy": _summarize_bucket(grade_stats),
        "problems_json": str(problems_path) if problems_path else "",
        "output_jsonl": str(args.output_jsonl),
    }
    summary_path = args.output_jsonl.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------


def _parse_final_decision_letter(text: str) -> str:
    content = str(text or "").strip()
    if not content:
        return ""

    patterns = [
        r"Final\s*Decision\s*:\s*([A-E])\b",
        r"Final\s*answer(?:\s*letter)?\s*:\s*([A-E])\b",
        r"Answer\s*:\s*([A-E])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, content, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    for line in reversed(lines[-8:]):
        match = re.fullmatch(r"[\(\[]?([A-E])[\)\]]?", line, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    letters = re.findall(r"\b([A-E])\b", content.upper())
    if letters:
        return letters[-1].upper()
    return ""


def _bucket_with_fallback(value: str, fallback: str) -> str:
    return str(fallback) if fallback else value


def _load_problem_images(
    problem_id: str, problem: dict[str, Any], image_root: Path
) -> list[dict[str, str]]:
    image_name = str(problem.get("image", "")).strip()
    if not image_name or image_name.lower() == "null":
        return []

    candidates = [
        Path(image_name),
        image_root / image_name,
        image_root / problem_id / image_name,
        image_root / str(problem.get("split", "")).strip() / problem_id / image_name,
    ]
    seen = set()
    images: list[dict[str, str]] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            images.append(
                {
                    "filename": candidate.name,
                    "absolute_path": str(candidate.resolve()),
                }
            )
    if images:
        return images

    guessed = image_root / problem_id / image_name
    return [{"filename": image_name, "absolute_path": str(guessed)}]


def _load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for row in _read_jsonl(path):
        qid = str(row.get("id", "")).strip()
        if qid:
            done.add(qid)
    return done


def _call_model_with_retry(
    client: Any,
    prompt_text: str,
    image_paths: Sequence[str],
    max_tokens: int,
    retry_times: int,
    retry_wait_sec: float,
) -> tuple[str, str | None, int]:
    attempts = 0
    error_message: str | None = None
    for attempt in range(max(0, int(retry_times)) + 1):
        attempts = attempt + 1
        try:
            response = client.chat(
                prompt=prompt_text,
                images=list(image_paths) or None,
                system_prompt="You are an expert science evaluator.",
                sampling_overrides={
                    "max_tokens": int(max_tokens),
                    "temperature": 0.0,
                    "top_k": -1,
                    "top_p": 0.9,
                },
            )
            return str(response or "").strip(), None, attempts
        except Exception as exc:
            error_message = str(exc)
            if attempt >= int(retry_times):
                break
            time.sleep(max(0.0, float(retry_wait_sec)))
    return "", error_message, attempts


def _build_judge_prompt(
    problem: dict[str, Any],
    direct_row: dict[str, Any],
    rag_row: dict[str, Any],
    image_names: Sequence[str],
) -> str:
    image_block = (
        ", ".join([name for name in image_names if name]) if image_names else "None"
    )
    return (
        "You are an expert science evaluator. Your task is to determine the correct answer to a multiple-choice question "
        "by analyzing two different AI-generated responses.\n\n"
        "--- Target Question ---\n"
        f"Question: {problem.get('question', direct_row.get('question', ''))}\n"
        f"Question image filenames: {image_block}\n"
        f"Choices:\n{render_choices([str(choice) for choice in direct_row.get('choices', [])])}\n\n"
        "--- Response A (Direct Knowledge) ---\n"
        f"{str(direct_row.get('response', '')).strip()}\n\n"
        "--- Response B (Retrieval-Augmented) ---\n"
        f"{str(rag_row.get('response', '')).strip()}\n\n"
        "--- Evaluation Guidelines ---\n"
        "Response A relies on direct internal knowledge.\n"
        "Response B relies on retrieved reference materials. You must critically evaluate both to find the true answer. "
        "Watch for these common failure modes in both responses:\n"
        "1. Truncation / Rambling: If a response is cut off, overly long, or fails to explicitly state a final answer letter at the end, its reasoning chain is likely broken.\n"
        "2. Ungrounded reasoning: Response B may force a connection between the reference and the question that does not logically exist; Response A may state an answer from misremembered or missing knowledge. Check the reasoning itself, not the confidence.\n"
        "3. Hallucination: Check whether either response invents facts not supported by the question, the choices, or (for Response B) the reference.\n\n"
        "--- Task ---\n"
        "1. Briefly compare the reasoning of Response A and Response B based on the guidelines.\n"
        "2. Determine the objectively correct answer based on the question, choices, and image if provided.\n"
        "3. You MUST end your response strictly with the format: Final Decision: <Letter>\n"
    )


def _create_auto_row(
    qid: str,
    direct_row: dict[str, Any],
    rag_row: dict[str, Any],
    problem: dict[str, Any],
) -> dict[str, Any]:
    direct_is_correct = _to_bool(direct_row.get("is_correct"))
    rag_is_correct = _to_bool(rag_row.get("is_correct"))
    if direct_is_correct and rag_is_correct:
        final_source = "auto_both_correct"
        final_letter = str(
            direct_row.get("parsed_letter")
            or rag_row.get("parsed_letter")
            or direct_row.get("answer_letter")
            or ""
        )
        final_is_correct = True
    else:
        final_source = "auto_both_wrong"
        final_letter = str(
            direct_row.get("parsed_letter") or rag_row.get("parsed_letter") or ""
        )
        final_is_correct = False

    final_index = "ABCDE".find(final_letter) if final_letter in "ABCDE" else -1
    return {
        "id": qid,
        "question": direct_row.get("question", ""),
        "choices": direct_row.get("choices", []),
        "answer": direct_row.get("answer", -1),
        "answer_letter": direct_row.get("answer_letter", ""),
        "direct_response": direct_row.get("response", ""),
        "direct_parsed_letter": direct_row.get("parsed_letter", ""),
        "direct_is_correct": direct_is_correct,
        "rag_response": rag_row.get("response", ""),
        "rag_parsed_letter": rag_row.get("parsed_letter", ""),
        "rag_is_correct": rag_is_correct,
        "needs_judge": False,
        "judge_prompt": "",
        "judge_response": "",
        "judge_decision_letter": "",
        "judge_decision_index": -1,
        "judge_error": "",
        "judge_attempts": 0,
        "final_source": final_source,
        "final_decision_letter": final_letter,
        "final_decision_index": final_index,
        "final_is_correct": final_is_correct,
        "subject_bucket": _bucket_with_fallback(
            _subject_bucket(problem.get("subject", "")),
            str(direct_row.get("subject_bucket", "")),
        ),
        "modality_bucket": _bucket_with_fallback(
            _modality_bucket(problem) if problem else "UNKNOWN",
            str(direct_row.get("modality_bucket", "")),
        ),
        "grade_bucket": _bucket_with_fallback(
            _grade_bucket(problem.get("grade", "")) if problem else "UNKNOWN",
            str(direct_row.get("grade_bucket", "")),
        ),
    }


def _process_judge_sample(
    qid: str,
    direct_row: dict[str, Any],
    rag_row: dict[str, Any],
    problem: dict[str, Any],
    caption_map: dict[str, str],
    image_root: Path,
    client: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    question_images = _load_problem_images(qid, problem, image_root)
    existing_image_paths: list[str] = []
    image_names: list[str] = []
    missing_images: list[str] = []
    for item in question_images:
        image_names.append(str(item.get("filename", "")).strip())
        raw_path = str(item.get("absolute_path", "")).strip()
        if raw_path and Path(raw_path).exists():
            existing_image_paths.append(raw_path)
        elif raw_path:
            missing_images.append(raw_path)

    prompt = _build_judge_prompt(
        problem=problem,
        direct_row=direct_row,
        rag_row=rag_row,
        image_names=image_names,
    )
    response, error_message, attempts = _call_model_with_retry(
        client=client,
        prompt_text=prompt,
        image_paths=existing_image_paths,
        max_tokens=int(args.max_tokens),
        retry_times=int(args.retry_times),
        retry_wait_sec=float(args.retry_wait_sec),
    )
    judge_letter = _parse_final_decision_letter(response)
    judge_index = "ABCDE".find(judge_letter) if judge_letter in "ABCDE" else -1
    gold_index = int(direct_row.get("answer", -1))
    final_is_correct = judge_index == gold_index and judge_index >= 0

    direct_letter = str(direct_row.get("parsed_letter", "")).strip().upper()
    rag_letter = str(rag_row.get("parsed_letter", "")).strip().upper()
    if judge_letter and judge_letter == direct_letter and judge_letter != rag_letter:
        final_source = "judge_matches_direct"
    elif judge_letter and judge_letter == rag_letter and judge_letter != direct_letter:
        final_source = "judge_matches_rag"
    elif judge_letter and judge_letter == direct_letter and judge_letter == rag_letter:
        final_source = "judge_matches_both"
    elif judge_letter:
        final_source = "judge_new_letter"
    else:
        final_source = "judge_unparsed"

    return {
        "id": qid,
        "question": direct_row.get("question", ""),
        "choices": direct_row.get("choices", []),
        "answer": gold_index,
        "answer_letter": direct_row.get("answer_letter", ""),
        "direct_response": direct_row.get("response", ""),
        "direct_parsed_letter": direct_letter,
        "direct_is_correct": _to_bool(direct_row.get("is_correct")),
        "rag_response": rag_row.get("response", ""),
        "rag_parsed_letter": rag_letter,
        "rag_is_correct": _to_bool(rag_row.get("is_correct")),
        "needs_judge": True,
        "question_images": question_images,
        "attached_image_paths": existing_image_paths,
        "missing_image_paths": missing_images,
        "question_image_caption": caption_map.get(qid, ""),
        "judge_prompt": prompt,
        "judge_response": response,
        "judge_decision_letter": judge_letter,
        "judge_decision_index": judge_index,
        "judge_error": error_message or "",
        "judge_attempts": attempts,
        "final_source": final_source,
        "final_decision_letter": judge_letter,
        "final_decision_index": judge_index,
        "final_is_correct": final_is_correct,
        "subject_bucket": _bucket_with_fallback(
            _subject_bucket(problem.get("subject", "")),
            str(direct_row.get("subject_bucket", "")),
        ),
        "modality_bucket": _bucket_with_fallback(
            _modality_bucket(problem) if problem else "UNKNOWN",
            str(direct_row.get("modality_bucket", "")),
        ),
        "grade_bucket": _bucket_with_fallback(
            _grade_bucket(problem.get("grade", "")) if problem else "UNKNOWN",
            str(direct_row.get("grade_bucket", "")),
        ),
    }


def _summarize_judge_results(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    final_correct = 0
    direct_correct = 0
    rag_correct = 0
    oracle_correct = 0
    judge_needed = 0
    judge_done = 0
    judge_failed_parse = 0
    source_counts: dict[str, int] = {}
    subject_stats: dict[str, dict[str, int]] = {}
    modality_stats: dict[str, dict[str, int]] = {}
    grade_stats: dict[str, dict[str, int]] = {}

    for row in rows:
        total += 1
        direct_is_correct = _to_bool(row.get("direct_is_correct"))
        rag_is_correct = _to_bool(row.get("rag_is_correct"))
        final_is_correct = _to_bool(row.get("final_is_correct"))
        needs_judge = _to_bool(row.get("needs_judge"))

        direct_correct += int(direct_is_correct)
        rag_correct += int(rag_is_correct)
        oracle_correct += int(direct_is_correct or rag_is_correct)
        final_correct += int(final_is_correct)
        judge_needed += int(needs_judge)
        judge_done += int(
            needs_judge and str(row.get("judge_response", "")).strip() != ""
        )
        judge_failed_parse += int(
            needs_judge and not str(row.get("judge_decision_letter", "")).strip()
        )

        source = str(row.get("final_source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1

        _update_bucket(
            subject_stats, str(row.get("subject_bucket", "UNKNOWN")), final_is_correct
        )
        _update_bucket(
            modality_stats, str(row.get("modality_bucket", "UNKNOWN")), final_is_correct
        )
        _update_bucket(
            grade_stats, str(row.get("grade_bucket", "UNKNOWN")), final_is_correct
        )

    return {
        "total": total,
        "direct_accuracy": (direct_correct / total) if total else 0.0,
        "rag_accuracy": (rag_correct / total) if total else 0.0,
        "oracle_or_accuracy": (oracle_correct / total) if total else 0.0,
        "final_accuracy": (final_correct / total) if total else 0.0,
        "judge_needed_count": judge_needed,
        "judge_done_count": judge_done,
        "judge_failed_parse_count": judge_failed_parse,
        "final_source_counts": source_counts,
        "subject_accuracy": _summarize_bucket(subject_stats),
        "modality_accuracy": _summarize_bucket(modality_stats),
        "grade_accuracy": _summarize_bucket(grade_stats),
    }


def _run_judge(args: argparse.Namespace, client: Any = None) -> None:
    if not args.direct_file.exists():
        raise FileNotFoundError(f"direct file not found: {args.direct_file}")
    if not args.rag_file.exists():
        raise FileNotFoundError(f"rag file not found: {args.rag_file}")
    if not args.problems_json.exists():
        raise FileNotFoundError(f"problems json not found: {args.problems_json}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    if args.output_jsonl is None:
        stem = f"{args.direct_file.stem}__vs__{args.rag_file.stem}__judge"
        args.output_jsonl = Path(f"{stem}.jsonl")
    if args.output_summary is None:
        args.output_summary = args.output_jsonl.with_suffix(".summary.json")

    direct_rows = _read_jsonl(args.direct_file)
    rag_rows = _read_jsonl(args.rag_file)
    problems = _load_problems(args.problems_json)

    caption_map: dict[str, str] = {}
    if args.captions_json.exists():
        captions_payload = load_json(args.captions_json)
        if isinstance(captions_payload, dict):
            raw_captions = captions_payload.get("captions", captions_payload)
            if isinstance(raw_captions, dict):
                caption_map = {str(k): str(v) for k, v in raw_captions.items()}

    direct_by_id = {str(row.get("id")): row for row in direct_rows}
    rag_by_id = {str(row.get("id")): row for row in rag_rows}
    common_ids = sorted(
        set(direct_by_id.keys()) & set(rag_by_id.keys()),
        key=lambda x: int(x) if x.isdigit() else x,
    )

    done_ids = set() if args.no_resume else _load_done_ids(args.output_jsonl)

    auto_rows: list[dict[str, Any]] = []
    judge_jobs: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for qid in common_ids:
        if qid in done_ids:
            continue
        direct_row = direct_by_id[qid]
        rag_row = rag_by_id[qid]
        problem = problems.get(qid, {})
        direct_is_correct = _to_bool(direct_row.get("is_correct"))
        rag_is_correct = _to_bool(rag_row.get("is_correct"))
        if direct_is_correct == rag_is_correct:
            auto_rows.append(_create_auto_row(qid, direct_row, rag_row, problem))
        else:
            judge_jobs.append((qid, direct_row, rag_row, problem))

    start_judged = max(0, int(args.start_judged))
    judge_jobs = judge_jobs[start_judged:]
    if args.max_judged is not None:
        judge_jobs = judge_jobs[: max(0, int(args.max_judged))]

    if not args.allowed_local_media_path:
        args.allowed_local_media_path = str(args.image_root.resolve())
    if client is None:
        client = _create_client(args)

    for row in auto_rows:
        _append_jsonl_row(args.output_jsonl, row)

    print(
        f"common={len(common_ids)} auto_now={len(auto_rows)} judge_pending={len(judge_jobs)} "
        f"resume_skipped={len(done_ids)} batch_size={args.batch_size}",
        flush=True,
    )

    lock = threading.Lock()
    processed_judge = 0

    def handle_result(result_row: dict[str, Any]) -> None:
        nonlocal processed_judge
        with lock:
            _append_jsonl_row(args.output_jsonl, result_row)
            processed_judge += 1
            if processed_judge % max(1, int(args.save_every)) == 0:
                print(f"saved judged={processed_judge}", flush=True)
            judge_letter = str(result_row.get("judge_decision_letter", "")).strip()
            print(
                f"[judge] id={result_row.get('id', '')} "
                f"final={judge_letter or 'UNPARSED'} "
                f"gold={result_row.get('answer_letter', '')} "
                f"ok={int(_to_bool(result_row.get('final_is_correct')))} "
                f"source={result_row.get('final_source', '')} "
                f"attempts={result_row.get('judge_attempts', 0)}",
                flush=True,
            )

    if judge_jobs:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=int(args.batch_size)
        ) as executor:
            futures = [
                executor.submit(
                    _process_judge_sample,
                    qid,
                    direct_row,
                    rag_row,
                    problem,
                    caption_map,
                    args.image_root,
                    client,
                    args,
                )
                for qid, direct_row, rag_row, problem in judge_jobs
            ]
            for future in concurrent.futures.as_completed(futures):
                handle_result(future.result())
                if args.sleep_sec > 0:
                    time.sleep(float(args.sleep_sec))

    output_rows = _read_jsonl(args.output_jsonl) if args.output_jsonl.exists() else []
    summary = _summarize_judge_results(output_rows)
    summary.update(
        {
            "common_total": len(common_ids),
            "resume_skipped_count": len(done_ids),
            "auto_written_now": len(auto_rows),
            "judge_requested_now": len(judge_jobs),
            "judge_processed_now": processed_judge,
            "start_judged": start_judged,
            "max_judged": args.max_judged,
            "direct_file": str(args.direct_file),
            "rag_file": str(args.rag_file),
            "problems_json": str(args.problems_json),
            "captions_json": str(args.captions_json),
            "image_root": str(args.image_root),
            "output_jsonl": str(args.output_jsonl),
            "model_name_or_path": args.model_name_or_path,
        }
    )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_summary, "w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"Output: {args.output_jsonl}", flush=True)
    print(f"Summary: {args.output_summary}", flush=True)


# ---------------------------------------------------------------------------
# evaluate (chained: micro-score -> merge -> answer RAG/direct -> judge)
# ---------------------------------------------------------------------------


def _run_evaluate(args: argparse.Namespace) -> None:
    """Run the full direct-vs-RAG evaluation chain with one shared vLLM client."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    micro_args = argparse.Namespace(
        dataset_root=args.dataset_root,
        data_name=args.data_name,
        test_file="",
        micro_kg="",
        output_dir=str(output_dir),
        micro_matcher_config=args.micro_matcher_config,
        llm_config=args.llm_config,
        llm_model=args.llm_model,
        top_k=args.micro_top_k,
        graph_topk=args.graph_topk,
        max_match_score=args.max_match_score,
        node_doc_match_topk=args.node_doc_match_topk,
        node_doc_max_distance=args.node_doc_max_distance,
        node_doc_weight=args.node_doc_weight,
    )
    _run_micro_score(micro_args)

    merged_jsonl = output_dir / "scienceqa_merged.jsonl"
    merge_args = argparse.Namespace(
        macro_jsonl=args.macro_jsonl,
        micro_jsonl=output_dir / "scienceqa_micro_scores.jsonl",
        dataset_root=args.dataset_root,
        data_name=args.data_name,
        output_jsonl=merged_jsonl,
        top_k=args.merge_top_k,
    )
    _run_merge(merge_args)

    if not args.allowed_local_media_path:
        args.allowed_local_media_path = str(args.image_root.resolve())
    client = _create_client(args)

    rag_jsonl = output_dir / "scienceqa_rag_answers.jsonl"
    direct_jsonl = output_dir / "scienceqa_direct_answers.jsonl"
    for zeroshot, out_path in ((False, rag_jsonl), (True, direct_jsonl)):
        answer_fields = dict(vars(args))
        answer_fields.update(
            input_jsonl=merged_jsonl,
            output_jsonl=out_path,
            zeroshot=zeroshot,
        )
        _run_answer(argparse.Namespace(**answer_fields), client=client)

    judge_fields = dict(vars(args))
    judge_fields.update(
        direct_file=direct_jsonl,
        rag_file=rag_jsonl,
        problems_json=Path(args.problems_json),
        captions_json=Path(args.captions_json),
        image_root=args.image_root,
        output_jsonl=output_dir / "scienceqa_judge.jsonl",
        output_summary=None,
    )
    _run_judge(argparse.Namespace(**judge_fields), client=client)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    micro = subparsers.add_parser(
        "micro-score",
        help="Score question anchors against the micro matching graph.",
    )
    micro.add_argument("--dataset-root", type=str, default="data")
    micro.add_argument("--data-name", type=str, default="scienceqa")
    micro.add_argument(
        "--test-file",
        type=str,
        default="",
        help="Processed test.json; default <root>/<name>/processed/stage1/test.json",
    )
    micro.add_argument(
        "--micro-kg",
        type=str,
        default="",
        help="micro_kg.jsonl; default <root>/<name>/processed/stage1/micro_kg.jsonl",
    )
    micro.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for scienceqa_micro_queries.jsonl / scienceqa_micro_scores.jsonl",
    )
    micro.add_argument(
        "--micro-matcher-config",
        type=Path,
        default=DEFAULT_MICRO_MATCHER_CONFIG,
    )
    micro.add_argument(
        "--llm-config",
        type=str,
        default="",
        help="Optional hydra-style yaml for the query-parser LLM.",
    )
    micro.add_argument("--llm-model", type=str, default="gpt-3.5-turbo")
    micro.add_argument("--top-k", type=int, default=200)
    micro.add_argument("--graph-topk", type=int, default=20)
    micro.add_argument("--max-match-score", type=float, default=10.0)
    micro.add_argument("--node-doc-match-topk", type=int, default=30)
    micro.add_argument("--node-doc-max-distance", type=float, default=1.0)
    micro.add_argument("--node-doc-weight", type=float, default=1.0)
    micro.set_defaults(func=_run_micro_score)

    merge = subparsers.add_parser(
        "merge",
        help="Additively merge macro and micro rankings (micro only for image questions).",
    )
    merge.add_argument("--macro-jsonl", type=Path, required=True)
    merge.add_argument("--micro-jsonl", type=Path, required=True)
    merge.add_argument("--dataset-root", type=str, default="data")
    merge.add_argument("--data-name", type=str, default="scienceqa")
    merge.add_argument("--output-jsonl", type=Path, required=True)
    merge.add_argument("--top-k", type=int, default=50)
    merge.set_defaults(func=_run_merge)

    answer = subparsers.add_parser(
        "answer",
        help="Answer questions with Qwen3-VL over the merged top-1 document.",
    )
    answer.add_argument("--input-jsonl", type=Path, required=True)
    answer.add_argument("--output-jsonl", type=Path, required=True)
    answer.add_argument("--prompt-file", type=Path, default=DEFAULT_ANSWER_PROMPT)
    answer.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="ScienceQA image_root directory; question images are re-resolved by relative_path.",
    )
    answer.add_argument("--problems-json", type=str, default="")
    answer.add_argument(
        "--zeroshot",
        action="store_true",
        help="Exclude the retrieved top-1 document from the prompt (direct no-retrieval run).",
    )
    _add_vllm_args(answer)
    answer.set_defaults(func=_run_answer)

    judge = subparsers.add_parser(
        "judge",
        help="Arbitrate direct-vs-RAG disagreements with a Qwen3-VL judge.",
    )
    judge.add_argument("--direct-file", type=Path, required=True)
    judge.add_argument("--rag-file", type=Path, required=True)
    judge.add_argument(
        "--problems-json", type=Path, default=Path("data/scienceqa/problems.json")
    )
    judge.add_argument(
        "--captions-json", type=Path, default=Path("data/scienceqa/captions.json")
    )
    judge.add_argument(
        "--image-root", type=Path, default=Path("data/scienceqa/pictures")
    )
    judge.add_argument("--output-jsonl", type=Path, default=None)
    judge.add_argument("--output-summary", type=Path, default=None)
    judge.add_argument("--batch-size", type=int, default=1)
    judge.add_argument("--retry-times", type=int, default=1)
    judge.add_argument("--retry-wait-sec", type=float, default=5.0)
    judge.add_argument("--sleep-sec", type=float, default=0.0)
    judge.add_argument("--save-every", type=int, default=10)
    judge.add_argument("--max-judged", type=int, default=None)
    judge.add_argument("--start-judged", type=int, default=0)
    judge.add_argument("--no-resume", action="store_true")
    _add_vllm_args(judge)
    judge.set_defaults(func=_run_judge)

    evaluate = subparsers.add_parser(
        "evaluate",
        help=(
            "Run the full direct-vs-RAG chain (micro-score -> merge -> answer RAG "
            "+ direct -> judge) with one shared vLLM client; intermediates go to "
            "--output-dir."
        ),
    )
    evaluate.add_argument(
        "--macro-jsonl",
        type=Path,
        required=True,
        help="Macro ranking export from stage3 (scienceqa_export_macro_only=true).",
    )
    evaluate.add_argument("--output-dir", type=str, default="outputs/scienceqa_eval")
    evaluate.add_argument("--dataset-root", type=str, default="data")
    evaluate.add_argument("--data-name", type=str, default="scienceqa")
    evaluate.add_argument(
        "--image-root", type=Path, default=Path("data/scienceqa/pictures")
    )
    evaluate.add_argument(
        "--problems-json", type=str, default="data/scienceqa/problems.json"
    )
    evaluate.add_argument(
        "--captions-json", type=Path, default=Path("data/scienceqa/captions.json")
    )
    evaluate.add_argument("--prompt-file", type=Path, default=DEFAULT_ANSWER_PROMPT)
    evaluate.add_argument(
        "--micro-matcher-config", type=Path, default=DEFAULT_MICRO_MATCHER_CONFIG
    )
    evaluate.add_argument("--llm-config", type=str, default="")
    evaluate.add_argument("--llm-model", type=str, default="gpt-3.5-turbo")
    evaluate.add_argument("--micro-top-k", type=int, default=200)
    evaluate.add_argument("--graph-topk", type=int, default=20)
    evaluate.add_argument("--max-match-score", type=float, default=10.0)
    evaluate.add_argument("--node-doc-match-topk", type=int, default=30)
    evaluate.add_argument("--node-doc-max-distance", type=float, default=1.0)
    evaluate.add_argument("--node-doc-weight", type=float, default=1.0)
    evaluate.add_argument("--merge-top-k", type=int, default=50)
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--retry-times", type=int, default=1)
    evaluate.add_argument("--retry-wait-sec", type=float, default=5.0)
    evaluate.add_argument("--sleep-sec", type=float, default=0.0)
    evaluate.add_argument("--save-every", type=int, default=10)
    evaluate.add_argument("--max-judged", type=int, default=None)
    evaluate.add_argument("--start-judged", type=int, default=0)
    evaluate.add_argument("--no-resume", action="store_true")
    _add_vllm_args(evaluate)
    evaluate.set_defaults(func=_run_evaluate)

    return parser


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] not in SUBCOMMANDS:
        argv.insert(0, "answer")
    args = _build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
