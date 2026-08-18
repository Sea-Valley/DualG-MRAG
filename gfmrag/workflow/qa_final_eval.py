from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from gfmrag.llms.qwenvl_vllm import get_or_create_qwen_vl_client

logger = logging.getLogger(__name__)

VITERBI_PATH_LIMIT = 1

JUDGE_SYSTEM_PROMPT = (
    "You are a strict but fair judge for question answering. You are shown a "
    "question and two candidate answers (A and B) from two systems. You do NOT see "
    "any reference answer or numeric scores. Judge solely which candidate is more "
    "likely to be factually correct and to directly answer the question. Respond "
    "with a single JSON object only, no markdown, no extra text."
)

JUDGE_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Candidate A: {answer_a}\n\n"
    "Candidate B: {answer_b}\n\n"
    'Which candidate is better? Respond as {{"choice": "A"}} or {{"choice": "B"}}.'
)


def extract_answer_from_response(response: str) -> str:
    text = (response or "").strip()
    if not text:
        return ""
    if "Answer:" in text:
        answer = text.split("Answer:")[-1].strip()
        return answer.splitlines()[0].strip()
    return text.splitlines()[0].strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_retrieval_recall_at_k(
    supporting_facts: list[str], retrieved_docs: list[dict[str, Any]], k: int
) -> float:
    if not supporting_facts:
        return 0.0
    gold_set = {
        str(item).strip().lower() for item in supporting_facts if str(item).strip()
    }
    if not gold_set:
        return 0.0
    pred_set = {
        str(doc.get("title", "")).strip().lower()
        for doc in retrieved_docs[: max(0, int(k))]
        if str(doc.get("title", "")).strip()
    }
    if not pred_set:
        return 0.0
    return len(gold_set & pred_set) / len(gold_set)


def render_table_text(table: dict[str, Any]) -> str:
    schema = [str(col) for col in table.get("schema", [])]
    rows = table.get("rows", [])
    lines = [" | ".join(schema)] if schema else []
    for row in rows:
        if isinstance(row, dict):
            lines.append(" | ".join(str(row.get(col, "")) for col in schema))
    return "\n".join(lines)


class CorpusResolver:
    """Resolve retrieved doc tokens against the released raw corpus files.

    Reads ``dataset_corpus.json`` (doc_id -> "title\\nbody") and
    ``dataset_multimodal.json`` (list of {doc_id, title, text, images, tables})
    from the dataset raw directory.
    """

    def __init__(
        self,
        dataset_root: Path,
        data_name: str,
        pictures_root: Path | None = None,
        tables_root: Path | None = None,
        table_image_extensions: list[str] | None = None,
    ) -> None:
        self.dataset_root = dataset_root
        self.data_name = data_name
        raw_dir = dataset_root / data_name / "raw"
        if not raw_dir.exists():
            raw_dir = dataset_root / data_name
        self.pictures_root = pictures_root or (dataset_root / data_name / "pictures")
        self.tables_root = tables_root or (dataset_root / data_name / "tables")
        self.table_image_extensions = table_image_extensions or [
            "png",
            "jpg",
            "jpeg",
            "webp",
        ]

        self.docs: dict[str, dict[str, Any]] = {}
        self.title2ids: dict[str, list[str]] = {}
        self._load(raw_dir)

    def _register(self, doc_id: str, doc: dict[str, Any]) -> None:
        doc_id = str(doc_id).strip()
        if not doc_id:
            return
        self.docs[doc_id] = doc
        title = str(doc.get("title", "")).strip().lower()
        if title:
            self.title2ids.setdefault(title, []).append(doc_id)

    def _load(self, raw_dir: Path) -> None:
        corpus_path = raw_dir / "dataset_corpus.json"
        if corpus_path.exists():
            payload = json.loads(corpus_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for doc_id, value in payload.items():
                    if isinstance(value, str):
                        parts = value.split("\n", 1)
                        title = parts[0].strip()
                        text = parts[1].strip() if len(parts) > 1 else ""
                        self._register(
                            doc_id,
                            {"title": title, "text": text, "images": [], "tables": []},
                        )
                    elif isinstance(value, dict):
                        self._register(doc_id, value)

        multimodal_path = raw_dir / "dataset_multimodal.json"
        if multimodal_path.exists():
            payload = json.loads(multimodal_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                items = list(payload.values())
            elif isinstance(payload, list):
                items = payload
            else:
                items = []
            for item in items:
                if isinstance(item, dict) and item.get("doc_id"):
                    self._register(str(item["doc_id"]), item)

    def candidate_ids(self, token: str) -> list[str]:
        t = str(token).strip()
        if not t:
            return []
        candidates = [t] if t in self.docs else []
        candidates.extend(self.title2ids.get(t.lower(), []))
        return list(dict.fromkeys(candidates))

    def resolve(self, token: str, fallback_content: str = "") -> dict[str, Any]:
        for doc_id in self.candidate_ids(token):
            doc = self.docs[doc_id]
            title = str(doc.get("title", "")).strip() or doc_id
            content = str(doc.get("text", "")).strip()
            caption = ""
            images = doc.get("images") or []
            tables = doc.get("tables") or []
            if images:
                caption = str(images[0].get("caption", "")).strip()
            if not content and tables:
                content = render_table_text(tables[0])
            if not content and images:
                content = caption
            return {
                "doc_id": doc_id,
                "title": title,
                "content": content or str(fallback_content).strip(),
                "caption": caption,
            }
        return {
            "doc_id": "",
            "title": str(token).strip(),
            "content": str(fallback_content).strip(),
            "caption": "",
        }

    def resolve_image_paths(self, token: str) -> list[str]:
        results: list[str] = []
        for doc_id in self.candidate_ids(token):
            doc = self.docs[doc_id]
            for image in doc.get("images") or []:
                raw = str(image.get("path_or_url", image.get("path", ""))).strip()
                if not raw:
                    continue
                if raw.startswith(("http://", "https://")):
                    results.append(raw)
                    continue
                candidate = self.pictures_root / Path(raw).name
                if candidate.exists():
                    results.append(str(candidate.resolve()))
            for table in doc.get("tables") or []:
                table_id = str(table.get("table_id", "")).strip()
                if not table_id:
                    continue
                for ext in self.table_image_extensions:
                    candidate = self.tables_root / f"{table_id}.{ext}"
                    if candidate.exists():
                        results.append(str(candidate.resolve()))
                        break
        return list(dict.fromkeys(results))

    def collect_image_paths(
        self,
        retrieved_docs: list[dict[str, Any]],
        max_images: int,
    ) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()
        for doc in retrieved_docs:
            token = str(doc.get("title", "")).strip()
            if not token:
                continue
            for candidate in self.resolve_image_paths(token):
                if candidate in seen:
                    continue
                seen.add(candidate)
                results.append(candidate)
                if len(results) >= max_images:
                    return results
        return results


def resolve_top_docs(
    top_docs: list[dict[str, Any]],
    resolver: CorpusResolver,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    resolved_docs: list[dict[str, Any]] = []
    alias_by_id: dict[str, str] = {}
    title_by_alias: dict[str, str] = {}

    for idx, doc in enumerate(top_docs, start=1):
        token = str(doc.get("title", "")).strip()
        fallback = str(doc.get("content", "")).strip()
        resolved = resolver.resolve(token, fallback_content=fallback)
        alias = f"doc{idx}"
        if token:
            alias_by_id[token] = alias
        for doc_id in resolver.candidate_ids(token):
            alias_by_id[doc_id] = alias
        title_by_alias[alias] = resolved["title"] or token or alias
        resolved_docs.append(
            {
                "alias": alias,
                "token": token,
                "title": resolved["title"] or token or alias,
                "content": resolved["content"] or fallback,
                "caption": resolved["caption"],
                "score": doc.get("score"),
            }
        )
    return resolved_docs, alias_by_id, title_by_alias


def format_path_links_for_prompt(
    links: list[dict[str, Any]],
    alias_by_id: dict[str, str],
    title_by_alias: dict[str, str],
    max_links: int,
) -> str:
    if not links:
        return "No logical path matched the current top-k documents."

    rendered: list[str] = []
    seen = set()
    sorted_links = sorted(links, key=lambda x: safe_float(x.get("score")), reverse=True)
    effective_max_links = min(max(1, int(max_links)), VITERBI_PATH_LIMIT)
    for item in sorted_links:
        if not isinstance(item, dict):
            continue
        doc_a = str(item.get("doc_a", "")).strip()
        doc_b = str(item.get("doc_b", "")).strip()
        alias_a = alias_by_id.get(doc_a)
        alias_b = alias_by_id.get(doc_b)
        if not alias_a or not alias_b:
            continue
        dedup_key = (alias_a, alias_b, int(item.get("hop_count", 0)))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        rendered.append(
            f"{alias_a} ({title_by_alias.get(alias_a, alias_a)}) <-> "
            f"{alias_b} ({title_by_alias.get(alias_b, alias_b)}) | "
        )
        if len(rendered) >= effective_max_links:
            break

    return (
        "\n".join(rendered)
        if rendered
        else "No logical path matched the current top-k documents."
    )


def build_path_prompt_from_template(
    question: str,
    resolved_docs: list[dict[str, Any]],
    user_prompt_template: str,
    viterbi_path_string: str,
) -> str:
    values: dict[str, str] = {
        "question": question,
        "viterbi_path_string": viterbi_path_string,
    }
    for i in range(1, 6):
        if i <= len(resolved_docs):
            doc = resolved_docs[i - 1]
            values[f"title_{i}"] = str(doc["title"])
            values[f"content_{i}"] = str(doc["content"])
            values[f"caption_{i}"] = str(doc["caption"])
            values[f"caption{i}"] = str(doc["caption"])
        else:
            values[f"title_{i}"] = ""
            values[f"content_{i}"] = ""
            values[f"caption_{i}"] = ""
            values[f"caption{i}"] = ""
    return user_prompt_template.format(**values).strip()


def build_standard_prompt(
    question: str,
    resolved_docs: list[dict[str, Any]],
    doc_prompt: str,
    question_prompt: str,
    examples: list[dict[str, Any]],
) -> str:
    chunks: list[str] = []
    for idx, ex in enumerate(examples, start=1):
        if not isinstance(ex, dict):
            continue
        ex_input = str(ex.get("input", "")).strip()
        ex_resp = str(ex.get("response", "")).strip()
        if ex_input and ex_resp:
            chunks.append(f"[Few-shot Example {idx}]\n{ex_input}\n{ex_resp}")

    for doc in resolved_docs:
        chunks.append(
            doc_prompt.format(
                title=doc["title"], content=doc["content"], caption=doc["caption"]
            )
        )

    chunks.append(question_prompt.format(question=question))
    return "\n".join(chunks).strip()


def generate_path_variant(
    qwen_client: Any,
    *,
    question: str,
    resolved_docs: list[dict[str, Any]],
    image_paths: list[str],
    system_prompt: str,
    user_prompt_template: str,
    viterbi_path_string: str,
    max_tokens: int,
) -> tuple[str, str, str]:
    """Generate an answer from the path-aware template with a given path string.

    Returns ``(prompt_text, response, pred_answer)``.
    """
    prompt_text = build_path_prompt_from_template(
        question=question,
        resolved_docs=resolved_docs,
        user_prompt_template=user_prompt_template,
        viterbi_path_string=viterbi_path_string,
    )
    response = qwen_client.chat(
        prompt=prompt_text,
        images=image_paths or None,
        system_prompt=system_prompt or None,
        sampling_overrides={"max_tokens": int(max_tokens)},
    )
    return prompt_text, response, extract_answer_from_response(response)


def call_judge(
    qwen_client: Any,
    *,
    question: str,
    answer_a: str,
    answer_b: str,
    max_tokens: int,
) -> str:
    """Judge two candidate answers with the local Qwen3-VL model.

    The judge is shown only the question and the two candidate answers -- never the
    gold answer -- so selection stays blind.
    """
    user_prompt = JUDGE_USER_TEMPLATE.format(
        question=question, answer_a=answer_a, answer_b=answer_b
    )
    return qwen_client.chat(
        prompt=user_prompt,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        sampling_overrides={"max_tokens": int(max_tokens)},
    )


def parse_judge_choice(text: str) -> str | None:
    """Extract ``"A"`` or ``"B"`` from the judge response; None if unparseable."""
    match = re.search(r'"choice"\s*:\s*"([AB])"', text)
    if match:
        return match.group(1)
    match = re.search(r"\b([AB])\b", text.strip())
    if match:
        return match.group(1)
    return None


def run_final_eval(cfg: DictConfig, prediction_file: str, output_dir: str) -> str:
    """Generate final answers for a stage3 prediction file; return the output path."""
    prompt_cfg = OmegaConf.to_container(cfg.qa_prompt, resolve=True)
    if not isinstance(prompt_cfg, dict):
        raise ValueError("Invalid qa_prompt config.")
    system_prompt = str(prompt_cfg.get("system_prompt", "")).strip()
    user_prompt_template = str(prompt_cfg.get("user_prompt", "")).strip()
    doc_prompt = str(prompt_cfg.get("doc_prompt", "{title}\n{content}\n"))
    question_prompt = str(prompt_cfg.get("question_prompt", "Question: {question}\n"))
    examples = prompt_cfg.get("examples", [])
    if not isinstance(examples, list):
        examples = []
    use_user_prompt = bool(user_prompt_template)

    use_path_select = bool(cfg.final_eval.get("path_select", False))
    if use_path_select and not user_prompt_template:
        raise ValueError(
            "final_eval.path_select requires a path-aware `user_prompt` in the "
            "qa_prompt config."
        )

    with open(prediction_file, encoding="utf-8") as fin:
        prediction_rows = [json.loads(line) for line in fin if line.strip()]

    resolver = CorpusResolver(
        dataset_root=Path(cfg.dataset.root),
        data_name=cfg.dataset.data_name,
        pictures_root=Path(cfg.final_generation.pictures_root),
        tables_root=Path(cfg.final_generation.tables_image_root),
        table_image_extensions=list(cfg.final_generation.table_image_extensions),
    )

    qwen_cfg = cfg.final_generation.qwen_vl
    qwen_client = get_or_create_qwen_vl_client(
        model_name_or_path=qwen_cfg.model_name_or_path,
        tensor_parallel_size=qwen_cfg.tensor_parallel_size,
        gpu_memory_utilization=qwen_cfg.gpu_memory_utilization,
        max_num_batched_tokens=qwen_cfg.max_num_batched_tokens,
        max_model_len=qwen_cfg.max_model_len,
        allowed_local_media_path=qwen_cfg.allowed_local_media_path,
    )

    top_k = int(cfg.final_eval.get("top_k", 5))
    max_images = int(cfg.final_eval.get("max_images", 5))
    max_path_links = int(cfg.final_eval.get("max_path_links", VITERBI_PATH_LIMIT))
    max_tokens = int(cfg.final_eval.get("max_tokens", 128))
    judge_max_tokens = int(cfg.final_eval.get("judge_max_tokens", 32))

    r2_sum = 0.0
    r5_sum = 0.0
    outputs: list[dict[str, Any]] = []
    selected_path_count = 0
    selected_no_path_count = 0

    for row in tqdm(prediction_rows, desc="Final answer eval"):
        question = str(row.get("question", ""))
        retrieved_docs = row.get("retrieved_docs", [])
        if not isinstance(retrieved_docs, list):
            retrieved_docs = []
        supporting_facts = row.get("supporting_facts", [])
        if not isinstance(supporting_facts, list):
            supporting_facts = []
        r2 = compute_retrieval_recall_at_k(supporting_facts, retrieved_docs, 2)
        r5 = compute_retrieval_recall_at_k(supporting_facts, retrieved_docs, 5)
        r2_sum += r2
        r5_sum += r5
        top_docs = retrieved_docs[: max(0, top_k)]

        resolved_docs, alias_by_id, title_by_alias = resolve_top_docs(
            top_docs=top_docs, resolver=resolver
        )

        path_links: list[dict[str, Any]] = []
        row_links = row.get("path_relation_global_links")
        if isinstance(row_links, list) and row_links:
            path_links = [x for x in row_links if isinstance(x, dict)]
        else:
            for doc in top_docs:
                links = doc.get("path_relation_global_links")
                if isinstance(links, list):
                    path_links.extend([x for x in links if isinstance(x, dict)])

        image_paths = resolver.collect_image_paths(
            retrieved_docs=top_docs,
            max_images=max(0, max_images),
        )

        selection_record: dict[str, Any] = {}
        if use_path_select:
            if not path_links:
                raise ValueError(
                    "final_eval.path_select requires `path_relation_global_links` in "
                    "the prediction file, but none was found for "
                    f"id={row.get('id', '')!r}. Re-run stage3 with "
                    "path_relation.enable=true."
                )
            viterbi_path_string = format_path_links_for_prompt(
                links=path_links,
                alias_by_id=alias_by_id,
                title_by_alias=title_by_alias,
                max_links=max_path_links,
            )
            _, response_path, pred_path = generate_path_variant(
                qwen_client,
                question=question,
                resolved_docs=resolved_docs,
                image_paths=image_paths,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                viterbi_path_string=viterbi_path_string,
                max_tokens=max_tokens,
            )
            _, response_nopath, pred_nopath = generate_path_variant(
                qwen_client,
                question=question,
                resolved_docs=resolved_docs,
                image_paths=image_paths,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                viterbi_path_string="",
                max_tokens=max_tokens,
            )

            # Candidate A = with-path, B = without-path; the judge never sees gold.
            # On an unparseable judge response, fall back to the with-path variant.
            judge_raw = call_judge(
                qwen_client,
                question=question,
                answer_a=pred_path,
                answer_b=pred_nopath,
                max_tokens=judge_max_tokens,
            )
            judge_choice = parse_judge_choice(judge_raw)
            if judge_choice == "B":
                selected_variant = "no_path"
                response, pred_answer = response_nopath, pred_nopath
                selected_no_path_count += 1
            else:
                selected_variant = "path"
                response, pred_answer = response_path, pred_path
                selected_path_count += 1
            prompt_mode = "path_select"
            selection_record = {
                "selected_variant": selected_variant,
                "judge_choice": judge_choice or "fallback",
                "judge_response": judge_raw,
                "path_variant": {"pred_answer": pred_path},
                "no_path_variant": {"pred_answer": pred_nopath},
            }
        else:
            if use_user_prompt:
                viterbi_path_string = format_path_links_for_prompt(
                    links=path_links,
                    alias_by_id=alias_by_id,
                    title_by_alias=title_by_alias,
                    max_links=max_path_links,
                )
                prompt_text = build_path_prompt_from_template(
                    question=question,
                    resolved_docs=resolved_docs,
                    user_prompt_template=user_prompt_template,
                    viterbi_path_string=viterbi_path_string,
                )
                prompt_mode = "user_prompt"
            else:
                prompt_text = build_standard_prompt(
                    question=question,
                    resolved_docs=resolved_docs,
                    doc_prompt=doc_prompt,
                    question_prompt=question_prompt,
                    examples=examples,
                )
                prompt_mode = "standard_prompt"

            response = qwen_client.chat(
                prompt=prompt_text,
                images=image_paths or None,
                system_prompt=system_prompt or None,
                sampling_overrides={"max_tokens": max_tokens},
            )
            pred_answer = extract_answer_from_response(response)

        outputs.append(
            {
                "id": row.get("id", ""),
                "question": question,
                "answer": str(row.get("answer", "")),
                "answer_aliases": row.get("answer_aliases", [])
                if isinstance(row.get("answer_aliases", []), list)
                else [],
                "supporting_facts": supporting_facts,
                "retrieved_docs_topk": resolved_docs,
                "path_links_used": path_links[:max_path_links],
                "prompt_mode": prompt_mode,
                "final_context_images": image_paths,
                "response": response,
                "pred_answer": pred_answer,
                "r2": r2,
                "r5": r5,
                **selection_record,
            }
        )

    output_file = cfg.final_eval.get("output_file") or os.path.join(
        output_dir, "final_answers.jsonl"
    )
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for out_row in outputs:
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    n = max(1, len(outputs))
    aggregated: dict[str, Any] = {
        "count": len(outputs),
        "top_k": top_k,
        "max_images": max_images,
        "prompt_mode": prompt_mode if outputs else None,
        "r2": r2_sum / n,
        "r5": r5_sum / n,
        "prediction_file": prediction_file,
    }
    if use_path_select:
        aggregated["path_select"] = True
        aggregated["selected_path_count"] = selected_path_count
        aggregated["selected_no_path_count"] = selected_no_path_count

    metrics_path = output_path.with_suffix(".metrics.json")
    with metrics_path.open("w", encoding="utf-8") as fout:
        json.dump(aggregated, fout, ensure_ascii=False, indent=2)

    logger.info("Final answer eval finished: %s", output_path)
    logger.info("Final answer eval summary: %s", json.dumps(aggregated))
    return str(output_path)
