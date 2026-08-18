from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from gfmrag.llms.qwenvl_vllm import Qwen3VLVLLM, get_or_create_qwen_vl_client

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Per-dataset defaults used when visual_budget.prompt_config/examples_file are
# null. Keys are matched case-insensitively against dataset.data_name.
DATASET_VISUAL_BUDGET_DEFAULTS: dict[str, dict[str, str]] = {
    "mmqa": {
        "prompt_config": "gfmrag/workflow/config/visual_budget/mmqa.yaml",
        "examples_file": "assets/visual_budget/mmqa_train_supporting_examples.json",
    },
    "webqa": {
        "prompt_config": "gfmrag/workflow/config/visual_budget/webqa.yaml",
        "examples_file": "assets/visual_budget/webqa_train_supporting_examples.json",
    },
}


def _resolve_dataset_key(data_name: str) -> str:
    name = data_name.strip().lower()
    for key in DATASET_VISUAL_BUDGET_DEFAULTS:
        if key in name:
            return key
    raise ValueError(
        f"Cannot auto-select visual budget config for dataset {data_name!r}. "
        "Set final_generation.visual_budget.prompt_config and "
        "final_generation.visual_budget.examples_file explicitly."
    )


def resolve_visual_budget_paths(
    prompt_config: str | None, examples_file: str | None, data_name: str
) -> tuple[Path, Path]:
    """Fill in null prompt_config/examples_file from the per-dataset defaults."""
    if prompt_config and examples_file:
        return Path(prompt_config), Path(examples_file)
    defaults = DATASET_VISUAL_BUDGET_DEFAULTS[_resolve_dataset_key(data_name)]
    resolved_prompt = (
        Path(prompt_config) if prompt_config else _REPO_ROOT / defaults["prompt_config"]
    )
    resolved_examples = (
        Path(examples_file) if examples_file else _REPO_ROOT / defaults["examples_file"]
    )
    return resolved_prompt, resolved_examples


def _extract_qid(obj: dict[str, Any]) -> str:
    return str(obj.get("qid", obj.get("id", ""))).strip()


def _load_completed_keys(cache_file: Path) -> set[tuple[int, str]]:
    done: set[tuple[int, str]] = set()
    if not cache_file.exists():
        return done
    with cache_file.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = obj.get("question_index")
            qid = obj.get("qid")
            if isinstance(idx, int) and isinstance(qid, str):
                done.add((idx, qid))
    return done


def _infer_label_from_supporting_parts(
    parts: list[str], category_labels: dict[str, str]
) -> str:
    parts_norm = [str(x).strip().lower() for x in parts]
    has_image = any(x == "image" for x in parts_norm)
    has_non_image = any(x != "image" for x in parts_norm)
    if has_image and not has_non_image:
        return category_labels["image_only"]
    if has_image and has_non_image:
        return category_labels.get("image_and_others", category_labels["image_only"])
    return category_labels["no_image"]


def _parse_model_label(text: str, allowed_labels: set[str]) -> str:
    candidate = text.strip()
    if candidate in allowed_labels:
        return candidate
    for ch in candidate:
        if ch in allowed_labels:
            return ch
    raise ValueError(f"Model output does not contain valid label: {text!r}")


def _build_fewshot_prompt(
    examples_json: dict[str, Any],
    prompt_prefix: str,
    category_labels: dict[str, str],
    category_order: list[str],
    few_shot: dict[str, int],
) -> str:
    categories = examples_json.get("categories", {})
    if not isinstance(categories, dict):
        raise ValueError("Invalid examples JSON: missing 'categories'")

    lines: list[str] = [prompt_prefix.strip(), "", "Here are examples:"]
    for cat in category_order:
        items = categories.get(cat, [])
        if not isinstance(items, list):
            continue
        for item in items[: max(0, int(few_shot.get(cat, 0)))]:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            parts = item.get("supporting_parts", [])
            label = _infer_label_from_supporting_parts(
                parts if isinstance(parts, list) else [], category_labels
            )
            lines.append(f"Question: {question}")
            lines.append(f"Label: {label}")
            lines.append("")

    lines.append("Now classify the new question.")
    return "\n".join(lines).strip()


class VisualBudgetClassifier:
    """Few-shot classifier for per-question image budgets, scored with the
    same local Qwen3-VL vLLM engine as final answer generation."""

    def __init__(
        self,
        prompt_config_path: str | Path,
        examples_file: str | Path,
        qwen_client: Qwen3VLVLLM,
        batch_size: int = 8,
        max_tokens: int = 8,
    ) -> None:
        prompt_cfg = OmegaConf.to_container(
            OmegaConf.load(str(prompt_config_path)), resolve=True
        )
        if not isinstance(prompt_cfg, dict):
            raise ValueError(
                f"Invalid visual budget prompt config: {prompt_config_path}"
            )
        self.allowed_labels = {str(x) for x in prompt_cfg["allowed_labels"]}
        self.category_labels = {
            str(k): str(v) for k, v in prompt_cfg["category_labels"].items()
        }
        self.category_order = [str(x) for x in prompt_cfg["category_order"]]
        self.few_shot = {
            str(k): int(v) for k, v in prompt_cfg.get("few_shot", {}).items()
        }
        self.prompt_prefix = str(prompt_cfg["prompt_prefix"])

        with open(examples_file, encoding="utf-8") as fin:
            examples_json = json.load(fin)
        self.prompt = _build_fewshot_prompt(
            examples_json=examples_json,
            prompt_prefix=self.prompt_prefix,
            category_labels=self.category_labels,
            category_order=self.category_order,
            few_shot=self.few_shot,
        )

        self.qwen_client = qwen_client
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.batch_size = int(batch_size)
        self.max_tokens = int(max_tokens)

    @classmethod
    def from_config(
        cls, cfg: DictConfig, qwen_client: Qwen3VLVLLM, data_name: str
    ) -> VisualBudgetClassifier:
        prompt_config_path, examples_file = resolve_visual_budget_paths(
            cfg.get("prompt_config", None),
            cfg.get("examples_file", None),
            data_name,
        )
        return cls(
            prompt_config_path=prompt_config_path,
            examples_file=examples_file,
            qwen_client=qwen_client,
            batch_size=int(cfg.get("batch_size", 8)),
            max_tokens=int(cfg.get("max_tokens", 8)),
        )

    def _score_chunk(
        self, chunk: list[tuple[int, dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        requests = [
            {
                "prompt": f"Question: {str(obj.get('question', '')).strip()}\nLabel:",
                "system_prompt": self.prompt,
            }
            for _, obj in chunk
        ]
        raws = self.qwen_client.batch_chat(
            requests,
            sampling_overrides={"temperature": 0.0, "max_tokens": self.max_tokens},
        )
        rows: list[dict[str, Any]] = []
        for (index, obj), raw in zip(chunk, raws, strict=True):
            score = _parse_model_label(raw, self.allowed_labels)
            rows.append(
                {
                    "question_index": index,
                    "qid": _extract_qid(obj),
                    "query": str(obj.get("question", "")).strip(),
                    "score": score,
                    "model_raw_output": raw,
                    "model": self.qwen_client.engine_config.model_name_or_path,
                }
            )
        return rows

    def score_questions(
        self, test_items: list[dict[str, Any]], cache_file: str | Path
    ) -> dict[str, int]:
        """Score every question, appending to ``cache_file`` (resumable), and
        return a ``{qid: int_label}`` mapping covering the whole test set."""
        cache_path = Path(cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        completed = _load_completed_keys(cache_path)

        scores: dict[str, int] = {}
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    qid = str(obj.get("qid", "")).strip()
                    if qid:
                        try:
                            scores[qid] = int(float(obj.get("score", 0)))
                        except (TypeError, ValueError):
                            continue

        pending = [
            (index, obj)
            for index, obj in enumerate(test_items, start=1)
            if (index, _extract_qid(obj)) not in completed
        ]
        logger.info(
            "Visual budget: %s questions total, %s cached, %s to score.",
            len(test_items),
            len(test_items) - len(pending),
            len(pending),
        )

        written = 0
        with cache_path.open("a", encoding="utf-8") as fout:
            for start in range(0, len(pending), self.batch_size):
                chunk = pending[start : start + self.batch_size]
                for row in self._score_chunk(chunk):
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout.flush()
                    os.fsync(fout.fileno())
                    scores[row["qid"]] = int(float(row["score"]))
                    written += 1

        if written:
            logger.info(
                "Visual budget: scored %s new questions -> %s", written, cache_path
            )
        return scores


def run_visual_budget_if_enabled(
    cfg: DictConfig,
    output_dir: str,
    test_items: list[dict[str, Any]],
    data_name: str,
) -> str | None:
    """Run the visual-budget classification when enabled in
    ``final_generation.visual_budget`` and return the cache score file path;
    return ``None`` when disabled."""
    vb_cfg = cfg.get("visual_budget", None)
    if vb_cfg is None or not bool(vb_cfg.get("enable", False)):
        return None
    cache_file = str(vb_cfg.get("cache_file", "") or "") or os.path.join(
        output_dir, "visual_budget_scores.jsonl"
    )
    qwen_cfg = cfg.qwen_vl
    qwen_client = get_or_create_qwen_vl_client(
        model_name_or_path=qwen_cfg.model_name_or_path,
        tensor_parallel_size=qwen_cfg.tensor_parallel_size,
        gpu_memory_utilization=qwen_cfg.gpu_memory_utilization,
        max_num_batched_tokens=qwen_cfg.max_num_batched_tokens,
        max_model_len=qwen_cfg.max_model_len,
        allowed_local_media_path=qwen_cfg.get("allowed_local_media_path", None),
    )
    classifier = VisualBudgetClassifier.from_config(vb_cfg, qwen_client, data_name)
    classifier.score_questions(test_items, cache_file)
    return cache_file
