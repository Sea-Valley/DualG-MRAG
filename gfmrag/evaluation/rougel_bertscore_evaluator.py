"""ROUGE-L + BERTScore evaluator for long-form answers (e.g. WebQA)."""

from __future__ import annotations

import json
from pathlib import Path

from gfmrag.evaluation.base_evaluator import BaseEvaluator

_METRIC_KEYS = (
    "rougeL_precision",
    "rougeL_recall",
    "rougeL_f1",
    "bertscore_precision",
    "bertscore_recall",
    "bertscore_f1",
)


def _normalize_text(text: object) -> str:
    return " ".join(str(text or "").strip().split())


def _extract_prediction(row: dict) -> str:
    pred_answer = _normalize_text(row.get("pred_answer"))
    if pred_answer:
        return pred_answer
    response = str(row.get("response", "") or "").strip()
    for anchor in ("Final short answer:", "Answer:"):
        if anchor in response:
            tail = response.split(anchor)[-1].strip()
            if tail:
                return tail.splitlines()[0].strip()
    return response


def _resolve_num_layers(model_type: str, num_layers: int | None) -> int | None:
    # bert-score needs num_layers for local model dirs; infer it from config.json.
    if num_layers is not None:
        return int(num_layers)
    candidate = Path(model_type)
    if not candidate.exists():
        return None
    config_path = candidate / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"`model_type` is a local directory without config.json: {model_type}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    resolved = config.get("num_hidden_layers")
    if not isinstance(resolved, int):
        raise ValueError(
            f"Cannot infer `num_hidden_layers` from {config_path}; set `num_layers` explicitly."
        )
    return int(resolved)


class RougeLBertscoreEvaluator(BaseEvaluator):
    """Average best-of-references ROUGE-L and BERTScore over prediction rows.

    Each row must provide the gold `answer` (optional `answer_aliases` are treated
    as extra references) and either `pred_answer` or a raw `response`.
    """

    def __init__(
        self,
        prediction_file: str,
        model_type: str = "roberta-large",
        num_layers: int | None = None,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        super().__init__(prediction_file)
        try:
            from bert_score import BERTScorer
            from rouge_score import rouge_scorer
        except ImportError as exc:
            raise ImportError(
                "RougeLBertscoreEvaluator requires `rouge-score` and `bert-score`."
            ) from exc

        if device is None:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        self.bertscorer = BERTScorer(
            model_type=model_type,
            num_layers=_resolve_num_layers(model_type, num_layers),
            device=device,
            batch_size=max(1, int(batch_size)),
            lang="en",
            rescale_with_baseline=False,
        )

    def _score_one(self, prediction: str, references: list[str]) -> dict:
        refs = [r for r in (_normalize_text(x) for x in references) if r]
        pred = _normalize_text(prediction)
        if not pred or not refs:
            return dict.fromkeys(_METRIC_KEYS, 0.0)

        best_rouge = {"rougeL_precision": 0.0, "rougeL_recall": 0.0, "rougeL_f1": 0.0}
        for ref in refs:
            score = self.rouge.score(target=ref, prediction=pred)["rougeL"]
            if score.fmeasure > best_rouge["rougeL_f1"]:
                best_rouge = {
                    "rougeL_precision": float(score.precision),
                    "rougeL_recall": float(score.recall),
                    "rougeL_f1": float(score.fmeasure),
                }

        precision, recall, f1 = self.bertscorer.score([pred] * len(refs), refs)
        best_index = max(range(len(refs)), key=lambda i: float(f1[i]))
        return {
            **best_rouge,
            "bertscore_precision": float(precision[best_index]),
            "bertscore_recall": float(recall[best_index]),
            "bertscore_f1": float(f1[best_index]),
        }

    def evaluate(self) -> dict:
        metrics = dict.fromkeys(_METRIC_KEYS, 0.0)
        for row in self.data:
            aliases = row.get("answer_aliases", [])
            if not isinstance(aliases, list):
                aliases = []
            references = [str(row.get("answer", ""))] + [str(x) for x in aliases]
            sample = self._score_one(_extract_prediction(row), references)
            for key in _METRIC_KEYS:
                metrics[key] += sample[key]

        n = max(1, len(self.data))
        for key in metrics:
            metrics[key] /= n
        return metrics
