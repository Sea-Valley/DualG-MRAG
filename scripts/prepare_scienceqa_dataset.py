from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from gfmrag.workflow.scienceqa_utils import (
    answer_index_to_letter,
    build_scienceqa_doc,
    collect_question_images,
    dump_json,
    load_json,
)


def _load_problems(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict problems file: {path}")
    return {str(pid): item for pid, item in payload.items() if isinstance(item, dict)}


def _load_train_qwen_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict):
        return {}
    mapping: dict[str, str] = {}
    for pid, item in payload.items():
        if not isinstance(item, dict):
            continue
        caption = str(item.get("qwen3vl_caption", "")).strip()
        if caption:
            mapping[str(pid)] = caption
    return mapping


def _load_caption_map(path: Path) -> dict[str, str]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        return {}
    source = payload.get("captions", payload)
    if not isinstance(source, dict):
        return {}
    captions: dict[str, str] = {}
    for pid, caption in source.items():
        text = str(caption).strip()
        if text:
            captions[str(pid)] = text
    return captions


def _build_raw_sample(
    pid: str,
    split: str,
    problem: dict[str, Any],
    image_root: Path,
    caption_map: dict[str, str],
    qwen_map: dict[str, str],
) -> dict[str, Any]:
    choices = [str(choice).strip() for choice in problem.get("choices", [])]
    answer = int(problem.get("answer", -1))
    question_images = collect_question_images(image_root, split, pid)
    main_caption = caption_map.get(pid, "")
    image_captions = [main_caption] if main_caption else []
    return {
        "id": pid,
        "question": str(problem.get("question", "")).strip(),
        "choices": choices,
        "answer": answer,
        "answer_letter": answer_index_to_letter(answer),
        "supporting_facts": [],
        "question_images": question_images,
        "question_image_captions": image_captions,
        "question_caption": main_caption,
        "has_image": bool(question_images),
        "hint": str(problem.get("hint", "")).strip(),
        "subject": str(problem.get("subject", "")).strip(),
        "topic": str(problem.get("topic", "")).strip(),
        "category": str(problem.get("category", "")).strip(),
        "skill": str(problem.get("skill", "")).strip(),
        "split": split,
        "qwen3vl_caption": qwen_map.get(pid, ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare ScienceQA raw files for gfmrag macro/micro evaluation."
    )
    parser.add_argument(
        "--scienceqa-dir",
        type=Path,
        required=True,
        help="ScienceQA root directory that contains problems.json, train.json, captions.json and image_root/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Dataset root directory used by gfmrag, e.g. ./data",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default="scienceqa_rag",
        help="Output dataset name under output-root.",
    )
    args = parser.parse_args()

    scienceqa_dir = args.scienceqa_dir
    output_dir = args.output_root / args.data_name
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    problems = _load_problems(scienceqa_dir / "problems.json")
    qwen_map = _load_train_qwen_map(scienceqa_dir / "train.json")
    caption_map = _load_caption_map(scienceqa_dir / "captions.json")
    image_root = scienceqa_dir / "image_root"

    dataset_corpus: dict[str, str] = {}
    dataset_multimodal: dict[str, dict[str, Any]] = {}
    dataset_corpus_image: dict[str, dict[str, Any]] = {}
    raw_train: list[dict[str, Any]] = []
    raw_test: list[dict[str, Any]] = []
    question_images_map: dict[str, dict[str, list[dict[str, Any]]]] = {
        "train": {},
        "test": {},
        "val": {},
    }

    for pid, problem in problems.items():
        split = str(problem.get("split", "")).strip()
        if split not in question_images_map:
            question_images_map[split] = {}
        raw_sample = _build_raw_sample(
            pid=pid,
            split=split,
            problem=problem,
            image_root=image_root,
            caption_map=caption_map,
            qwen_map=qwen_map,
        )
        question_images_map[split][pid] = raw_sample["question_images"]

        if split == "train":
            raw_train.append(raw_sample)
            doc_text = build_scienceqa_doc(problem)
            dataset_corpus[pid] = doc_text
            short_caption = caption_map.get(pid, "")
            qwen_caption = qwen_map.get(pid, "")
            image_item = {
                "image_id": pid,
                "path": f"train/{pid}/image.png",
                "title": short_caption or f"ScienceQA image {pid}",
                "caption": short_caption,
                "qwen3vl_caption": qwen_caption,
            }
            if qwen_caption:
                dataset_multimodal[pid] = {
                    "doc_id": pid,
                    "title": pid,
                    "text": doc_text,
                    "images": [image_item],
                    "tables": [],
                }
                dataset_corpus_image[pid] = {
                    "title": short_caption or f"ScienceQA image {pid}",
                    "path": image_item["path"],
                    "caption": short_caption,
                    "qwen3vl_caption": qwen_caption,
                }
            else:
                dataset_multimodal[pid] = {
                    "doc_id": pid,
                    "title": pid,
                    "text": doc_text,
                    "images": [],
                    "tables": [],
                }
        elif split == "test":
            raw_test.append(raw_sample)

    dump_json(raw_dir / "dataset_corpus.json", dataset_corpus)
    dump_json(raw_dir / "dataset_multimodal.json", dataset_multimodal)
    dump_json(raw_dir / "dataset_corpus_image.json", dataset_corpus_image)
    dump_json(raw_dir / "train.json", raw_train)
    dump_json(raw_dir / "test.json", raw_test)
    dump_json(raw_dir / "scienceqa_question_images.json", question_images_map)

    summary = {
        "data_name": args.data_name,
        "scienceqa_dir": str(scienceqa_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "num_train_docs": len(dataset_corpus),
        "num_train_samples": len(raw_train),
        "num_test_samples": len(raw_test),
        "num_micro_image_docs": len(dataset_corpus_image),
    }
    dump_json(raw_dir / "scienceqa_prepare_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
