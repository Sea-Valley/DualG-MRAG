from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ANSWER_LETTERS = "ABCDE"


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fin:
        return json.load(fin)


def dump_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2, ensure_ascii=False)


def dump_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def answer_index_to_letter(answer_index: int) -> str:
    if 0 <= int(answer_index) < len(ANSWER_LETTERS):
        return ANSWER_LETTERS[int(answer_index)]
    return ""


def parse_answer_letter(text: str) -> str:
    content = str(text or "").strip().upper()
    if not content:
        return ""
    exact = re.fullmatch(r"\(?([A-E])\)?", content)
    if exact:
        return exact.group(1)
    match = re.search(r"\b([A-E])\b", content)
    if match:
        return match.group(1)
    match = re.search(r"ANSWER\s*[:：]\s*\(?([A-E])\)?", content)
    if match:
        return match.group(1)
    return ""


def render_choices(choices: list[str]) -> str:
    lines = []
    for idx, choice in enumerate(choices):
        lines.append(f"({answer_index_to_letter(idx)}) {choice}")
    return "\n".join(lines)


def render_question_with_choices(question: str, choices: list[str]) -> str:
    choices_block = render_choices(choices)
    if choices_block:
        return f"{question}\nChoices:\n{choices_block}"
    return question


def build_scienceqa_doc(problem: dict[str, Any]) -> str:
    question = str(problem.get("question", "")).strip()
    choices = [str(choice).strip() for choice in problem.get("choices", [])]
    answer_idx = int(problem.get("answer", -1))
    answer_letter = answer_index_to_letter(answer_idx)
    answer_text = choices[answer_idx] if 0 <= answer_idx < len(choices) else ""
    lecture = str(problem.get("lecture", "")).strip()
    solution = str(problem.get("solution", "")).strip()

    parts = [f"Question: {question}"]
    if choices:
        parts.append("Choices:")
        parts.append(render_choices(choices))
    if answer_text:
        if answer_letter:
            parts.append(f"Answer: ({answer_letter}) {answer_text}")
        else:
            parts.append(f"Answer: {answer_text}")
    if lecture:
        parts.append(f"Lecture: {lecture}")
    elif solution:
        parts.append(f"Solution: {solution}")
    return "\n".join(parts).strip()


def collect_question_images(
    image_root: str | Path, split: str, pid: str
) -> list[dict[str, Any]]:
    sample_dir = Path(image_root) / split / str(pid)
    if not sample_dir.exists():
        return []
    image_files = sorted(
        [
            path
            for path in sample_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}
        ],
        key=lambda path: (path.name != "image.png", path.name),
    )
    items: list[dict[str, Any]] = []
    for path in image_files:
        items.append(
            {
                "filename": path.name,
                "relative_path": str(Path(split) / str(pid) / path.name).replace(
                    "\\", "/"
                ),
                "absolute_path": str(path.resolve()),
                "is_main_image": path.name == "image.png",
            }
        )
    return items


def normalize_caption_list(raw_value: Any) -> list[str]:
    if isinstance(raw_value, list):
        values = [str(item).strip() for item in raw_value if str(item).strip()]
        return list(dict.fromkeys(values))
    text = str(raw_value or "").strip()
    return [text] if text else []
