import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def extract_answers(row: dict) -> tuple[str, list[str]]:
    candidates: list[str] = []

    direct_answer = row.get("answer")
    if isinstance(direct_answer, str):
        text = direct_answer.strip()
        if text:
            candidates.append(text)
    elif isinstance(direct_answer, list):
        for item in direct_answer:
            if isinstance(item, dict):
                text = str(item.get("answer", "")).strip()
                if text:
                    candidates.append(text)
            elif isinstance(item, str):
                text = item.strip()
                if text:
                    candidates.append(text)

    if not candidates:
        answers = row.get("answers", [])
        if isinstance(answers, list):
            for item in answers:
                if isinstance(item, dict):
                    text = str(item.get("answer", "")).strip()
                    if text:
                        candidates.append(text)
                elif isinstance(item, str):
                    text = item.strip()
                    if text:
                        candidates.append(text)

    if not candidates:
        return "", []
    return candidates[0], candidates[1:]


def linearize_table(table_obj: dict) -> str:
    table = table_obj.get("table", {})
    table_name = table.get("table_name", "")
    headers = [h.get("column_name", "") for h in table.get("header", [])]
    rows = table.get("table_rows", [])

    lines = []
    if table_name:
        lines.append(f"Table: {table_name}")
    if headers:
        lines.append("Columns: " + " | ".join(headers))
    for row in rows:
        vals = []
        for cell in row:
            if isinstance(cell, dict):
                vals.append(str(cell.get("text", "")))
            else:
                vals.append(str(cell))
        lines.append("Row: " + " | ".join(vals))
    return "\n".join(lines)


def build_table_schema(table_obj: dict) -> tuple[list[str], list[dict]]:
    table = table_obj.get("table", {})
    headers = [
        h.get("column_name", f"col_{idx}")
        for idx, h in enumerate(table.get("header", []))
    ]
    rows = table.get("table_rows", [])
    dict_rows = []
    for row in rows:
        row_dict = {}
        for idx, cell in enumerate(row):
            key = headers[idx] if idx < len(headers) else f"col_{idx}"
            if isinstance(cell, dict):
                row_dict[key] = cell.get("text", "")
            else:
                row_dict[key] = str(cell)
        dict_rows.append(row_dict)
    return headers, dict_rows


def resolve_media_path(
    base_dir: Path, subdir: str, raw_path: str, absolute: bool
) -> str:
    value = str(raw_path or "").strip()
    if not value:
        return ""
    if value.startswith(("http://", "https://")):
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if not absolute:
        return value
    return str((base_dir / subdir / value).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare MMQA raw files for Stage1 (test split only)."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/MMQA"),
        help="MMQA directory containing corpus files, pictures/, tables/, and test.json.",
    )
    parser.add_argument(
        "--qa-file",
        type=str,
        default="test.json",
        help="QA source file name under --base-dir.",
    )
    parser.add_argument(
        "--absolute-media-path",
        action="store_true",
        default=True,
        help="Use absolute path for image/table files in dataset_multimodal.json.",
    )
    parser.add_argument(
        "--relative-media-path",
        dest="absolute_media_path",
        action="store_false",
        help="Keep media path relative to pictures/ or tables/.",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    txt_corpus = load_json(base_dir / "dataset_corpus_txt.json")
    table_corpus = load_json(base_dir / "dataset_corpus_table.json")
    image_corpus = load_json(base_dir / "dataset_corpus_image.json")
    qa_data = load_json(base_dir / args.qa_file)

    dataset_corpus = {}
    for doc_id, item in txt_corpus.items():
        title = item.get("title", doc_id)
        text = item.get("text", "")
        dataset_corpus[doc_id] = f"{title}\n{text}".strip()

    for doc_id, item in table_corpus.items():
        title = item.get("title", doc_id)
        table_text = linearize_table(item)
        dataset_corpus[doc_id] = f"{title}\n{table_text}".strip()

    multimodal_docs: dict[str, dict[str, Any]] = {}

    for doc_id, item in txt_corpus.items():
        multimodal_docs.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "title": item.get("title", doc_id),
                "text": "",
                "images": [],
                "tables": [],
            },
        )
        multimodal_docs[doc_id]["text"] = item.get("text", "")

    for doc_id, item in image_corpus.items():
        multimodal_docs.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "title": item.get("title", doc_id),
                "text": "",
                "images": [],
                "tables": [],
            },
        )
        image_path = resolve_media_path(
            base_dir=base_dir,
            subdir="pictures",
            raw_path=item.get("path", ""),
            absolute=bool(args.absolute_media_path),
        )
        multimodal_docs[doc_id]["images"].append(
            {
                "image_id": item.get("id", doc_id),
                "path_or_url": image_path,
                "caption": item.get("title", doc_id),
            }
        )

    for doc_id, item in table_corpus.items():
        multimodal_docs.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "title": item.get("title", doc_id),
                "text": "",
                "images": [],
                "tables": [],
            },
        )
        headers, rows = build_table_schema(item)
        table_item = {
            "table_id": item.get("id", doc_id),
            "schema": headers,
            "rows": rows,
        }
        if item.get("path"):
            table_item["path_or_url"] = resolve_media_path(
                base_dir=base_dir,
                subdir="tables",
                raw_path=item.get("path", ""),
                absolute=bool(args.absolute_media_path),
            )
        multimodal_docs[doc_id]["tables"].append(table_item)

    converted_test = []
    for row in qa_data:
        supporting_facts = list(
            dict.fromkeys(
                [
                    ctx.get("doc_id")
                    for ctx in row.get("supporting_context", [])
                    if isinstance(ctx, dict) and ctx.get("doc_id")
                ]
            )
        )
        answer, answer_aliases = extract_answers(row)
        converted_test.append(
            {
                "id": row.get("id", row.get("qid", "")),
                "question": row.get("question", ""),
                "answer": answer,
                "answer_aliases": answer_aliases,
                "supporting_facts": supporting_facts,
            }
        )

    dump_json(raw_dir / "dataset_corpus.json", dataset_corpus)
    dump_json(raw_dir / "dataset_multimodal.json", list(multimodal_docs.values()))
    dump_json(raw_dir / "test.json", converted_test)

    print(
        json.dumps(
            {
                "base_dir": str(base_dir.resolve()),
                "raw_dir": str(raw_dir.resolve()),
                "dataset_corpus_docs": len(dataset_corpus),
                "multimodal_docs": len(multimodal_docs),
                "test_size": len(converted_test),
                "qa_file": args.qa_file,
                "absolute_media_path": bool(args.absolute_media_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
