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


def normalize_id(value: Any) -> str:
    return str(value).strip()


def resolve_image_path(
    base_dir: Path,
    image_dir: str,
    image_id: str,
    absolute: bool,
    default_suffix: str,
) -> str:
    image_id = str(image_id or "").strip()
    if not image_id:
        return ""
    rel_path = f"{image_id}{default_suffix}"
    if not absolute:
        return rel_path
    return str((base_dir / image_dir / rel_path).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare WebQA raw files for Stage1 (test split only)."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/WebQA"),
        help="WebQA directory containing corpus files, web1000picture/, and test.json.",
    )
    parser.add_argument(
        "--qa-file",
        type=str,
        default="test.json",
        help="QA source file name under --base-dir.",
    )
    parser.add_argument(
        "--txt-corpus-file",
        type=str,
        default="dataset_corpus_txt.json",
        help="Text corpus json file name under --base-dir.",
    )
    parser.add_argument(
        "--img-corpus-file",
        type=str,
        default="dataset_corpus_img.json",
        help="Image corpus json file name under --base-dir.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="web1000picture",
        help="Image directory name under --base-dir.",
    )
    parser.add_argument(
        "--image-suffix",
        type=str,
        default=".jpg",
        help="Image filename suffix used to build image path from image_id.",
    )
    parser.add_argument(
        "--absolute-media-path",
        action="store_true",
        default=True,
        help="Use absolute path for image files in dataset_multimodal.json.",
    )
    parser.add_argument(
        "--relative-media-path",
        dest="absolute_media_path",
        action="store_false",
        help="Keep image path as relative filename (e.g. 30087995.jpg).",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    raw_dir = base_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    txt_corpus = load_json(base_dir / args.txt_corpus_file)
    image_corpus = load_json(base_dir / args.img_corpus_file)
    qa_data = load_json(base_dir / args.qa_file)

    dataset_corpus: dict[str, str] = {}
    for doc_id, item in txt_corpus.items():
        text_id = normalize_id(doc_id)
        if not text_id:
            continue
        title = str(item.get("title", text_id))
        text = str(item.get("fact", item.get("text", "")))
        dataset_corpus[text_id] = f"{title}\n{text}".strip()

    multimodal_docs: dict[str, dict] = {}

    for doc_id, item in txt_corpus.items():
        text_id = normalize_id(doc_id)
        if not text_id:
            continue
        multimodal_docs.setdefault(
            text_id,
            {
                "doc_id": text_id,
                "title": item.get("title", text_id),
                "text": "",
                "images": [],
                "tables": [],
            },
        )
        multimodal_docs[text_id]["text"] = str(item.get("fact", item.get("text", "")))

    for doc_id, item in image_corpus.items():
        image_id = normalize_id(item.get("image_id", doc_id))
        if not image_id:
            continue
        multimodal_docs.setdefault(
            image_id,
            {
                "doc_id": image_id,
                "title": item.get("title", image_id),
                "text": "",
                "images": [],
                "tables": [],
            },
        )
        image_path = resolve_image_path(
            base_dir=base_dir,
            image_dir=args.image_dir,
            image_id=image_id,
            absolute=bool(args.absolute_media_path),
            default_suffix=args.image_suffix,
        )
        multimodal_docs[image_id]["images"].append(
            {
                "image_id": image_id,
                "path_or_url": image_path,
                "caption": item.get("caption", item.get("title", image_id)),
            }
        )

    converted_test = []
    for row in qa_data:
        supporting_text_ids = row.get("supporting_text_ids", [])
        supporting_image_ids = row.get("supporting_image_ids", [])

        combined_supporting = []
        if isinstance(supporting_text_ids, list):
            combined_supporting.extend(supporting_text_ids)
        if isinstance(supporting_image_ids, list):
            combined_supporting.extend(supporting_image_ids)

        supporting_facts = list(
            dict.fromkeys(
                normalize_id(x) for x in combined_supporting if normalize_id(x)
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
