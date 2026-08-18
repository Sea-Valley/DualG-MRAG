"""Convert WebQA1000.json into corpus and test files.

Outputs:
1) dataset_corpus_txt.json — merged txt_posFacts + txt_negFacts, deduped by snippet_id,
   keeping only snippet_id/title/fact.
2) dataset_corpus_img.json — merged img_posFacts + img_negFacts, deduped by image_id,
   keeping only image_id/title/caption.
3) test.json — one item per question with id/question/answer plus
   supporting_text_ids (snippet_ids from txt_posFacts) and
   supporting_image_ids (image_ids from img_posFacts).
"""

import argparse
import json
import os
from typing import Any


def simplify_txt_fact(fact: dict[str, Any]) -> dict[str, Any]:
    return {
        "snippet_id": fact.get("snippet_id"),
        "title": fact.get("title"),
        "fact": fact.get("fact"),
    }


def simplify_img_fact(fact: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_id": fact.get("image_id"),
        "title": fact.get("title"),
        "caption": fact.get("caption"),
    }


def normalize_quoted_string(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    return text


def normalize_answer(answer: Any) -> Any:
    if isinstance(answer, list):
        return [normalize_quoted_string(x) for x in answer]
    return normalize_quoted_string(answer)


def build_outputs(
    data: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    corpus_txt_by_snippet_id: dict[str, dict[str, Any]] = {}
    corpus_img_by_image_id: dict[str, dict[str, Any]] = {}
    test_items: list[dict[str, Any]] = []

    for qid, item in data.items():
        txt_pos = item.get("txt_posFacts", []) or []
        txt_neg = item.get("txt_negFacts", []) or []
        img_pos = item.get("img_posFacts", []) or []
        img_neg = item.get("img_negFacts", []) or []

        for fact in txt_pos + txt_neg:
            simple_txt = simplify_txt_fact(fact)
            snippet_id = simple_txt.get("snippet_id")
            if not snippet_id:
                continue
            snippet_id = str(snippet_id)
            if snippet_id not in corpus_txt_by_snippet_id:
                corpus_txt_by_snippet_id[snippet_id] = simple_txt

        for fact in img_pos + img_neg:
            simple_img = simplify_img_fact(fact)
            image_id = simple_img.get("image_id")
            if image_id is None:
                continue
            image_id = str(image_id)
            if image_id not in corpus_img_by_image_id:
                corpus_img_by_image_id[image_id] = {
                    "image_id": fact.get("image_id"),
                    "title": fact.get("title"),
                    "caption": fact.get("caption"),
                }

        supporting_text_ids = []
        for fact in txt_pos:
            snippet_id = fact.get("snippet_id")
            if snippet_id:
                supporting_text_ids.append(str(snippet_id))

        supporting_image_ids = []
        for fact in img_pos:
            image_id = fact.get("image_id")
            if image_id is not None:
                supporting_image_ids.append(image_id)

        test_items.append(
            {
                "id": item.get("Guid", qid),
                "question": normalize_quoted_string(item.get("Q")),
                "answer": normalize_answer(item.get("A")),
                "supporting_text_ids": supporting_text_ids,
                "supporting_image_ids": supporting_image_ids,
            }
        )

    return corpus_txt_by_snippet_id, corpus_img_by_image_id, test_items


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert WebQA1000.json into dataset_corpus_txt.json, "
            "dataset_corpus_img.json and test.json."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="WebQA1000.json",
        help="Input WebQA1000.json path",
    )
    parser.add_argument(
        "--corpus-txt-output",
        type=str,
        default="dataset_corpus_txt.json",
        help="Output dataset_corpus_txt.json path",
    )
    parser.add_argument(
        "--corpus-img-output",
        type=str,
        default="dataset_corpus_img.json",
        help="Output dataset_corpus_img.json path",
    )
    parser.add_argument(
        "--test-output", type=str, default="test.json", help="Output test.json path"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    corpus_txt, corpus_img, test_items = build_outputs(data)

    with open(args.corpus_txt_output, "w", encoding="utf-8") as f:
        json.dump(corpus_txt, f, ensure_ascii=False, indent=2)

    with open(args.corpus_img_output, "w", encoding="utf-8") as f:
        json.dump(corpus_img, f, ensure_ascii=False, indent=2)

    with open(args.test_output, "w", encoding="utf-8") as f:
        json.dump(test_items, f, ensure_ascii=False, indent=2)

    print(f"input questions: {len(data)}")
    print(f"deduped text corpus: {len(corpus_txt)}")
    print(f"deduped image corpus: {len(corpus_img)}")
    print(f"written: {args.corpus_txt_output}")
    print(f"written: {args.corpus_img_output}")
    print(f"written: {args.test_output}")


if __name__ == "__main__":
    main()
