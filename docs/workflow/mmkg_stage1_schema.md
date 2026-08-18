# Stage1 MMKG Schema

This document defines the optional multimodal input and output artifacts used by Stage1.

## Raw Input (Optional)

File: `raw/dataset_multimodal.json`

Supported formats:

1. List format

```json
[
  {
    "doc_id": "doc_1",
    "title": "Movie B",
    "text": "...",
    "images": [
      {
        "image_id": "img_1",
        "path_or_url": "images/img_1.jpg",
        "caption": "Prince hugs Princess"
      }
    ],
    "tables": [
      {
        "table_id": "tb_1",
        "schema": ["person", "birth_place"],
        "rows": [
          {"person": "Obama", "birth_place": "Honolulu"}
        ]
      }
    ]
  }
]
```

2. Dict format (`doc_id` as key)

```json
{
  "doc_1": {
    "title": "Movie B",
    "text": "...",
    "images": [],
    "tables": []
  }
}
```

## Stage1 Outputs (Additional, Optional)

Under `processed/stage1/`:

- `macro_kg.txt`: macro triples only.
- `micro_kg.jsonl`: micro triples with provenance.
- `cross_tier_index.json`: `micro_triple_id -> [macro_entity, ...]`.
- `entity2micro.json`: `macro_entity -> [micro_triple_id, ...]`.
- `document2entities.json`: text/table branch `document -> entities`.
- `mm_document2entities.json`: image branch `document -> entities`.
- `mmkg_stats.json`: statistics for offline quality checks.

`kg.txt` remains unchanged. `document2entities.json` keeps the legacy file name but now
stores only text/table entities to support dual-branch retrieval.

Document key scope constraints:

- `document2entities.json` keys must be exactly the document IDs in `raw/dataset_corpus.json`.
- `mm_document2entities.json` keys must be exactly
  `doc_ids(raw/dataset_multimodal.json) - doc_ids(raw/dataset_corpus.json)`.
- The key sets of the two files must be disjoint.

## Dual-Branch Retrieval Notes

- Stage2 builds two inverted indices:
  - `ent2doc_text.pt` from `document2entities.json`
  - `ent2doc_image.pt` from `mm_document2entities.json`
- Online retrieval computes text/image scores independently, applies min-max normalization
  per branch, then merges by normalized score for final top-k ranking.
