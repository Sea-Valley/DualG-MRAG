# Data Preparation

DualG-MRAG expects each dataset to live under `data/<data_name>/` with the following
raw inputs. The pipeline is dataset-agnostic: any dataset that follows these formats
can be indexed and retrieved.

## Directory layout

```text
data/<data_name>/
├── raw/
│   ├── dataset_corpus.json        # document corpus
│   ├── dataset_multimodal.json    # multimodal metadata (images/tables)
│   ├── train.json                 # QA pairs for training (optional)
│   └── test.json                  # QA pairs for evaluation
├── pictures/                      # image files referenced by basename
├── tables/                        # rendered table images (MMQA)
└── processed/                     # pipeline output
```

## Document corpus — `dataset_corpus.json`

A JSON object mapping each document title to its text content:

```json
{
  "Document Title 1": "The text content of the document...",
  "Document Title 2": "The text content of the document..."
}
```

The document title is the unique identifier used throughout the pipeline (supporting
facts, retrieval results, reasoning paths all reference titles).

For multimodal datasets, `dataset_multimodal.json` carries the per-document image and
table metadata (`images[].path_or_url`, rendered table references, captions). See
[Per-Dataset Configurations](dataset_configs.md) and the
[MMKG Stage1 Schema](mmkg_stage1_schema.md) for the multimodal fields.

## QA data — `train.json` / `test.json`

A JSON list of QA examples. Each entry has:

| Field | Required | Description |
| --- | --- | --- |
| `id` | yes | Unique question identifier. |
| `question` | yes | The question text. |
| `supporting_facts` | train: yes / test: optional | List of supporting document titles. Used as retrieval ground truth. |
| `answer` | optional | The gold answer string, used for answer evaluation. |

```json
[
  {
    "id": "q-0001",
    "question": "Which city is the capital of the country where X is located?",
    "supporting_facts": ["X", "Country Y"],
    "answer": "City Z"
  }
]
```

Any additional fields on an entry are copied through the pipeline untouched, so you
can carry dataset-specific metadata (e.g. MMQA question type, image ids) alongside
these core fields.

`train.json` provides the query–document supervision used in
[retriever training](training.md); `test.json` is the evaluation split consumed by
[stage3 inference](inference.md). Both are annotated with extracted entities during
[index construction](kg_index.md).

For the per-dataset preparation scripts that build these files from the original
MMQA / WebQA / ScienceQA releases, see [Per-Dataset Configurations](dataset_configs.md).
