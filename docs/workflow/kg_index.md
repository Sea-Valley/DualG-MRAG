# Index Construction (Stage 1)

Stage 1 builds the dual-tier graph index from the raw dataset files described in
[Data Preparation](data_preparation.md). It constructs the macro reasoning graph, the
micro matching graph, and the cross-tier alignment index, and annotates the QA splits
with extracted entities.

Run it with:

```bash
python -m gfmrag.workflow.stage1_index_dataset dataset.root=data dataset.data_name=mmqa
```

Set `dataset.force_rebuild=true` to overwrite existing artifacts.

## Output artifacts

All outputs are written under `data/<data_name>/processed/stage1/`. The core
KG-index files are:

| File | Description |
| --- | --- |
| `kg.txt` | The macro knowledge graph, one triple per line as `subject\trelation\tobject`. |
| `document2entities.json` | Mapping `{document_title: [entities]}` linking each document to the entities it mentions. |
| `train.json` / `test.json` | The input QA splits annotated with `question_entities` and `supporting_entities` (entities extracted from the question and the supporting documents). |

DualG-MRAG additionally produces the multimodal micro-graph and cross-tier artifacts:

- `micro_kg.jsonl` — fine-grained 4-tuple micro-facts `(head, relation, tail, doc)`.
- `cross_tier_index.json` — links between macro entities and micro facts.
- `entity2micro.json` — entity → micro-fact mapping used by pattern activation.

For the full field-level schema of every stage1 artifact, see the
[MMKG Stage1 Schema](mmkg_stage1_schema.md).

## What the index is used for

- **Stage 2 (training)** reads `kg.txt` triples for KG pre-training and the annotated
  `train.json` query–document pairs for QA fine-tuning. See
  [Retriever Training](training.md).
- **Stage 3 (inference)** loads the index to score and retrieve documents. See
  [Retrieval + Inference](inference.md).

The construction pipeline is configurable per component — NER, entity linking,
OpenIE, text embedding, and document ranking. See the [Config](../config/kg_index_config.md)
reference for the available options.
