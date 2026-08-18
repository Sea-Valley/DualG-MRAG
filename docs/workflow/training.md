# Retriever Training (Stage 2)

Stage 2 trains the GNN retriever over the index built in
[Stage 1](kg_index.md). There are two training modes, both launched with `torchrun`
for distributed (multi-GPU / multi-node) training.

## KG pre-training (unsupervised)

Pre-trains the retriever on the knowledge-graph triples from `kg.txt`, without any QA
supervision:

```bash
torchrun --nproc_per_node=4 -m gfmrag.workflow.stage2_kg_pretrain
```

For multi-node training, add the usual torchrun rendezvous arguments:

```bash
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=<host> --master_port=<port> \
    -m gfmrag.workflow.stage2_kg_pretrain
```

Helper script: `scripts/stage2_pretrain.sh`.

## QA fine-tuning (supervised)

Fine-tunes the retriever on the query–document pairs from the annotated `train.json`:

```bash
torchrun --nproc_per_node=4 -m gfmrag.workflow.stage2_qa_finetune
```

Helper script: `scripts/stage2_finetune.sh`.

See the [pre-training](../config/gfmrag_pretrain_config.md) and
[fine-tuning](../config/gfmrag_finetune_config.md) configuration references for the
available options.
