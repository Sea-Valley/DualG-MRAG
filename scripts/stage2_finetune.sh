# Supervised fine-tuning of the GFM retriever on QA pairs.
DATA_ROOT="data"
N_GPU=4
N_EPOCH=15
torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.stage2_qa_finetune \
    datasets.train_names=[mmqa_train_example] \
    datasets.cfgs.root=${DATA_ROOT} \
    train.num_epoch=${N_EPOCH}

# Retrieval evaluation only (no training):
# torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.stage2_qa_finetune \
#     train.checkpoint=<path-to-checkpoint> \
#     datasets.cfgs.root=${DATA_ROOT} \
#     datasets.train_names=[] \
#     train.num_epoch=0
