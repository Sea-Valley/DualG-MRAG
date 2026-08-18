# Unsupervised pre-training of the GFM retriever on the constructed KG-index.
DATA_ROOT="data"
N_GPU=4
N_EPOCH=1
BATCH_PER_EPOCH=30000
BATCH_SIZE=4
torchrun --nproc-per-node=${N_GPU} -m gfmrag.workflow.stage2_kg_pretrain \
    datasets.train_names=[mmqa] \
    datasets.cfgs.root=${DATA_ROOT} \
    train.fast_test=5000 \
    train.num_epoch=${N_EPOCH} \
    train.batch_per_epoch=${BATCH_PER_EPOCH} \
    train.batch_size=${BATCH_SIZE}
