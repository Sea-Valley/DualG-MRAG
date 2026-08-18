# Build the KG index (macro + micro + cross-tier) for each dataset.
# Expects data/<DATA_NAME>/raw/{dataset_corpus.json, dataset_multimodal.json, test.json, ...}
N_GPU=1
DATA_ROOT="data"
DATA_NAME_LIST="mmqa webqa scienceqa"
for DATA_NAME in ${DATA_NAME_LIST}; do
   python -m gfmrag.workflow.stage1_index_dataset \
   dataset.root=${DATA_ROOT} \
   dataset.data_name=${DATA_NAME}
done
