# Single-step QA inference on the test set (MMQA / WebQA).
N_GPU=4
DATA_ROOT="data"
DATA_NAME="mmqa" # mmqa or webqa
LLM="gpt-4o-mini"
DOC_TOP_K=5
N_THREAD=10
QA_EVALUATOR="em_f1"
if [ "${DATA_NAME}" = "webqa" ]; then
    QA_EVALUATOR="rougel_bertscore"
fi
torchrun --nproc_per_node=${N_GPU} -m gfmrag.workflow.stage3_qa_inference \
    dataset.root=${DATA_ROOT} \
    qa_prompt=${DATA_NAME} \
    qa_evaluator=${QA_EVALUATOR} \
    llm.model_name_or_path=${LLM} \
    test.n_threads=${N_THREAD} \
    test.top_k=${DOC_TOP_K} \
    dataset.data_name=${DATA_NAME}_test
