# Per-Dataset Configurations

DualG-MRAG ships three dataset configurations: **MMQA**, **WebQA**, and **ScienceQA**.
This page summarizes where they differ. All paths are relative to the repository root.

## Data preparation

| Dataset | Raw source | Preparation script | Output (`data/<name>/raw/`) |
| --- | --- | --- | --- |
| MMQA | MultimodalQA camera-ready release | `scripts/sample_mmqa1000.py` (optional subset), `scripts/prepare_mmqa_stage1.py`, `scripts/render_tables_to_images.py` | `dataset_corpus.json`, `dataset_multimodal.json`, `test.json` |
| WebQA | WebQA train_val JSON | `scripts/sample_webqa1000.py`, `scripts/build_webqa_txt_dataset.py`, `scripts/prepare_webqa_stage1.py` | same layout |
| ScienceQA | ScienceQA problems/pid_splits | `scripts/prepare_scienceqa_dataset.py` | same layout |

## Prompts (`gfmrag/workflow/config/qa_prompt/`)

| | MMQA (`mmqa.yaml`) | WebQA (`webqa.yaml`) | ScienceQA (`scienceqa.yaml`) |
| --- | --- | --- | --- |
| Answer style | Minimal span (EM-scored) | Complete sentence | Single option letter A–E |
| `doc_prompt` | `{title}\n{content}\n` | `{title}\n{caption}\n{content}\n` | `{title}\n{content}\n` |
| Reasoning path slot | `{viterbi_path_string}` at the top of `user_prompt` | same | n/a (macro top-1 document instead) |

The `{viterbi_path_string}` slot is filled with the decoded cross-document reasoning
path, formatted as `docA (title) <-> docB (title) |`. See `gfmrag/workflow/qa_final_eval.py`
(`format_path_links_for_prompt`, `build_path_prompt_from_template`).

## Evaluators

- MMQA: `qa_evaluator=em_f1` (`gfmrag.evaluation.em_f1_evaluator.EMF1Evaluator`,
  EM/F1/precision/recall).
- WebQA: `qa_evaluator=rougel_bertscore`
  (`gfmrag.evaluation.rougel_bertscore_evaluator.RougeLBertscoreEvaluator`,
  best-of-references ROUGE-L + BERTScore).
- ScienceQA: two commands —
  1. `python -m gfmrag.workflow.stage3_qa_inference dataset.data_name=scienceqa test.scienceqa_export_macro_only=true`
  2. `python scripts/run_scienceqa_qwen3vl_eval.py evaluate --macro-jsonl <macro_export.jsonl> --output-dir <out_dir>`
