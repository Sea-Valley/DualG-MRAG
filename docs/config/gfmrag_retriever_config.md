# Retriever Configuration

An example configuration file for the stage3 retriever is shown below:

!!! example

    ```yaml title="gfmrag/workflow/config/stage3_qa_inference.yaml"
    --8<-- "gfmrag/workflow/config/stage3_qa_inference.yaml"
    ```

## General Configuration

| Parameter | Options |              Note               |
| :-------: | :-----: | :-----------------------------: |
| `run.dir` |  None   | The output directory of the log |

## Defaults

| Parameter | Options | Note |
| :-------: | :-----: | :--- |
| `doc_ranker` | None | The config of the [doc_ranker](doc_ranker_config.md) |
| `qa_evaluator` | None | The config of the [qa_evaluator][gfmrag.evaluation] |
| `qa_prompt` | None | The config of the [PromptBuilder][gfmrag.prompt_builder] |
| `micro_matcher` | None | The config of the micro-fact matcher (SimGRAG-style) |
| `text_emb_model` | None | The config of the [text embedding model][gfmrag.text_emb_models] |


## Dataset

|  Parameter  | Options |          Note           |
| :---------: | :-----: | :---------------------: |
|   `root`    |  None   | The data root directory |
| `data_name` |  None   |      The data name      |


## LLM

|       Parameter       | Options |                           Note                           |
| :-------------------: | :-----: | :------------------------------------------------------: |
|      `_target_`       |  None   |         The [language model][gfmrag.llms] to use         |
| `model_name_or_path`  |  None   |                  The model name or path                  |
| Additional parameters |  None   | Parameters to initialize a [language model][gfmrag.llms] |

Please refer to the [LLMs][gfmrag.llms] page for more details.

## Graph Retriever

|       Parameter        |    Options     |                                Note                                |
| :--------------------: | :------------: | :----------------------------------------------------------------: |
|       `_target_`       |      None      |        The [graph retriever][gfmrag.GFMRetriever] to use        |
|      `model_path`      |      None      |          Checkpoint path of the pre-trained GFM-RAG model          |
|      `doc_ranker`      |      None      |          The [document ranker][gfmrag.doc_rankers] to use          |
|      `ner_model`       |      None      |      The [NER model][gfmrag.kg_construction.ner_model] to use      |
|       `el_model`       |      None      | The [EL model][gfmrag.kg_construction.entity_linking_model] to use |
| `init_entities_weight` | `True`,`False` |             Whether to initialize the entities weight              |


## Test

| Parameter | Options | Note |
| :-------: | :-----: | :--- |
| `retrieval_batch_size` | None | Batch size for retrieval scoring |
| `top_k` | None | Number of documents to retrieve |
| `retrieval_only` | `True`,`False` | Only run retrieval (combine with `final_eval.enable` for local answer generation) |
| `retrieved_result_path` | None | Optional path to a precomputed retrieval result |
| `prediction_result_path` | None | Optional output path for `prediction.jsonl` |
| `init_entities_weight` | `True`,`False` | Whether to initialize the entities weight |
| `scienceqa_export_macro_only` | `True`,`False` | Export the macro top-k ranking only (ScienceQA flow) |
| `scienceqa_macro_top_k` | None | Number of macro documents exported per question |
