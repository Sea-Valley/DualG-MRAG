import json
import logging
import os

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@hydra.main(
    config_path="config",
    config_name="stage1_build_micro_matcher_index",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    output_dir = HydraConfig.get().runtime.output_dir
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("Current working directory: %s", os.getcwd())
    logger.info("Output directory: %s", output_dir)

    stage1_dir = os.path.join(
        cfg.dataset.root, cfg.dataset.data_name, "processed", "stage1"
    )
    micro_path = os.path.join(stage1_dir, cfg.input.micro_kg_filename)
    if not os.path.exists(micro_path):
        raise FileNotFoundError(
            f"micro-KG file not found: {micro_path}. Run stage1_index_dataset first."
        )

    micro_triples = _read_jsonl(micro_path)
    if not micro_triples:
        raise RuntimeError(f"micro-KG file is empty: {micro_path}")

    matcher = instantiate(cfg.micro_matcher)
    matcher.build_index(micro_triples)
    logger.info(
        "Micro matcher index built successfully. triples=%s, source=%s",
        len(micro_triples),
        micro_path,
    )

    if cfg.output.write_summary:
        summary_path = os.path.join(stage1_dir, cfg.output.summary_filename)
        backend = str(cfg.micro_matcher.get("vector_backend", "milvus"))
        summary = {
            "source_micro_kg": micro_path,
            "num_micro_triples": len(micro_triples),
            "vector_backend": backend,
            "node_collection": cfg.micro_matcher.node_collection,
            "relation_collection": cfg.micro_matcher.relation_collection,
        }
        if backend == "milvus":
            summary["milvus_uri"] = cfg.micro_matcher.milvus_uri
        if backend == "faiss":
            summary["faiss_storage_dir"] = cfg.micro_matcher.faiss_storage_dir
        with open(summary_path, "w", encoding="utf-8") as fout:
            json.dump(summary, fout, indent=2, ensure_ascii=False)
        logger.info("Saved index summary to %s", summary_path)


if __name__ == "__main__":
    main()
