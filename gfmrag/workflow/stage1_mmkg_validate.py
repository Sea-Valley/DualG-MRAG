import json
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _read_json(path: str, default: dict | list) -> dict | list:
    if not os.path.exists(path):
        return default
    with open(path) as fin:
        return json.load(fin)


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@hydra.main(config_path="config", config_name="stage1_mmkg_validate", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    stage1_dir = os.path.join(
        cfg.dataset.root, cfg.dataset.data_name, "processed", "stage1"
    )
    macro_path = os.path.join(stage1_dir, "macro_kg.txt")
    micro_path = os.path.join(stage1_dir, "micro_kg.jsonl")
    cross_path = os.path.join(stage1_dir, "cross_tier_index.json")
    stats_path = os.path.join(stage1_dir, "mmkg_stats.json")

    macro_triples = 0
    if os.path.exists(macro_path):
        with open(macro_path) as fin:
            macro_triples = sum(1 for line in fin if line.strip())

    micro_rows = _read_jsonl(micro_path)
    cross_tier = _read_json(cross_path, {})
    stats = _read_json(stats_path, {})
    if not isinstance(cross_tier, dict):
        cross_tier = {}
    if not isinstance(stats, dict):
        stats = {}

    micro_total = len(micro_rows)
    cross_aligned = sum(1 for entities in cross_tier.values() if entities)
    coverage = cross_aligned / micro_total if micro_total else 0.0
    docs = {row.get("doc_id") for row in micro_rows if row.get("doc_id")}
    empty_micro_ratio = 1.0 if macro_triples > 0 and micro_total == 0 else 0.0
    unresolved_ratio = 1 - coverage if micro_total else 0.0
    duplicate_ratio = 0.0
    if micro_total > 0:
        unique_edges = {
            (
                row.get("doc_id"),
                row.get("head"),
                row.get("relation"),
                row.get("tail"),
            )
            for row in micro_rows
        }
        duplicate_ratio = 1 - (len(unique_edges) / micro_total)

    logger.info("MMKG validation summary")
    logger.info("macro_triples=%s", macro_triples)
    logger.info("micro_triples=%s", micro_total)
    logger.info("micro_docs=%s", len(docs))
    logger.info("cross_tier_coverage=%.4f", coverage)
    logger.info("empty_micro_ratio=%.4f", empty_micro_ratio)
    logger.info("unresolved_micro_ratio=%.4f", unresolved_ratio)
    logger.info("duplicate_micro_ratio=%.4f", duplicate_ratio)
    if stats:
        logger.info("cached_mmkg_stats=%s", stats)

    if coverage < cfg.thresholds.cross_tier_coverage_warn:
        logger.warning(
            "Cross-tier coverage %.4f is below warn threshold %.4f",
            coverage,
            cfg.thresholds.cross_tier_coverage_warn,
        )
    if unresolved_ratio > cfg.thresholds.unresolved_micro_ratio_warn:
        logger.warning(
            "Unresolved micro ratio %.4f exceeds warn threshold %.4f",
            unresolved_ratio,
            cfg.thresholds.unresolved_micro_ratio_warn,
        )
    if duplicate_ratio > cfg.thresholds.duplicate_micro_ratio_warn:
        logger.warning(
            "Duplicate micro ratio %.4f exceeds warn threshold %.4f",
            duplicate_ratio,
            cfg.thresholds.duplicate_micro_ratio_warn,
        )


if __name__ == "__main__":
    main()
