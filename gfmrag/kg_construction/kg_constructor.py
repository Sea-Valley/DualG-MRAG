import hashlib
import json
import logging
import os
import re
import shutil
from abc import ABC, abstractmethod
from multiprocessing.dummy import Pool as ThreadPool
from typing import Any

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from gfmrag.kg_construction.utils import KG_DELIMITER, processing_phrases

from .entity_linking_model import BaseELModel
from .multimodal_extractor import BaseMultimodalExtractor
from .openie_model.base_model import BaseOPENIEModel

logger = logging.getLogger(__name__)


class BaseKGConstructor(ABC):
    """
    Abstract base class for knowledge graph construction.

    This class defines the interface for constructing knowledge graphs from datasets.
    Subclasses must implement create_kg() and get_document2entities() methods.

    Attributes:
        None

    Methods:
        create_kg: Creates a knowledge graph from the specified dataset.

        get_document2entities: Get mapping between documents and their associated entities.
    """

    @abstractmethod
    def create_kg(self, data_root: str, data_name: str) -> list[tuple[str, str, str]]:
        """
        Create a knowledge graph from the dataset

        Args:
            data_root (str): path to the dataset
            data_name (str): name of the dataset

        Returns:
            list[tuple[str, str, str]]: list of triples
        """
        pass

    @abstractmethod
    def get_document2entities(self, data_root: str, data_name: str) -> dict:
        """
        Get the document to entities mapping from the dataset

        Args:
            data_root (str): path to the dataset
            data_name (str): name of the dataset

        Returns:
            dict: document to entities mapping
        """
        pass


class KGConstructor(BaseKGConstructor):
    """A class for constructing Knowledge Graphs (KG) from text data using Open Information Extraction and Entity Linking.


    Args:
        open_ie_model (BaseOPENIEModel): Model for performing Open Information Extraction
        el_model (BaseELModel): Model for Entity Linking
        root (str, optional): Root directory for storing temporary files. Defaults to "tmp/kg_construction".
        num_processes (int, optional): Number of processes to use for parallel processing. Defaults to 1.
        cosine_sim_edges (bool, optional): Whether to add edges based on cosine similarity between entities. Defaults to True.
        threshold (float, optional): Similarity threshold for adding edges between similar entities. Defaults to 0.8.
        max_sim_neighbors (int, optional): Maximum number of similar neighbors to consider per entity. Defaults to 100.
        add_title (bool, optional): Whether to prepend document titles to passages. Defaults to True.
        force (bool, optional): Whether to force recomputation of cached results. Defaults to False.

    Attributes:
        data_name (str): Name of the current dataset being processed
        tmp_dir (str): Temporary directory for storing intermediate results

    Methods:
        from_config(cfg): Creates a KGConstructor instance from a configuration object
        create_kg(data_root, data_name): Creates a knowledge graph from the documents in the specified dataset
        get_document2entities(data_root, data_name): Gets mapping of documents to their extracted entities
        open_ie_extraction(raw_path): Performs Open IE on the dataset corpus
        create_graph(open_ie_result_path): Creates a knowledge graph from Open IE results
        augment_graph(graph, kb_phrase_dict): Augments the graph with similarity-based edges

    Notes:
        The knowledge graph is constructed in multiple steps:

        1. Open Information Extraction to get initial triples
        2. Entity Linking to normalize entities
        3. Optional augmentation with similarity-based edges
        4. Creation of the final graph structure
    """

    DELIMITER = KG_DELIMITER

    def __init__(
        self,
        open_ie_model: BaseOPENIEModel,
        el_model: BaseELModel,
        multimodal_extractor: BaseMultimodalExtractor | None = None,
        root: str = "tmp/kg_construction",
        num_processes: int = 1,
        cosine_sim_edges: bool = True,
        threshold: float = 0.8,
        max_sim_neighbors: int = 100,
        add_title: bool = True,
        force: bool = False,
        enable_multimodal: bool = False,
        enable_micro_kg: bool = False,
        modalities: list[str] | None = None,
        micro_to_macro_topk: int = 3,
        micro_to_macro_threshold: float = 0.75,
        micro_conf_threshold: float = 0.5,
        unaligned_conf_threshold: float = 0.7,
        micro_relation_whitelist: list[str] | None = None,
        min_entity_alnum_len: int = 2,
        merge_micro_into_kg: bool = False,
        enable_image_macro_description: bool = False,
        max_image_desc_per_doc: int = 3,
        max_image_desc_chars: int = 240,
        macro_image_desc_use_cache: bool = True,
    ) -> None:
        """Initialize the KGConstructor class.

        Args:
            open_ie_model (BaseOPENIEModel): Model for Open Information Extraction.
            el_model (BaseELModel): Model for Entity Linking.
            root (str, optional): Root directory for storing KG construction outputs. Defaults to "tmp/kg_construction".
            num_processes (int, optional): Number of processes for parallel processing. Defaults to 1.
            cosine_sim_edges (bool, optional): Whether to add cosine similarity edges. Defaults to True.
            threshold (float, optional): Similarity threshold for adding edges. Defaults to 0.8.
            max_sim_neighbors (int, optional): Maximum number of similar neighbors to connect. Defaults to 100.
            add_title (bool, optional): Whether to add document titles as nodes. Defaults to True.
            force (bool, optional): Whether to force reconstruction of existing outputs. Defaults to False.

        Attributes:
            open_ie_model: Model instance for Open Information Extraction
            el_model: Model instance for Entity Linking
            root: Root directory path
            num_processes: Number of parallel processes
            cosine_sim_edges: Flag for adding similarity edges
            threshold: Similarity threshold value
            max_sim_neighbors: Max number of similar neighbors
            add_title: Flag for adding document titles
            force: Flag for forced reconstruction
            data_name: Name of the dataset being processed
        """

        self.open_ie_model = open_ie_model
        self.el_model = el_model
        self.multimodal_extractor = multimodal_extractor
        self.root = root
        self.num_processes = num_processes
        self.cosine_sim_edges = cosine_sim_edges
        self.threshold = threshold
        self.max_sim_neighbors = max_sim_neighbors
        self.add_title = add_title
        self.force = force
        self.enable_multimodal = enable_multimodal
        self.enable_micro_kg = enable_micro_kg
        self.modalities = set(modalities or ["image", "table"])
        self.micro_to_macro_topk = micro_to_macro_topk
        self.micro_to_macro_threshold = micro_to_macro_threshold
        self.micro_conf_threshold = micro_conf_threshold
        self.unaligned_conf_threshold = unaligned_conf_threshold
        self.micro_relation_whitelist = (
            {processing_phrases(rel) for rel in micro_relation_whitelist}
            if micro_relation_whitelist
            else None
        )
        self.min_entity_alnum_len = min_entity_alnum_len
        self.merge_micro_into_kg = merge_micro_into_kg
        self.enable_image_macro_description = enable_image_macro_description
        self.max_image_desc_per_doc = max(0, int(max_image_desc_per_doc))
        self.max_image_desc_chars = max(0, int(max_image_desc_chars))
        self.macro_image_desc_use_cache = macro_image_desc_use_cache
        self.data_name = None
        self._latest_macro_image_desc: dict[str, list[str]] = {}

    @property
    def tmp_dir(self) -> str:
        """
        Returns the temporary directory path for data processing.

        This property method creates and returns a directory path specific to the current
        data_name under the root directory. The directory is created if it doesn't exist.

        Returns:
            str: Path to the temporary directory.

        Raises:
            AssertionError: If data_name is not set before accessing this property.
        """
        assert (
            self.data_name is not None
        )  # data_name should be set before calling this property
        tmp_dir = os.path.join(self.root, self.data_name)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        return tmp_dir

    @staticmethod
    def from_config(cfg: DictConfig) -> "KGConstructor":
        """
        Creates a KGConstructor instance from a configuration.

        This method initializes a KGConstructor using parameters specified in an OmegaConf
        configuration object. It creates a unique fingerprint of the configuration and sets up
        a temporary directory for storing processed data.

        Args:
            cfg (DictConfig): An OmegaConf configuration object containing the following parameters:

                - root: Base directory for storing temporary files
                - open_ie_model: Configuration for the Open IE model
                - el_model: Configuration for the Entity Linking model
                - num_processes: Number of processes to use
                - cosine_sim_edges: Whether to use cosine similarity for edges
                - threshold: Similarity threshold
                - max_sim_neighbors: Maximum number of similar neighbors
                - add_title: Whether to add titles
                - force: Whether to force reprocessing

        Returns:
            KGConstructor: An initialized KGConstructor instance

        Notes:
            The method creates a fingerprint of the configuration (excluding 'force' parameters)
            and uses it to create a temporary directory. The configuration is saved in this
            directory for reference.
        """
        config = OmegaConf.to_container(cfg, resolve=True)
        if "force" in config:
            del config["force"]
        if "force" in config["el_model"]:
            del config["el_model"]["force"]
        fingerprint = hashlib.md5(json.dumps(config).encode()).hexdigest()

        base_tmp_dir = os.path.join(cfg.root, fingerprint)
        if not os.path.exists(base_tmp_dir):
            os.makedirs(base_tmp_dir)
            json.dump(
                config,
                open(os.path.join(base_tmp_dir, "config.json"), "w"),
                indent=4,
            )
        multimodal_extractor = None
        if "multimodal_extractor" in cfg and cfg.multimodal_extractor is not None:
            if "_target_" in cfg.multimodal_extractor:
                multimodal_extractor = instantiate(cfg.multimodal_extractor)

        return KGConstructor(
            root=base_tmp_dir,
            open_ie_model=instantiate(cfg.open_ie_model),
            el_model=instantiate(cfg.el_model),
            multimodal_extractor=multimodal_extractor,
            num_processes=cfg.num_processes,
            cosine_sim_edges=cfg.cosine_sim_edges,
            threshold=cfg.threshold,
            max_sim_neighbors=cfg.max_sim_neighbors,
            add_title=cfg.add_title,
            force=cfg.force,
            enable_multimodal=cfg.get("enable_multimodal", False),
            enable_micro_kg=cfg.get("enable_micro_kg", False),
            modalities=list(cfg.get("modalities", ["image", "table"])),
            micro_to_macro_topk=cfg.get("micro_to_macro_topk", 3),
            micro_to_macro_threshold=cfg.get("micro_to_macro_threshold", 0.75),
            micro_conf_threshold=cfg.get("micro_conf_threshold", 0.5),
            unaligned_conf_threshold=cfg.get("unaligned_conf_threshold", 0.7),
            micro_relation_whitelist=list(cfg.get("micro_relation_whitelist", [])),
            min_entity_alnum_len=cfg.get("min_entity_alnum_len", 2),
            merge_micro_into_kg=cfg.get("merge_micro_into_kg", False),
            enable_image_macro_description=cfg.get(
                "enable_image_macro_description", False
            ),
            max_image_desc_per_doc=cfg.get("max_image_desc_per_doc", 3),
            max_image_desc_chars=cfg.get("max_image_desc_chars", 240),
            macro_image_desc_use_cache=cfg.get("macro_image_desc_use_cache", True),
        )

    def create_kg(self, data_root: str, data_name: str) -> list[tuple[str, str, str]]:
        """
        Create a knowledge graph from raw data.

        This method processes raw data to extract triples and construct a knowledge graph.
        It first performs Open IE extraction on the raw data, then creates a graph structure,
        and finally converts the graph into a list of triples.

        Args:
            data_root (str): Root directory path containing the data.
            data_name (str): Name of the dataset to process.

        Returns:
            list[tuple[str, str, str]]: List of extracted triples in the format (head, relation, tail).

        Note:
            If self.force is True, it will clear all temporary files before processing.
        """
        self.data_name = data_name  # type: ignore
        raw_path = os.path.join(data_root, data_name, "raw")
        corpus_doc_ids = self._load_corpus_doc_ids(raw_path)
        multimodal_doc_ids = self._load_multimodal_doc_ids(raw_path)
        text_doc_ids = set(corpus_doc_ids)
        image_doc_ids = set(multimodal_doc_ids) - text_doc_ids

        if self.force:
            for tmp_file in os.listdir(self.tmp_dir):
                tmp_path = os.path.join(self.tmp_dir, tmp_file)
                if os.path.isdir(tmp_path):
                    shutil.rmtree(tmp_path)
                else:
                    os.remove(tmp_path)

        open_ie_result_path = self.open_ie_extraction(raw_path)
        graph = self.create_graph(open_ie_result_path)
        macro_triples = [(h, r, t) for (h, t), r in graph.items()]
        self._write_json(os.path.join(self.tmp_dir, "macro_kg.json"), macro_triples)

        final_triples = list(macro_triples)
        macro_entities = {
            entity for triple in macro_triples for entity in (triple[0], triple[2])
        }
        base_document2entities = self._build_base_document2entities()
        image_desc_entities = self._extract_image_description_entities(
            self._latest_macro_image_desc,
            macro_entities,
        )
        text_table_document2entities = self.build_text_table_document2entities(
            base_document2entities,
            image_description_entities=image_desc_entities,
            macro_entities=macro_entities,
            allowed_doc_ids=text_doc_ids,
        )
        mm_document2entities = self.build_mm_document2entities(
            base_document2entities,
            [],
            {},
            macro_entities=macro_entities,
            image_description_entities=image_desc_entities,
            allowed_doc_ids=image_doc_ids,
        )

        self._write_json(
            os.path.join(self.tmp_dir, "document2entities_text_table.json"),
            text_table_document2entities,
        )
        self._write_jsonl(os.path.join(self.tmp_dir, "micro_kg.jsonl"), [])
        self._write_json(os.path.join(self.tmp_dir, "cross_tier_index.json"), {})
        self._write_json(os.path.join(self.tmp_dir, "entity2micro.json"), {})
        self._write_json(
            os.path.join(self.tmp_dir, "mm_document2entities.json"),
            mm_document2entities,
        )
        self._write_json(os.path.join(self.tmp_dir, "mmkg_stats.json"), {})

        if self.enable_multimodal and self.enable_micro_kg:
            micro_docs = self._load_multimodal_corpus(raw_path)
            if micro_docs:
                raw_micro_triples = self.construct_micro_kg(micro_docs)
                filtered_micro_triples = self.filter_micro_triples(raw_micro_triples)
                cross_tier_index, entity2micro = self.build_cross_tier_index(
                    filtered_micro_triples, list(macro_entities)
                )
                (
                    filtered_micro_triples,
                    cross_tier_index,
                    entity2micro,
                ) = self.prune_unaligned_micro_triples(
                    filtered_micro_triples, cross_tier_index, entity2micro
                )
                text_table_document2entities = self.build_text_table_document2entities(
                    base_document2entities,
                    filtered_micro_triples,
                    cross_tier_index,
                    image_description_entities=image_desc_entities,
                    macro_entities=macro_entities,
                    allowed_doc_ids=text_doc_ids,
                )
                mm_document2entities = self.build_mm_document2entities(
                    base_document2entities,
                    filtered_micro_triples,
                    cross_tier_index,
                    macro_entities=macro_entities,
                    image_description_entities=image_desc_entities,
                    allowed_doc_ids=image_doc_ids,
                )
                self._write_jsonl(
                    os.path.join(self.tmp_dir, "micro_kg.jsonl"), filtered_micro_triples
                )
                self._write_json(
                    os.path.join(self.tmp_dir, "cross_tier_index.json"),
                    cross_tier_index,
                )
                self._write_json(
                    os.path.join(self.tmp_dir, "entity2micro.json"), entity2micro
                )
                self._write_json(
                    os.path.join(self.tmp_dir, "mm_document2entities.json"),
                    mm_document2entities,
                )
                self._write_json(
                    os.path.join(self.tmp_dir, "document2entities_text_table.json"),
                    text_table_document2entities,
                )

                if self.merge_micro_into_kg:
                    final_triples.extend(
                        self._convert_micro_to_aligned_triples(
                            filtered_micro_triples,
                            cross_tier_index,
                        )
                    )

                self._write_json(
                    os.path.join(self.tmp_dir, "mmkg_stats.json"),
                    self._build_mmkg_stats(
                        filtered_micro_triples,
                        cross_tier_index,
                        micro_docs,
                    ),
                )
            else:
                logger.info(
                    "dataset_multimodal.json not found or empty, skip micro-KG construction."
                )

        return final_triples

    def get_document2entities(self, data_root: str, data_name: str) -> dict:
        """
        Retrieves a mapping of document titles to their associated entities from a preprocessed dataset.

        This method requires that a knowledge graph has been previously created using create_kg().
        If the necessary files do not exist, it will automatically call create_kg() first.

        Args:
            data_root (str): Root directory containing the dataset
            data_name (str): Name of the dataset to process

        Returns:
            dict: A dictionary mapping document titles (str) to lists of entity IDs (list)

        Raises:
            Warning: If passage information file is not found and create_kg needs to be run first
        """
        self.data_name = data_name  # type: ignore

        if not os.path.exists(os.path.join(self.tmp_dir, "passage_info.json")):
            logger.warning(
                "Document to entities mapping is not available. Run create_kg first"
            )
            self.create_kg(data_root, data_name)

        split_path = os.path.join(self.tmp_dir, "document2entities_text_table.json")
        if os.path.exists(split_path):
            with open(split_path) as fin:
                return json.load(fin)

        with open(os.path.join(self.tmp_dir, "passage_info.json")) as fin:
            passage_info = json.load(fin)
        return {doc["title"]: doc["entities"] for doc in passage_info}

    def get_macro_kg(
        self, data_root: str, data_name: str
    ) -> list[tuple[str, str, str]]:
        self.data_name = data_name  # type: ignore
        macro_path = os.path.join(self.tmp_dir, "macro_kg.json")
        if not os.path.exists(macro_path):
            self.create_kg(data_root, data_name)
        with open(macro_path) as fin:
            return [tuple(item) for item in json.load(fin)]

    def get_micro_kg(self, data_root: str, data_name: str) -> list[dict[str, Any]]:
        self.data_name = data_name  # type: ignore
        micro_path = os.path.join(self.tmp_dir, "micro_kg.jsonl")
        if not os.path.exists(micro_path):
            self.create_kg(data_root, data_name)
        if not os.path.exists(micro_path):
            return []
        with open(micro_path) as fin:
            return [json.loads(line) for line in fin if line.strip()]

    def get_cross_tier_index(
        self, data_root: str, data_name: str
    ) -> dict[str, list[str]]:
        self.data_name = data_name  # type: ignore
        cross_path = os.path.join(self.tmp_dir, "cross_tier_index.json")
        if not os.path.exists(cross_path):
            self.create_kg(data_root, data_name)
        if not os.path.exists(cross_path):
            return {}
        with open(cross_path) as fin:
            return json.load(fin)

    def get_entity2micro(self, data_root: str, data_name: str) -> dict[str, list[str]]:
        self.data_name = data_name  # type: ignore
        path = os.path.join(self.tmp_dir, "entity2micro.json")
        if not os.path.exists(path):
            self.create_kg(data_root, data_name)
        if not os.path.exists(path):
            return {}
        with open(path) as fin:
            return json.load(fin)

    def get_mm_document2entities(
        self, data_root: str, data_name: str
    ) -> dict[str, list[str]]:
        self.data_name = data_name  # type: ignore
        path = os.path.join(self.tmp_dir, "mm_document2entities.json")
        if not os.path.exists(path):
            self.create_kg(data_root, data_name)
        if not os.path.exists(path):
            return self.get_document2entities(data_root, data_name)
        with open(path) as fin:
            return json.load(fin)

    def get_mmkg_stats(self, data_root: str, data_name: str) -> dict[str, Any]:
        self.data_name = data_name  # type: ignore
        path = os.path.join(self.tmp_dir, "mmkg_stats.json")
        if not os.path.exists(path):
            self.create_kg(data_root, data_name)
        if not os.path.exists(path):
            return {}
        with open(path) as fin:
            return json.load(fin)

    def _write_json(self, path: str, payload: Any) -> None:
        with open(path, "w") as fout:
            json.dump(payload, fout, indent=4)

    def _write_jsonl(self, path: str, payload: list[dict[str, Any]]) -> None:
        with open(path, "w") as fout:
            for item in payload:
                fout.write(json.dumps(item) + "\n")

    def _build_base_document2entities(self) -> dict[str, list[str]]:
        passage_path = os.path.join(self.tmp_dir, "passage_info.json")
        if not os.path.exists(passage_path):
            return {}
        with open(passage_path) as fin:
            passage_info = json.load(fin)
        return {doc["title"]: doc.get("entities", []) for doc in passage_info}

    def _load_corpus_doc_ids(self, raw_path: str) -> set[str]:
        corpus_path = os.path.join(raw_path, "dataset_corpus.json")
        if not os.path.exists(corpus_path):
            return set()
        with open(corpus_path) as fin:
            payload = json.load(fin)
        if isinstance(payload, dict):
            return {str(doc_id) for doc_id in payload.keys()}
        if isinstance(payload, list):
            doc_ids: set[str] = set()
            for idx, item in enumerate(payload):
                if isinstance(item, dict):
                    doc_id = str(item.get("doc_id", item.get("title", f"doc_{idx}")))
                    doc_ids.add(doc_id)
                else:
                    doc_ids.add(f"doc_{idx}")
            return doc_ids
        return set()

    def _load_multimodal_doc_ids(self, raw_path: str) -> set[str]:
        multimodal_docs = self._load_multimodal_corpus(raw_path)
        return {
            str(item.get("doc_id", item.get("title", ""))).strip()
            for item in multimodal_docs
            if str(item.get("doc_id", item.get("title", ""))).strip()
        }

    def _extract_image_description_entities(
        self,
        macro_image_desc: dict[str, list[str]],
        macro_entities: set[str] | list[str],
    ) -> dict[str, list[str]]:
        if not macro_image_desc:
            return {}

        extracted_mentions: dict[str, list[str]] = {
            doc_id: [] for doc_id in macro_image_desc
        }
        for doc_id, descriptions in macro_image_desc.items():
            for description in descriptions:
                desc = str(description).strip()
                if not desc:
                    continue
                try:
                    openie_output = self.open_ie_model(desc)
                except Exception as exc:
                    logger.warning(
                        "Image description OpenIE extraction failed for %s: %s",
                        doc_id,
                        exc,
                    )
                    continue

                if not isinstance(openie_output, dict):
                    continue
                extracted_entities = openie_output.get("extracted_entities", [])
                if isinstance(extracted_entities, list):
                    extracted_mentions[doc_id].extend(extracted_entities)

        return self.link_image_description_entities(extracted_mentions, macro_entities)

    def link_image_description_entities(
        self,
        image_desc_mentions: dict[str, list[str]],
        macro_entities: set[str] | list[str],
    ) -> dict[str, list[str]]:
        if not image_desc_mentions:
            return {}

        macro_entity_set = set(macro_entities)
        if not macro_entity_set:
            return {doc_id: [] for doc_id in image_desc_mentions}

        self.el_model.index(list(macro_entity_set))
        image_desc_entities: dict[str, set[str]] = {
            doc_id: set() for doc_id in image_desc_mentions
        }
        for doc_id, mentions in image_desc_mentions.items():
            for entity in mentions:
                normalized = processing_phrases(str(entity))
                if not normalized:
                    continue
                if normalized in macro_entity_set:
                    image_desc_entities[doc_id].add(normalized)
                    continue
                try:
                    el_results = self.el_model(
                        [normalized], topk=self.micro_to_macro_topk
                    )
                except Exception as exc:
                    logger.warning(
                        "Entity linking failed for image description entity '%s': %s",
                        normalized,
                        exc,
                    )
                    continue
                for candidate in el_results.get(normalized, []):
                    if (
                        candidate.get("norm_score", 0.0)
                        >= self.micro_to_macro_threshold
                        and candidate.get("entity") in macro_entity_set
                    ):
                        image_desc_entities[doc_id].add(candidate["entity"])

        return {
            doc_id: sorted(entities) for doc_id, entities in image_desc_entities.items()
        }

    def _load_multimodal_corpus(self, raw_path: str) -> list[dict[str, Any]]:
        multimodal_path = os.path.join(raw_path, "dataset_multimodal.json")
        if not os.path.exists(multimodal_path):
            return []
        with open(multimodal_path) as fin:
            payload = json.load(fin)
        if isinstance(payload, dict):
            docs = []
            for doc_id, item in payload.items():
                if isinstance(item, dict):
                    docs.append(
                        {
                            "doc_id": item.get("doc_id", doc_id),
                            "title": item.get("title", doc_id),
                            "text": item.get("text", ""),
                            "images": item.get("images", []),
                            "tables": item.get("tables", []),
                        }
                    )
            return docs
        if isinstance(payload, list):
            docs = []
            for idx, item in enumerate(payload):
                if not isinstance(item, dict):
                    continue
                doc_id = str(item.get("doc_id", item.get("title", f"doc_{idx}")))
                docs.append(
                    {
                        "doc_id": doc_id,
                        "title": item.get("title", doc_id),
                        "text": item.get("text", ""),
                        "images": item.get("images", []),
                        "tables": item.get("tables", []),
                    }
                )
            return docs
        return []

    def _load_macro_image_desc_cache(self, cache_path: str) -> dict[str, str]:
        cache: dict[str, str] = {}
        if not os.path.exists(cache_path):
            return cache
        with open(cache_path, encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                row = json.loads(line)
                cache_key = str(row.get("cache_key", "")).strip()
                description = str(row.get("description", "")).strip()
                if cache_key:
                    cache[cache_key] = description
        return cache

    def _append_macro_image_desc_cache(
        self, cache_path: str, cache_key: str, description: str
    ) -> None:
        with open(cache_path, "a", encoding="utf-8") as fout:
            fout.write(
                json.dumps(
                    {"cache_key": cache_key, "description": description},
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _build_macro_image_cache_key(
        self, doc_id: str, image_item: dict[str, Any]
    ) -> str:
        source_id = (
            image_item.get("image_id")
            or image_item.get("id")
            or image_item.get("path_or_url")
            or image_item.get("path")
            or ""
        )
        signature = hashlib.md5(
            json.dumps(image_item, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        return f"{doc_id}|{source_id}|{signature}"

    def _collect_macro_image_descriptions(self, raw_path: str) -> dict[str, list[str]]:
        if (
            not self.enable_image_macro_description
            or self.multimodal_extractor is None
            or not hasattr(self.multimodal_extractor, "describe_image_for_macro")
        ):
            return {}

        multimodal_docs = self._load_multimodal_corpus(raw_path)
        if not multimodal_docs:
            return {}

        cache_path = os.path.join(self.tmp_dir, "macro_image_desc_cache.jsonl")
        cache = (
            self._load_macro_image_desc_cache(cache_path)
            if self.macro_image_desc_use_cache
            else {}
        )
        descriptions: dict[str, list[str]] = {}

        for doc in multimodal_docs:
            doc_id = str(doc.get("doc_id", doc.get("title", ""))).strip()
            if not doc_id:
                continue
            image_items = doc.get("images", []) or []
            if not isinstance(image_items, list):
                continue
            doc_descs: list[str] = []
            for image_item in image_items:
                if not isinstance(image_item, dict):
                    continue
                if (
                    self.max_image_desc_per_doc > 0
                    and len(doc_descs) >= self.max_image_desc_per_doc
                ):
                    break
                cache_key = self._build_macro_image_cache_key(doc_id, image_item)
                description = cache.get(cache_key, "")
                if not description:
                    try:
                        description = str(
                            self.multimodal_extractor.describe_image_for_macro(
                                image_item, doc
                            )
                        ).strip()
                    except Exception as exc:
                        logger.warning(
                            "Image macro description generation failed for doc=%s: %s",
                            doc_id,
                            exc,
                        )
                        description = ""
                    if self.max_image_desc_chars > 0:
                        description = description[: self.max_image_desc_chars].strip()
                    if self.macro_image_desc_use_cache:
                        self._append_macro_image_desc_cache(
                            cache_path, cache_key, description
                        )
                        cache[cache_key] = description
                if description:
                    doc_descs.append(" ".join(description.split()))
            if doc_descs:
                descriptions[doc_id] = doc_descs
        return descriptions

    def construct_micro_kg(
        self, multimodal_docs: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if self.multimodal_extractor is None:
            logger.warning(
                "Multimodal extraction is enabled but no multimodal_extractor is configured."
            )
            return []
        cache_path = os.path.join(self.tmp_dir, "micro_cache.jsonl")
        cached_sources = self._load_micro_cache(cache_path)

        for doc in tqdm(multimodal_docs, desc="Construct Micro-KG"):
            doc_id = str(doc.get("doc_id", doc.get("title", "")))
            if "image" in self.modalities:
                for image_item in doc.get("images", []) or []:
                    if not isinstance(image_item, dict):
                        continue
                    cache_key = self._build_micro_cache_key(doc_id, "image", image_item)
                    if cache_key in cached_sources:
                        continue
                    image_triples = self.multimodal_extractor.extract_from_image(
                        image_item, doc
                    )
                    normalized = self._normalize_micro_triples(
                        image_triples, doc_id=doc_id
                    )
                    self._append_micro_cache(cache_path, cache_key, normalized)
                    cached_sources[cache_key] = normalized
            if "table" in self.modalities:
                for table_item in doc.get("tables", []) or []:
                    if not isinstance(table_item, dict):
                        continue
                    cache_key = self._build_micro_cache_key(doc_id, "table", table_item)
                    if cache_key in cached_sources:
                        continue
                    table_triples = self.multimodal_extractor.extract_from_table(
                        table_item, doc
                    )
                    normalized = self._normalize_micro_triples(
                        table_triples, doc_id=doc_id
                    )
                    self._append_micro_cache(cache_path, cache_key, normalized)
                    cached_sources[cache_key] = normalized

        micro_triples: list[dict[str, Any]] = []
        for triples in cached_sources.values():
            micro_triples.extend(triples)
        return micro_triples

    def _normalize_micro_triples(
        self, triples: list[dict[str, Any]], doc_id: str
    ) -> list[dict[str, Any]]:
        normalized = []
        for triple in triples:
            head = processing_phrases(str(triple.get("head", "")))
            relation = processing_phrases(str(triple.get("relation", "")))
            tail = processing_phrases(str(triple.get("tail", "")))
            if not head or not relation or not tail:
                continue
            confidence = float(triple.get("confidence", 0.0))
            source_ref = str(triple.get("source_ref", ""))
            modality = str(triple.get("modality", "unknown"))
            triple_id_src = f"{doc_id}|{modality}|{head}|{relation}|{tail}|{source_ref}"
            triple_id = hashlib.md5(triple_id_src.encode()).hexdigest()
            normalized.append(
                {
                    "triple_id": triple_id,
                    "head": head,
                    "relation": relation,
                    "tail": tail,
                    "modality": modality,
                    "source_ref": source_ref,
                    "confidence": confidence,
                    "doc_id": doc_id,
                }
            )
        return normalized

    def filter_micro_triples(
        self, micro_triples: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        seen = set()
        for triple in micro_triples:
            confidence = float(triple.get("confidence", 0.0))
            if confidence < self.micro_conf_threshold:
                continue
            if self.micro_relation_whitelist is not None:
                relation = processing_phrases(str(triple.get("relation", "")))
                if relation not in self.micro_relation_whitelist:
                    continue
            if (
                self._alnum_len(triple["head"]) < self.min_entity_alnum_len
                or self._alnum_len(triple["tail"]) < self.min_entity_alnum_len
            ):
                continue
            key = (triple["doc_id"], triple["head"], triple["relation"], triple["tail"])
            if key in seen:
                continue
            seen.add(key)
            filtered.append(triple)
        return filtered

    def prune_unaligned_micro_triples(
        self,
        micro_triples: list[dict[str, Any]],
        cross_tier_index: dict[str, list[str]],
        entity2micro: dict[str, list[str]],
    ) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, list[str]]]:
        kept = []
        for triple in micro_triples:
            linked_entities = cross_tier_index.get(triple["triple_id"], [])
            confidence = float(triple.get("confidence", 0.0))
            if not linked_entities and confidence < self.unaligned_conf_threshold:
                continue
            kept.append(triple)

        kept_ids = {item["triple_id"] for item in kept}
        new_cross = {k: v for k, v in cross_tier_index.items() if k in kept_ids}
        new_entity2micro = {}
        for entity, triple_ids in entity2micro.items():
            remained = [triple_id for triple_id in triple_ids if triple_id in kept_ids]
            if remained:
                new_entity2micro[entity] = remained
        return kept, new_cross, new_entity2micro

    def _alnum_len(self, text: str) -> int:
        return len(re.sub(r"[^A-Za-z0-9]", "", text))

    def _build_micro_cache_key(
        self, doc_id: str, modality: str, item: dict[str, Any]
    ) -> str:
        source_id = (
            item.get("image_id")
            or item.get("table_id")
            or item.get("id")
            or item.get("path_or_url")
            or item.get("path")
            or ""
        )
        signature = hashlib.md5(
            json.dumps(item, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        return f"{doc_id}|{modality}|{source_id}|{signature}"

    def _load_micro_cache(self, cache_path: str) -> dict[str, list[dict[str, Any]]]:
        cache: dict[str, list[dict[str, Any]]] = {}
        if not os.path.exists(cache_path):
            return cache
        with open(cache_path) as fin:
            for line in fin:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row.get("cache_key")
                triples = row.get("triples", [])
                if isinstance(key, str) and isinstance(triples, list):
                    cache[key] = triples
        return cache

    def _append_micro_cache(
        self, cache_path: str, cache_key: str, triples: list[dict[str, Any]]
    ) -> None:
        with open(cache_path, "a") as fout:
            fout.write(json.dumps({"cache_key": cache_key, "triples": triples}) + "\n")

    def build_cross_tier_index(
        self, micro_triples: list[dict[str, Any]], macro_entities: list[str]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        cross_tier: dict[str, list[str]] = {}
        entity2micro: dict[str, list[str]] = {}
        if not micro_triples or not macro_entities:
            return cross_tier, entity2micro

        macro_entity_set = set(macro_entities)
        self.el_model.index(macro_entities)

        for triple in micro_triples:
            aligned = set()
            for endpoint in (triple["head"], triple["tail"]):
                if endpoint in macro_entity_set:
                    aligned.add(endpoint)
                    continue
                el_results = self.el_model([endpoint], topk=self.micro_to_macro_topk)
                for candidate in el_results.get(endpoint, []):
                    if candidate["norm_score"] >= self.micro_to_macro_threshold:
                        aligned.add(candidate["entity"])
            aligned_entities = sorted(aligned)
            cross_tier[triple["triple_id"]] = aligned_entities
            for entity in aligned_entities:
                entity2micro.setdefault(entity, []).append(triple["triple_id"])
        return cross_tier, entity2micro

    def build_mm_document2entities(
        self,
        base_document2entities: dict[str, list[str]],
        micro_triples: list[dict[str, Any]],
        cross_tier_index: dict[str, list[str]],
        macro_entities: set[str] | list[str] | None = None,
        image_description_entities: dict[str, list[str]] | None = None,
        allowed_doc_ids: set[str] | list[str] | None = None,
    ) -> dict[str, list[str]]:
        macro_entity_set = (
            set(macro_entities)
            if macro_entities is not None
            else {
                entity
                for entities in base_document2entities.values()
                for entity in entities
            }
        )
        allowed_docs = (
            set(allowed_doc_ids)
            if allowed_doc_ids is not None
            else set(base_document2entities.keys())
        )
        mm_document2entities: dict[str, list[str]] = {
            title: [] for title in allowed_docs
        }
        image_description_entities = image_description_entities or {}
        for doc_id, entities in image_description_entities.items():
            if doc_id not in allowed_docs:
                continue
            mm_document2entities.setdefault(doc_id, [])
            mm_document2entities[doc_id].extend(
                [entity for entity in entities if entity in macro_entity_set]
            )
        for triple in micro_triples:
            if str(triple.get("modality", "")).strip().lower() != "image":
                continue
            doc_id = triple["doc_id"]
            if doc_id not in allowed_docs:
                continue
            linked_entities = [
                entity
                for entity in cross_tier_index.get(triple["triple_id"], [])
                if entity in macro_entity_set
            ]
            if doc_id not in mm_document2entities:
                mm_document2entities[doc_id] = []
            mm_document2entities[doc_id].extend(linked_entities)

        for doc_id in list(mm_document2entities.keys()):
            mm_document2entities[doc_id] = sorted(set(mm_document2entities[doc_id]))
        return mm_document2entities

    def build_text_table_document2entities(
        self,
        base_document2entities: dict[str, list[str]],
        micro_triples: list[dict[str, Any]] | None = None,
        cross_tier_index: dict[str, list[str]] | None = None,
        image_description_entities: dict[str, list[str]] | None = None,
        macro_entities: set[str] | list[str] | None = None,
        allowed_doc_ids: set[str] | list[str] | None = None,
    ) -> dict[str, list[str]]:
        micro_triples = micro_triples or []
        cross_tier_index = cross_tier_index or {}
        image_description_entities = image_description_entities or {}

        macro_entity_set = (
            set(macro_entities)
            if macro_entities is not None
            else {
                entity
                for entities in base_document2entities.values()
                for entity in entities
            }
        )
        allowed_docs = (
            set(allowed_doc_ids)
            if allowed_doc_ids is not None
            else set(base_document2entities.keys())
        )
        image_entity_map = {
            doc_id: set(entities)
            for doc_id, entities in image_description_entities.items()
        }
        for triple in micro_triples:
            if str(triple.get("modality", "")).strip().lower() != "image":
                continue
            doc_id = str(triple.get("doc_id", ""))
            image_entity_map.setdefault(doc_id, set()).update(
                cross_tier_index.get(triple["triple_id"], [])
            )

        text_table_document2entities: dict[str, list[str]] = {}
        for doc_id in allowed_docs:
            entities = base_document2entities.get(doc_id, [])
            image_entities = image_entity_map.get(doc_id, set())
            kept_entities = [
                entity
                for entity in entities
                if entity in macro_entity_set and entity not in image_entities
            ]
            text_table_document2entities[doc_id] = sorted(set(kept_entities))

        for triple in micro_triples:
            if str(triple.get("modality", "")).strip().lower() != "table":
                continue
            doc_id = triple["doc_id"]
            if doc_id not in allowed_docs:
                continue
            linked_entities = [
                entity
                for entity in cross_tier_index.get(triple["triple_id"], [])
                if entity in macro_entity_set
            ]
            text_table_document2entities.setdefault(doc_id, [])
            text_table_document2entities[doc_id].extend(linked_entities)

        for doc_id in list(text_table_document2entities.keys()):
            text_table_document2entities[doc_id] = sorted(
                set(text_table_document2entities[doc_id])
            )
        return text_table_document2entities

    def _convert_micro_to_aligned_triples(
        self,
        micro_triples: list[dict[str, Any]],
        cross_tier_index: dict[str, list[str]],
    ) -> list[tuple[str, str, str]]:
        merged_micro = []
        for triple in micro_triples:
            aligned_entities = cross_tier_index.get(triple["triple_id"], [])
            if not aligned_entities:
                continue
            anchor = aligned_entities[0]
            merged_micro.append((anchor, triple["relation"], triple["tail"]))
        return merged_micro

    def _build_mmkg_stats(
        self,
        micro_triples: list[dict[str, Any]],
        cross_tier_index: dict[str, list[str]],
        multimodal_docs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        total_micro = len(micro_triples)
        aligned = sum(1 for entities in cross_tier_index.values() if entities)
        doc_count = len(multimodal_docs)
        docs_with_micro = len({triple["doc_id"] for triple in micro_triples})
        return {
            "num_docs": doc_count,
            "num_docs_with_micro_triples": docs_with_micro,
            "num_micro_triples": total_micro,
            "num_aligned_micro_triples": aligned,
            "cross_tier_coverage": aligned / total_micro if total_micro else 0.0,
            "avg_micro_triples_per_doc": total_micro / doc_count if doc_count else 0.0,
        }

    def open_ie_extraction(self, raw_path: str) -> str:
        """
        Perform open information extraction on the dataset corpus

        Args:
            raw_path (str): Path to the raw dataset

        Returns:
            str: Path to the openie results
        """
        with open(os.path.join(raw_path, "dataset_corpus.json")) as f:
            corpus = json.load(f)

        # Optionally inject concise image descriptions into macro passages.
        # This makes image-derived entities visible to macro OpenIE and downstream macro entity set.
        macro_image_desc = self._collect_macro_image_descriptions(raw_path)
        self._latest_macro_image_desc = macro_image_desc
        corpus_doc_ids = set(corpus.keys())
        if macro_image_desc:
            for doc_id, descriptions in macro_image_desc.items():
                if doc_id not in corpus_doc_ids:
                    continue
                base_text = str(corpus.get(doc_id, "")).strip()
                image_section = "\n".join(
                    [f"Image evidence: {desc}" for desc in descriptions if desc]
                ).strip()
                if not image_section:
                    continue
                merged = (
                    f"{base_text}\n{image_section}".strip()
                    if base_text
                    else image_section
                )
                corpus[doc_id] = merged
            logger.info(
                "Injected macro image descriptions for %s docs.",
                len(macro_image_desc),
            )

        if self.add_title:
            corpus = {
                title: title + "\n" + passage for title, passage in corpus.items()
            }
        passage_to_title = {corpus[title]: title for title in corpus.keys()}

        logger.info(f"Number of passages: {len(corpus)}")

        open_ie_result_path = f"{self.tmp_dir}/openie_results.jsonl"
        open_ie_results = {}
        if os.path.exists(open_ie_result_path):
            logger.info(f"OpenIE results already exist at {open_ie_result_path}")
            with open(open_ie_result_path) as f:
                for line in f:
                    data = json.loads(line)
                    open_ie_results[data["passage"]] = data

        remining_passages = [
            passage for passage in corpus.values() if passage not in open_ie_results
        ]
        logger.info(
            f"Number of passages which require processing: {len(remining_passages)}"
        )

        if len(remining_passages) > 0:
            with open(open_ie_result_path, "a") as f:
                with ThreadPool(processes=self.num_processes) as pool:
                    for result in tqdm(
                        pool.imap(self.open_ie_model, remining_passages),
                        total=len(remining_passages),
                        desc="Perform OpenIE",
                    ):
                        if isinstance(result, dict):
                            passage_title = passage_to_title[result["passage"]]
                            result["title"] = passage_title
                            f.write(json.dumps(result) + "\n")
                            f.flush()

        logger.info(f"OpenIE results saved to {open_ie_result_path}")
        return open_ie_result_path

    def create_graph(self, open_ie_result_path: str) -> dict:
        """
        Create a knowledge graph from the openie results

        Args:
            open_ie_result_path (str): Path to the openie results

        Returns:
            dict: Knowledge graph

                - key: (head, tail)
                - value: relation
        """

        with open(open_ie_result_path) as f:
            extracted_triples = [json.loads(line) for line in f]

        passage_json = []  # document-level information
        phrases = []  # clean triples
        entities = []  # entities from clean triples
        graph = {}  # {(h, t): r}
        incorrectly_formatted_triples = []  # those triples that len(triples) != 3
        triples_wo_ner_entity = []  # those triples that have entities out of ner entities
        triple_tuples = []  # all clean triples

        # Step 1: process OpenIE results
        for row in tqdm(extracted_triples, total=len(extracted_triples)):
            ner_entities = [processing_phrases(p) for p in row["extracted_entities"]]
            triples = row["extracted_triples"]
            doc_json = row

            clean_triples = []
            unclean_triples = []
            doc_entities = set()  # clean entities related to each sample

            for triple in triples:
                if not isinstance(triple, list) or any(
                    isinstance(i, list) or isinstance(i, tuple) for i in triple
                ):
                    continue

                if len(triple) > 1:
                    if len(triple) != 3:
                        clean_triple = [processing_phrases(p) for p in triple]
                        incorrectly_formatted_triples.append(triple)
                        unclean_triples.append(triple)
                    else:
                        clean_triple = [processing_phrases(p) for p in triple]
                        if "" in clean_triple or None in clean_triple:
                            incorrectly_formatted_triples.append(triple)
                            unclean_triples.append(triple)
                            continue

                        clean_triples.append(clean_triple)
                        phrases.extend(clean_triple)

                        head_ent = clean_triple[0]
                        tail_ent = clean_triple[2]

                        if (
                            head_ent not in ner_entities
                            and tail_ent not in ner_entities
                        ):
                            triples_wo_ner_entity.append(triple)

                        graph[(head_ent, tail_ent)] = clean_triple[1]

                        for triple_entity in [clean_triple[0], clean_triple[2]]:
                            entities.append(triple_entity)
                            doc_entities.add(triple_entity)

                doc_json["entities"] = list(set(doc_entities))
                doc_json["clean_triples"] = clean_triples
                doc_json["noisy_triples"] = unclean_triples
                triple_tuples.append(clean_triples)

                passage_json.append(doc_json)

        with open(os.path.join(self.tmp_dir, "passage_info.json"), "w") as f:
            json.dump(passage_json, f, indent=4)

        logging.info(f"Total number of processed data: {len(triple_tuples)}")

        lose_facts = []  # clean triples
        for triples in triple_tuples:
            lose_facts.extend([tuple(t) for t in triples])
        lose_fact_dict = {f: i for i, f in enumerate(lose_facts)}  # triples2id
        unique_phrases = list(np.unique(entities))  # Number of entities from documents
        unique_relations = np.unique(
            list(graph.values()) + ["equivalent"]
        )  # Number of relations from documents
        kb_phrase_dict = {p: i for i, p in enumerate(unique_phrases)}  # entities2id
        # Step 2: create raw graph
        logger.info("Creating Graph from OpenIE results")

        if self.cosine_sim_edges:
            self.augment_graph(
                graph, kb_phrase_dict=kb_phrase_dict
            )  # combine raw graph with synonyms edges

        synonymy_edges = {edge for edge in graph.keys() if graph[edge] == "equivalent"}
        stat_df = [
            ("Total Phrases", len(phrases)),
            ("Unique Phrases", len(unique_phrases)),
            ("Number of Individual Triples", len(lose_facts)),
            (
                "Number of Incorrectly Formatted Triples (ChatGPT Error)",
                len(incorrectly_formatted_triples),
            ),
            (
                "Number of Triples w/o NER Entities (ChatGPT Error)",
                len(triples_wo_ner_entity),
            ),
            ("Number of Unique Individual Triples", len(lose_fact_dict)),
            ("Number of Entities", len(entities)),
            ("Number of Edges", len(graph)),
            ("Number of Unique Entities", len(np.unique(entities))),
            ("Number of Synonymy Edges", len(synonymy_edges)),
            ("Number of Unique Relations", len(unique_relations)),
        ]

        logger.info("\n%s", pd.DataFrame(stat_df).set_index(0))

        return graph

    def augment_graph(self, graph: dict[Any, Any], kb_phrase_dict: dict) -> None:
        """
        Augment the graph with synonym edges between similar phrases.

        This method adds "equivalent" edges between phrases that are semantically similar based on embeddings.
        Similar phrases are found using an entity linking model and filtered based on similarity thresholds.

        Args:
            graph (dict[Any, Any]): The knowledge graph to augment, represented as an edge dictionary
                where keys are (phrase1, phrase2) tuples and values are edge types
            kb_phrase_dict (dict): Dictionary mapping phrases to their unique IDs in the knowledge base

        Returns:
            None: The graph is modified in place by adding new edges

        Notes:
            - Only processes phrases with >2 alphanumeric characters
            - Adds up to self.max_sim_neighbors equivalent edges per phrase
            - Only adds edges for pairs with similarity score above self.threshold
            - Uses self.el_model for computing phrase similarities
        """
        logger.info("Augmenting graph from similarity")

        unique_phrases = list(kb_phrase_dict.keys())
        processed_phrases = [processing_phrases(p) for p in unique_phrases]
        # Use a normalized phrase lookup to avoid key mismatches such as
        # original "False" vs processed "false".
        processed_kb_phrase_dict: dict[str, int] = {}
        for phrase, phrase_id in kb_phrase_dict.items():
            normalized = processing_phrases(phrase)
            if normalized and normalized not in processed_kb_phrase_dict:
                processed_kb_phrase_dict[normalized] = phrase_id

        self.el_model.index(processed_phrases)

        logger.info("Finding similar entities")
        sim_neighbors = self.el_model(processed_phrases, topk=self.max_sim_neighbors)

        logger.info("Adding synonymy edges")
        for phrase, neighbors in tqdm(sim_neighbors.items()):
            synonyms = []  # [(phrase_id, score)]
            if len(re.sub("[^A-Za-z0-9]", "", phrase)) > 2:
                phrase_id = processed_kb_phrase_dict.get(phrase)
                if phrase_id is not None:
                    num_nns = 0
                    for neighbor in neighbors:
                        n_entity = neighbor["entity"]
                        n_score = neighbor["norm_score"]
                        if n_score < self.threshold or num_nns > self.max_sim_neighbors:
                            break
                        if n_entity != phrase:
                            phrase2_id = processed_kb_phrase_dict.get(n_entity)
                            if phrase2_id is not None:
                                phrase2 = n_entity
                                synonyms.append((n_entity, n_score))
                                graph[(phrase, phrase2)] = "equivalent"
                                num_nns += 1
