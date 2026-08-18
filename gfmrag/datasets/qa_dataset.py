import hashlib
import json
import logging
import os
import os.path as osp
import sys
import warnings
from typing import Any

import datasets
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils import data as torch_data
from torch_geometric.data import InMemoryDataset, makedirs
from torch_geometric.data.dataset import _repr, files_exist

from gfmrag.datasets.kg_dataset import KGDataset
from gfmrag.text_emb_models import BaseTextEmbModel
from gfmrag.utils import get_rank
from gfmrag.utils.qa_utils import entities_to_mask

logger = logging.getLogger(__name__)


class QADataset(InMemoryDataset):
    """A dataset class for Question-Answering tasks built on top of a Knowledge Graph.

    This dataset inherits from torch_geometric's InMemoryDataset and processes raw QA data
    into a format suitable for graph-based QA models. It handles both training and test splits.

    Args:
        root (str): Root directory where the dataset should be saved.
        data_name (str): Name of the dataset.
        text_emb_model_cfgs (DictConfig): Configuration for the text embedding model used to encode questions.
        force_rebuild (bool, optional): If True, forces the dataset to be reprocessed even if it exists. Defaults to False.

    Attributes:
        name (str): Name of the dataset.
        kg (KGDataset): The underlying knowledge graph dataset.
        rel_emb_dim (int): Dimension of relation embeddings.
        ent2id (dict): Mapping from entity names to IDs.
        rel2id (dict): Mapping from relation names to IDs.
        doc (dict): Corpus of documents.
        doc2entities (dict): Mapping from documents to contained entities.
        raw_train_data (list): Raw training data samples.
        raw_test_data (list): Raw test data samples.
        ent2docs (torch.Tensor): Sparse tensor mapping entities to documents.
        id2doc (dict): Mapping from document IDs to document names.

    Notes:
        The processed dataset contains:
        - Question embeddings
        - Question entity masks
        - Supporting entity masks
        - Supporting document masks
        - Sample IDs

        The dataset processes raw JSON files and creates PyTorch tensors for efficient training.
    """

    def __init__(
        self,
        root: str,
        data_name: str,
        text_emb_model_cfgs: DictConfig,
        force_rebuild: bool = False,
    ):
        self.name = data_name
        self.force_rebuild = force_rebuild
        self.text_emb_model_cfgs = text_emb_model_cfgs
        self.fingerprint = hashlib.md5(
            json.dumps(
                OmegaConf.to_container(text_emb_model_cfgs, resolve=True)
            ).encode()
        ).hexdigest()
        kg = KGDataset(root, data_name, text_emb_model_cfgs, force_rebuild)
        self.kg = kg[0]  # The first element of the KGDataset is the Graph
        self.feat_dim = kg.feat_dim
        super().__init__(root, None, None)
        self.data = torch.load(self.processed_paths[0], weights_only=False)
        self.load_property()

    def __repr__(self) -> str:
        return f"{self.name}()"

    @property
    def raw_file_names(self) -> list:
        return ["train.json", "test.json"]

    @property
    def raw_dir(self) -> str:
        return os.path.join(str(self.root), str(self.name), "processed", "stage1")

    @property
    def processed_dir(self) -> str:
        return os.path.join(
            str(self.root),
            str(self.name),
            "processed",
            "stage2",
            self.fingerprint,
        )

    @property
    def processed_file_names(self) -> str:
        return "qa_data.pt"

    def _resolve_doc2entities_paths(self) -> tuple[str, str | None]:
        text_table_path = os.path.join(self.raw_dir, "document2entities.json")
        image_path = os.path.join(self.raw_dir, "mm_document2entities.json")
        return text_table_path, image_path if os.path.exists(image_path) else None

    def _load_doc2entities(self, path: str | None) -> dict[str, list[str]]:
        if path is None or not os.path.exists(path):
            return {}
        with open(path) as fin:
            payload = json.load(fin)
        if not isinstance(payload, dict):
            return {}
        normalized: dict[str, list[str]] = {}
        for doc_id, entities in payload.items():
            if not isinstance(entities, list):
                continue
            normalized[str(doc_id)] = [str(entity) for entity in entities]
        return normalized

    def _build_doc_order(
        self,
        text_doc2entities: dict[str, list[str]],
        image_doc2entities: dict[str, list[str]],
        doc_corpus: dict[str, str] | None = None,
    ) -> list[str]:
        allowed_docs = set(text_doc2entities.keys()) | set(image_doc2entities.keys())
        doc_order: list[str] = []
        seen: set[str] = set()
        sources: list[dict[str, Any]] = [
            text_doc2entities,
            image_doc2entities,
            doc_corpus or {},
        ]
        for source in sources:
            for doc_id in source:
                if doc_id not in allowed_docs:
                    continue
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                doc_order.append(doc_id)
        return doc_order

    def _build_mm_doc_content(self, item: dict, fallback_id: str) -> str:
        title = str(item.get("title", fallback_id)).strip()
        text = str(item.get("text", "")).strip()
        lines: list[str] = []
        if title:
            lines.append(f"Title: {title}")
        if text:
            lines.append(text)

        for image_item in item.get("images", []) or []:
            if not isinstance(image_item, dict):
                continue
            img_title = str(image_item.get("title", "")).strip()
            img_caption = str(image_item.get("caption", "")).strip()
            if img_caption:
                lines.append(f"Image caption: {img_caption}")
            elif img_title:
                lines.append(f"Image: {img_title}")

        for table_item in item.get("tables", []) or []:
            if not isinstance(table_item, dict):
                continue
            schema = table_item.get("schema", table_item.get("headers", []))
            if isinstance(schema, list) and schema:
                cols = [str(col).strip() for col in schema if str(col).strip()]
                if cols:
                    lines.append("Table columns: " + " | ".join(cols))
            rows = table_item.get("rows", [])
            if isinstance(rows, list) and rows:
                first_row = rows[0]
                if isinstance(first_row, dict):
                    row_text = ", ".join(
                        f"{k}: {v}" for k, v in first_row.items() if str(v).strip()
                    )
                    if row_text:
                        lines.append(f"Table row: {row_text}")
                elif isinstance(first_row, list):
                    vals = [str(v).strip() for v in first_row if str(v).strip()]
                    if vals:
                        lines.append("Table row: " + " | ".join(vals))

        return "\n".join(lines).strip() or fallback_id

    def _load_doc_corpus_with_multimodal_fallback(self) -> dict:
        corpus_path = os.path.join(
            str(self.root), str(self.name), "raw", "dataset_corpus.json"
        )
        if os.path.exists(corpus_path):
            with open(corpus_path) as fin:
                doc = json.load(fin)
        else:
            doc = {}

        multimodal_path = os.path.join(
            str(self.root), str(self.name), "raw", "dataset_multimodal.json"
        )
        if not os.path.exists(multimodal_path):
            return doc

        with open(multimodal_path, encoding="utf-8") as fin:
            multimodal_payload = json.load(fin)

        mm_items: list[tuple[str, dict]] = []
        if isinstance(multimodal_payload, dict):
            mm_items = [
                (str(doc_id), item)
                for doc_id, item in multimodal_payload.items()
                if isinstance(item, dict)
            ]
        elif isinstance(multimodal_payload, list):
            for idx, item in enumerate(multimodal_payload):
                if not isinstance(item, dict):
                    continue
                doc_id = str(item.get("doc_id", item.get("title", f"doc_{idx}")))
                mm_items.append((doc_id, item))

        for doc_id, item in mm_items:
            if doc_id not in doc or not str(doc.get(doc_id, "")).strip():
                doc[doc_id] = self._build_mm_doc_content(item, doc_id)

        return doc

    def _load_image_doc_ids(self) -> set[str]:
        candidates = [
            os.path.join(str(self.root), str(self.name), "dataset_corpus_image.json"),
            os.path.join(
                str(self.root), str(self.name), "raw", "dataset_corpus_image.json"
            ),
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as fin:
                payload = json.load(fin)
            if isinstance(payload, dict):
                return {str(doc_id) for doc_id in payload.keys()}
        # Fallback for datasets that do not expose dataset_corpus_image.json.
        return set(self.doc2entities_image.keys())

    def load_property(self) -> None:
        """
        Load necessary properties from the KG dataset.
        """
        with open(os.path.join(self.processed_dir, "ent2id.json")) as fin:
            self.ent2id = json.load(fin)
        with open(os.path.join(self.processed_dir, "rel2id.json")) as fin:
            self.rel2id = json.load(fin)
        self.doc = self._load_doc_corpus_with_multimodal_fallback()
        text_path, image_path = self._resolve_doc2entities_paths()
        self.doc2entities_text = self._load_doc2entities(text_path)
        self.doc2entities_image = self._load_doc2entities(image_path)
        self.image_doc_ids = self._load_image_doc_ids()
        self.doc2entities = self.doc2entities_text
        if os.path.exists(os.path.join(self.raw_dir, "train.json")):
            with open(os.path.join(self.raw_dir, "train.json")) as fin:
                self.raw_train_data = json.load(fin)
        else:
            self.raw_train_data = []
        if os.path.exists(os.path.join(self.raw_dir, "test.json")):
            with open(os.path.join(self.raw_dir, "test.json")) as fin:
                self.raw_test_data = json.load(fin)
        else:
            self.raw_test_data = []

        text_ent2doc_path = os.path.join(self.processed_dir, "ent2doc_text.pt")
        if os.path.exists(text_ent2doc_path):
            self.ent2docs_text = torch.load(text_ent2doc_path, weights_only=True)
        else:
            self.ent2docs_text = torch.load(
                os.path.join(self.processed_dir, "ent2doc.pt"), weights_only=True
            )
        image_ent2doc_path = os.path.join(self.processed_dir, "ent2doc_image.pt")
        if os.path.exists(image_ent2doc_path):
            self.ent2docs_image = torch.load(image_ent2doc_path, weights_only=True)
        else:
            self.ent2docs_image = torch.zeros(
                self.ent2docs_text.shape,
                dtype=self.ent2docs_text.dtype,
            ).to_sparse()
        self.ent2docs = self.ent2docs_text  # backward-compatible alias
        doc_order = self._build_doc_order(
            self.doc2entities_text,
            self.doc2entities_image,
            self.doc,
        )
        self.id2doc = {i: doc for i, doc in enumerate(doc_order)}

    def _process(self) -> None:
        f = osp.join(self.processed_dir, "pre_transform.pt")
        if osp.exists(f) and torch.load(f, weights_only=False) != _repr(
            self.pre_transform
        ):
            warnings.warn(
                f"The `pre_transform` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-processing technique, make sure to "
                f"delete '{self.processed_dir}' first",
                stacklevel=1,
            )

        f = osp.join(self.processed_dir, "pre_filter.pt")
        if osp.exists(f) and torch.load(f, weights_only=False) != _repr(
            self.pre_filter
        ):
            warnings.warn(
                f"The `pre_filter` argument differs from the one used in "
                f"the pre-processed version of this dataset. If you want to "
                f"make use of another pre-fitering technique, make sure to "
                f"delete '{self.processed_dir}' first",
                stacklevel=1,
            )

        if self.force_rebuild or not files_exist(self.processed_paths):
            logger.warning(f"Processing QA dataset {self.name} at rank {get_rank()}")
            if self.log and "pytest" not in sys.modules:
                print("Processing...", file=sys.stderr)

            makedirs(self.processed_dir)
            self.process()

            path = osp.join(self.processed_dir, "pre_transform.pt")
            torch.save(_repr(self.pre_transform), path)
            path = osp.join(self.processed_dir, "pre_filter.pt")
            torch.save(_repr(self.pre_filter), path)

            if self.log and "pytest" not in sys.modules:
                print("Done!", file=sys.stderr)

    def process(self) -> None:
        """Process and prepare the question-answering dataset.

        This method processes raw data files to create a structured dataset for question answering
        tasks. It performs the following main operations:

        1. Loads entity and relation mappings from processed files
        2. Creates entity-document mapping tensors
        3. Processes question samples to generate:
            - Question embeddings
            - Question entity masks
            - Supporting entity masks
            - Supporting document masks

        The processed dataset is saved as torch splits containing:

        - Question embeddings
        - Various mask tensors for entities and documents
        - Sample IDs

        Files created:

        - ent2doc.pt: Sparse tensor mapping entities to documents
        - qa_data.pt: Processed QA dataset
        - text_emb_model_cfgs.json: Text embedding model configuration

        The method also saves the text embedding model configuration.

        Returns:
            None
        """
        with open(os.path.join(self.processed_dir, "ent2id.json")) as fin:
            self.ent2id = json.load(fin)
        with open(os.path.join(self.processed_dir, "rel2id.json")) as fin:
            self.rel2id = json.load(fin)
        text_path, image_path = self._resolve_doc2entities_paths()
        self.doc2entities_text = self._load_doc2entities(text_path)
        self.doc2entities_image = self._load_doc2entities(image_path)
        self.image_doc_ids = self._load_image_doc_ids()
        self.doc2entities = self.doc2entities_text
        self.doc = self._load_doc_corpus_with_multimodal_fallback()

        num_nodes = self.kg.num_nodes
        doc_order = self._build_doc_order(
            self.doc2entities_text,
            self.doc2entities_image,
            self.doc,
        )
        doc2id = {doc: i for i, doc in enumerate(doc_order)}
        n_docs = len(doc_order)

        def _build_ent2doc(doc2entities: dict[str, list[str]]) -> torch.Tensor:
            doc2ent = torch.zeros((n_docs, num_nodes))
            for doc, entities in doc2entities.items():
                if doc not in doc2id:
                    continue
                entity_ids = [
                    self.ent2id[ent] for ent in entities if ent in self.ent2id
                ]
                doc2ent[doc2id[doc], entity_ids] = 1
            return doc2ent.T.to_sparse()

        ent2doc_text = _build_ent2doc(self.doc2entities_text)
        ent2doc_image = _build_ent2doc(self.doc2entities_image)
        torch.save(ent2doc_text, os.path.join(self.processed_dir, "ent2doc_text.pt"))
        torch.save(ent2doc_image, os.path.join(self.processed_dir, "ent2doc_image.pt"))
        torch.save(ent2doc_text, os.path.join(self.processed_dir, "ent2doc.pt"))

        sample_id = []
        questions = []
        question_entities_masks = []  # Convert question entities to mask with number of nodes
        supporting_entities_masks = []
        supporting_docs_masks = []
        num_samples = []

        for path in self.raw_paths:
            if not os.path.exists(path):
                num_samples.append(0)
                continue  # Skip if the file does not exist
            num_sample = 0
            is_test_split = os.path.basename(path) == "test.json"
            with open(path) as fin:
                data = json.load(fin)
                for index, item in enumerate(data):
                    question_entities = [
                        self.ent2id[x]
                        for x in item["question_entities"]
                        if x in self.ent2id
                    ]

                    supporting_entities = [
                        self.ent2id[x]
                        for x in item["supporting_entities"]
                        if x in self.ent2id
                    ]

                    supporting_docs = [
                        doc2id[doc] for doc in item["supporting_facts"] if doc in doc2id
                    ]

                    # Do not drop test samples only because linked entities are empty.
                    # This keeps evaluation coverage close to the original test size.
                    if (not is_test_split) and len(question_entities) == 0:
                        continue
                    num_sample += 1
                    sample_id.append(index)
                    question = item["question"]
                    questions.append(question)

                    question_entities_masks.append(
                        entities_to_mask(question_entities, num_nodes)
                    )

                    supporting_entities_masks.append(
                        entities_to_mask(supporting_entities, num_nodes)
                    )

                    supporting_docs_masks.append(
                        entities_to_mask(supporting_docs, n_docs)
                    )
                num_samples.append(num_sample)

        logger.info("Generating question embeddings")
        text_emb_model: BaseTextEmbModel = instantiate(self.text_emb_model_cfgs)
        question_embeddings = text_emb_model.encode(
            questions,
            is_query=True,
        ).cpu()
        question_entities_masks = torch.stack(question_entities_masks)
        supporting_entities_masks = torch.stack(supporting_entities_masks)
        supporting_docs_masks = torch.stack(supporting_docs_masks)
        sample_id = torch.tensor(sample_id, dtype=torch.long)

        dataset = datasets.Dataset.from_dict(
            {
                "question_embeddings": question_embeddings,
                "question_entities_masks": question_entities_masks,
                "supporting_entities_masks": supporting_entities_masks,
                "supporting_docs_masks": supporting_docs_masks,
                "sample_id": sample_id,
            }
        ).with_format("torch")
        offset = 0
        splits = []
        for num_sample in num_samples:
            split = torch_data.Subset(dataset, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        torch.save(splits, self.processed_paths[0])

        with open(self.processed_dir + "/text_emb_model_cfgs.json", "w") as f:
            json.dump(OmegaConf.to_container(self.text_emb_model_cfgs), f, indent=4)
