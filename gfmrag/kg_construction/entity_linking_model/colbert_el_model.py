import hashlib
import os
import shutil

from ragatouille import RAGPretrainedModel

from gfmrag.kg_construction.utils import processing_phrases

from .base_model import BaseELModel


class ColbertELModel(BaseELModel):
    """ColBERT-based Entity Linking Model.

    This class implements an entity linking model using ColBERT, a neural information retrieval
    framework. It indexes a list of entities and performs entity linking by finding the most
    similar entities in the index for given named entities.

    Attributes:
        model_name_or_path (str): Path to the ColBERT checkpoint file
        root (str): Root directory for storing indices
        force (bool): Whether to force reindex if index exists
        entity_list (list): List of entities to be indexed
        index_path (str): Path to the index created by ColBERT

    Raises:
        AttributeError: If entity linking is attempted before indexing.

    Examples:
        >>> model = ColbertELModel("colbert-ir/colbertv2.0")
        >>> model.index(["entity1", "entity2", "entity3"])
        >>> results = model(["query1", "query2"], topk=3)
        >>> print(results)
        {'paris city': [{'entity': 'entity1', 'score': 0.82, 'norm_score': 1.0},
                        {'entity': 'entity2', 'score': 0.35, 'norm_score': 0.43}]}
    """

    def __init__(
        self,
        model_name_or_path: str = "colbert-ir/colbertv2.0",
        root: str = "tmp",
        force: bool = False,
        **kwargs: str,
    ) -> None:
        """
        Initialize the ColBERT entity linking model.

        This initializes a ColBERT model for entity linking using pre-trained checkpoints and indices.

        Args:
            model_name_or_path (str, optional): Path to the ColBERT checkpoint file. Defaults to "colbert-ir/colbertv2.0".
            root (str, optional): Root directory for storing indices. Defaults to "tmp".
            force (bool, optional): Whether to force recomputation of existing indices. Defaults to False.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist at the specified path.

        Returns:
            None
        """
        self.model_name_or_path = model_name_or_path
        self.root = root
        self.force = force
        self.colbert_model = RAGPretrainedModel.from_pretrained(
            self.model_name_or_path,
            index_root=self.root,
        )

    def index(self, entity_list: list) -> None:
        """
        Index a list of entities using ColBERT for efficient similarity search.

        This method processes and indexes a list of entities using the ColBERT model. It creates
        a unique index based on the MD5 hash of the entity list and stores it in the specified
        root directory.

        Args:
            entity_list (list): List of entity strings to be indexed.

        Returns:
            None

        Notes:
            - Creates a unique index directory based on MD5 hash of entities
            - If force=True, will delete existing index with same fingerprint
            - Processes entities into phrases before indexing
            - Sets up ColBERT indexer and searcher with specified configuration
            - Stores phrase_searcher as instance variable for later use
        """
        fingerprint = hashlib.md5("".join(entity_list).encode()).hexdigest()
        index_name = f"Entity_index_{fingerprint}"
        if os.path.exists(f"{self.root}/colbert/{fingerprint}") and self.force:
            shutil.rmtree(f"{self.root}/colbert/{fingerprint}")
        phrases = [processing_phrases(p) for p in entity_list]
        index_path = self.colbert_model.index(
            index_name=index_name,
            collection=phrases,
            overwrite_index=self.force if self.force else "reuse",
            split_documents=False,
            use_faiss=True,
        )
        self.index_path = index_path

    def __call__(self, ner_entity_list: list, topk: int = 1) -> dict:
        """
        Link entities in the given text to the knowledge graph.

        Args:
            ner_entity_list (list): list of named entities
            topk (int): number of linked entities to return

        Returns:
            dict: dict of linked entities in the knowledge graph

                - key (str): named entity
                - value (list[dict]): list of linked entities

                    - entity: linked entity
                    - score: score of the entity
                    - norm_score: normalized score of the entity
        """

        try:
            self.__getattribute__("index_path")
        except AttributeError as e:
            raise AttributeError("Index the entities first using index method") from e

        queries = [processing_phrases(p) for p in ner_entity_list]

        results = self.colbert_model.search(
            queries,
            k=topk,
        )

        # RAGatouille may return different shapes:
        # - multi-query: List[List[dict]]
        # - single-query: List[dict]
        # Normalize to List[List[dict]] for consistent downstream handling.
        if not isinstance(results, list):
            results = [[] for _ in queries]
        elif len(queries) == 1 and (
            len(results) == 0 or (len(results) > 0 and isinstance(results[0], dict))
        ):
            results = [results]

        linked_entity_dict: dict[str, list] = {}
        for i in range(len(queries)):
            query = queries[i]
            result = (
                results[i] if i < len(results) and isinstance(results[i], list) else []
            )
            linked_entity_dict[query] = []
            max_score = (
                max([r["score"] for r in result]) if result else 1.0
            )  # Avoid division by zero

            for r in result:
                if not isinstance(r, dict):
                    continue
                linked_entity_dict[query].append(
                    {
                        "entity": r.get("content", ""),
                        "score": r.get("score", 0.0),
                        "norm_score": (r.get("score", 0.0) / max_score)
                        if max_score
                        else 0.0,
                    }
                )

        return linked_entity_dict
