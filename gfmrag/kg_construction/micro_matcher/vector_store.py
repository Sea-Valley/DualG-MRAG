from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from pymilvus import DataType, MilvusClient
except ImportError:  # pragma: no cover - optional runtime dependency
    DataType = None
    MilvusClient = None

try:
    import faiss
except ImportError:  # pragma: no cover - optional runtime dependency
    faiss = None


class FaissVectorStore:
    """In-memory FAISS wrapper with SimGRAG-compatible search payload."""

    def __init__(
        self,
        name: str,
        dim: int = 768,
        storage_dir: str = "tmp/micro_matcher_faiss",
        persist: bool = True,
    ):
        if faiss is None:
            raise RuntimeError(
                "faiss is required for FaissVectorStore. Install faiss-cpu/faiss-gpu first."
            )
        self.name = name
        self.dim = int(dim)
        self.persist = persist
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.storage_dir / f"{self.name}.faiss"
        self._meta_path = self.storage_dir / f"{self.name}.names.json"
        self._index = self._new_index()
        self._names: list[str] = []
        self._try_load()

    def _new_index(self) -> Any:
        return faiss.IndexFlatL2(self.dim)

    def _try_load(self) -> None:
        if not self.persist:
            return
        if self._index_path.exists() and self._meta_path.exists():
            self._index = faiss.read_index(str(self._index_path))
            with open(self._meta_path, encoding="utf-8") as fin:
                names = json.load(fin)
            self._names = [str(item) for item in names]

    def _save(self) -> None:
        if not self.persist:
            return
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "w", encoding="utf-8") as fout:
            json.dump(self._names, fout, ensure_ascii=False)

    def reset(self) -> None:
        self._index = self._new_index()
        self._names = []
        if self.persist:
            if self._index_path.exists():
                os.remove(self._index_path)
            if self._meta_path.exists():
                os.remove(self._meta_path)

    def insert(self, data: list[dict[str, Any]]) -> None:
        if not data:
            return
        vectors = np.asarray([row["vector"] for row in data], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dim mismatch for {self.name}, expected {self.dim}."
            )
        self._index.add(vectors)
        self._names.extend(str(row["name"]) for row in data)
        self._save()

    def search(self, data: list[list[float]], top_k: int) -> list[list[dict[str, Any]]]:
        if not data:
            return []
        if self._index.ntotal == 0:
            return [[] for _ in data]

        query = np.asarray(data, dtype=np.float32)
        top_k = min(max(int(top_k), 1), self._index.ntotal)
        distances, indices = self._index.search(query, top_k)

        payload: list[list[dict[str, Any]]] = []
        for row_dist, row_idx in zip(distances, indices):
            hits: list[dict[str, Any]] = []
            for dist, idx in zip(row_dist, row_idx):
                if idx < 0:
                    continue
                hits.append(
                    {
                        "distance": float(dist),
                        "entity": {"name": self._names[int(idx)]},
                    }
                )
            payload.append(hits)
        return payload

    def count(self) -> int:
        return int(self._index.ntotal)


class MilvusVectorStore:
    """Minimal Milvus wrapper for SimGRAG-style vector retrieval."""

    def __init__(self, name: str, uri: str = "http://localhost:19530", dim: int = 768):
        if MilvusClient is None or DataType is None:
            raise RuntimeError(
                "pymilvus is required for MilvusVectorStore. Install pymilvus first."
            )
        self.client = MilvusClient(uri=uri)
        self.name = name
        self.dim = int(dim)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.name not in self.client.list_collections():
            schema = MilvusClient.create_schema(
                auto_id=False, enable_dynamic_field=True
            )
            schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
            schema.add_field(
                field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim
            )
            self.client.create_collection(collection_name=self.name, schema=schema)

            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                metric_type="L2",
                index_type="HNSW",
                index_name="vector_index",
                params={"M": 64, "efConstruction": 512},
            )
            self.client.create_index(
                collection_name=self.name,
                index_params=index_params,
                sync=True,
            )
        self.client.load_collection(self.name)

    def reset(self) -> None:
        if self.name in self.client.list_collections():
            self.client.drop_collection(collection_name=self.name)
        self._ensure_collection()

    def insert(self, data: list[dict[str, Any]]) -> None:
        if not data:
            return
        self.client.insert(collection_name=self.name, data=data)

    def search(self, data: list[list[float]], top_k: int) -> list[list[dict[str, Any]]]:
        if not data:
            return []
        results = self.client.search(
            collection_name=self.name,
            data=data,
            limit=int(top_k),
            search_params={
                "metric_type": "L2",
                "params": {"efSearch": max(int(top_k) * 8, 16)},
            },
            output_fields=["name"],
        )
        payload = json.loads(json.dumps(results))
        return [sorted(row, key=lambda item: item["distance"]) for row in payload]

    def count(self) -> int:
        result = self.client.query(
            collection_name=self.name, output_fields=["count(*)"]
        )
        for row in result:
            return int(row["count(*)"])
        return 0
