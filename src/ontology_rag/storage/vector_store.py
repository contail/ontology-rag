"""Vector store using ChromaDB."""

from typing import Any
import chromadb
from chromadb.config import Settings


class VectorStore:
    """Simple vector store using ChromaDB."""

    def __init__(self, collection_name: str = "documents", persist_dir: str = "./chroma_db") -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """Add documents to the vector store."""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        self._collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def search(self, query_embedding: list[float], top_k: int = 3) -> dict[str, Any]:
        """Search for similar documents."""
        return self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    def get_all(self) -> dict[str, Any]:
        """Get all documents."""
        return self._collection.get()

    def clear(self) -> None:
        """Clear all documents."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(name=self._collection.name)

    def count(self) -> int:
        """Get document count."""
        return self._collection.count()
