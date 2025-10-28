"""Embedding client for generating vector embeddings."""

import httpx


class EmbeddingClient:
    """Client for generating embeddings using Ollama."""

    def __init__(self, base_url: str, model: str = "mxbai-embed-large", timeout: float = 60.0) -> None:
        self.base_url = base_url
        self.model = model
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        response = self._client.post(
            "/api/embeddings",
            json={"model": self.model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]
