"""RAG (Retrieval-Augmented Generation) engine."""

import httpx
from ..embeddings.client import EmbeddingClient
from ..storage.vector_store import VectorStore


class RAGEngine:
    """Main RAG engine for document retrieval and generation."""

    def __init__(
        self,
        llm_base_url: str,
        llm_model: str,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        timeout: float = 60.0,
    ) -> None:
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.embedder = embedding_client
        self.store = vector_store
        self._llm_client = httpx.Client(base_url=llm_base_url, timeout=timeout)

    def add_documents(self, texts: list[str]) -> None:
        """Add documents to the knowledge base."""
        embeddings = self.embedder.embed_batch(texts)
        self.store.add(texts=texts, embeddings=embeddings)

    def query(self, question: str, top_k: int = 3, debug: bool = False) -> str:
        """Query with RAG: retrieve relevant docs and generate answer."""
        # 1. Embed the question
        query_embedding = self.embedder.embed(question)

        # 2. Search for relevant documents
        results = self.store.search(query_embedding, top_k=top_k)
        
        if debug:
            print(f"Search results: {results}")
        
        # 3. Extract retrieved documents
        documents = results.get("documents", [[]])[0]
        
        if debug:
            print(f"Found {len(documents)} documents")
            for i, doc in enumerate(documents):
                print(f"Doc {i+1}: {doc[:100]}")
        
        # 4. Check if no documents found
        if not documents:
            return "죄송합니다. 해당 질문에 대한 정보를 찾을 수 없습니다."

        # 5. Build context from retrieved documents
        context = "\n\n".join([f"[문서 {i+1}]\n{doc}" for i, doc in enumerate(documents)])

        # 6. Generate answer with context
        prompt = f"""다음 문서들을 참고하여 질문에 답변하세요.

{context}

질문: {question}

답변 규칙:
- 반드시 제공된 문서의 내용만을 기반으로 답변하세요.
- 문서에 없는 내용은 추측하지 마세요.
- 확실하지 않으면 "문서에서 해당 정보를 찾을 수 없습니다"라고 답변하세요.

답변:"""
        
        return self._generate(prompt)

    def _generate(self, prompt: str) -> str:
        """Generate text using LLM."""
        response = self._llm_client.post(
            "/api/generate",
            json={"model": self.llm_model, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        return response.json().get("response", "")
