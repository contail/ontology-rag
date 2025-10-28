"""RAG Engine for document retrieval and generation."""

import httpx


class RAGEngine:
    """Main RAG engine combining retrieval and generation."""
    
    def __init__(self, llm_base_url: str, llm_model: str, embedding_client, vector_store):
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.embedding_client = embedding_client
        self.vector_store = vector_store
    
    def add_documents(self, documents: list[str]) -> None:
        """Add documents to the vector store."""
        embeddings = [self.embedding_client.embed(doc) for doc in documents]
        self.vector_store.add(documents, embeddings)
    
    def query(self, question: str, top_k: int = 3, debug: bool = False) -> str:
        """Query the RAG system."""
        # 1. Retrieve relevant documents
        question_embedding = self.embedding_client.embed(question)
        results = self.vector_store.search(question_embedding, top_k=top_k)
        
        if debug:
            print(f"🔍 검색 중...\n")
            print(f"Search results: {results}")
            print(f"Found {len(results['documents'][0])} documents")
            for i, doc in enumerate(results['documents'][0], 1):
                preview = doc[:100] + "..." if len(doc) > 100 else doc
                print(f"Doc {i}: {preview}\n")
        
        # 2. Generate answer
        if not results['documents'][0]:
            return "관련 문서를 찾을 수 없습니다."
        
        context = "\n\n".join(results['documents'][0])
        prompt = f"""다음 문서를 참고하여 질문에 답변하세요. 문서에 없는 내용은 답변하지 마세요.

문서:
{context}

질문: {question}

답변:"""
        
        response = httpx.post(
            f"{self.llm_base_url}/api/generate",
            json={"model": self.llm_model, "prompt": prompt, "stream": False},
            timeout=60.0
        )
        
        return response.json()["response"]
