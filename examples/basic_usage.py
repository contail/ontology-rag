"""Basic usage example of Ontology RAG."""

import os
from dotenv import load_dotenv

from ontology_rag.core.rag_engine import RAGEngine
from ontology_rag.embeddings.client import EmbeddingClient
from ontology_rag.storage.vector_store import VectorStore

# Load environment
load_dotenv()


def main():
    """Run basic RAG example."""
    # Setup
    llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    llm_model = os.getenv("LLM_MODEL", "qwen2.5vl:72b")
    embed_model = os.getenv("EMBED_MODEL", "mxbai-embed-large")
    
    print("🧠 Initializing Ontology RAG...")
    
    embedder = EmbeddingClient(base_url=llm_base_url, model=embed_model)
    store = VectorStore(collection_name="example")
    
    rag = RAGEngine(
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        embedding_client=embedder,
        vector_store=store,
    )
    
    # Add sample documents
    print("\n📄 Adding documents...")
    documents = [
        "온톨로지는 특정 도메인의 개념과 관계를 형식적으로 표현한 지식 체계입니다.",
        "RDF는 Resource Description Framework의 약자로, 웹 리소스를 기술하기 위한 프레임워크입니다.",
        "OWL은 Web Ontology Language로, 복잡한 온톨로지를 표현하기 위한 W3C 표준입니다.",
    ]
    
    rag.add_documents(documents)
    print(f"✓ Added {len(documents)} documents")
    
    # Query
    print("\n💬 Querying...")
    questions = [
        "온톨로지란 무엇인가요?",
        "RDF는 무엇의 약자인가요?",
        "OWL에 대해 설명해주세요.",
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = rag.query(question, top_k=2)
        print(f"A: {answer}")
    
    # Cleanup
    print("\n🗑️  Cleaning up...")
    store.clear()
    print("✓ Done!")


if __name__ == "__main__":
    main()
