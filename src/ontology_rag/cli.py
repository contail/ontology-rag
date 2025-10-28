"""CLI for Ontology RAG."""

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv

from .core.rag_engine import RAGEngine
from .embeddings.client import EmbeddingClient
from .storage.vector_store import VectorStore

load_dotenv()

app = typer.Typer(help="🧠 Ontology RAG - Simple RAG for Learning")
console = Console()


def get_rag_engine() -> RAGEngine:
    """Create and return RAG engine instance."""
    llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    llm_model = os.getenv("LLM_MODEL", "qwen2.5vl:72b")
    embed_model = os.getenv("EMBED_MODEL", "mxbai-embed-large")
    
    embedder = EmbeddingClient(base_url=llm_base_url, model=embed_model)
    store = VectorStore()
    
    return RAGEngine(
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        embedding_client=embedder,
        vector_store=store,
    )


@app.command()
def add(file_path: str) -> None:
    """📄 Add documents from a text file to the knowledge base.
    
    Example:
        ontology-rag add data/my_document.txt
    """
    rag = get_rag_engine()
    
    # Read file
    content = Path(file_path).read_text(encoding="utf-8")
    
    # Smart chunking: merge titles with content
    raw_chunks = content.split("\n\n")
    chunks = []
    i = 0
    while i < len(raw_chunks):
        chunk = raw_chunks[i].strip()
        if i + 1 < len(raw_chunks) and len(chunk) < 50:
            chunk = chunk + "\n\n" + raw_chunks[i + 1].strip()
            i += 2
        else:
            i += 1
        if chunk:
            chunks.append(chunk)
    
    rag.add_documents(chunks)
    console.print(f"[green]✓[/green] Added {len(chunks)} documents to knowledge base")


@app.command()
def query(question: str) -> None:
    """💬 Ask a question to the RAG system.
    
    Example:
        ontology-rag query "What is ontology?"
    """
    rag = get_rag_engine()
    answer = rag.query(question, top_k=3, debug=False)
    
    console.print("\n[bold cyan]Answer:[/bold cyan]")
    console.print(answer)


@app.command()
def list() -> None:
    """📚 List all documents in the knowledge base."""
    store = VectorStore()
    results = store.get_all()
    
    table = Table(title=f"Knowledge Base ({store.count()} documents)")
    table.add_column("ID", style="cyan")
    table.add_column("Content Preview", style="white")
    
    for doc_id, doc in zip(results['ids'], results['documents']):
        preview = doc[:80] + "..." if len(doc) > 80 else doc
        table.add_row(doc_id, preview)
    
    console.print(table)


@app.command()
def clear() -> None:
    """🗑️  Clear all documents from the knowledge base."""
    store = VectorStore()
    store.clear()
    console.print("[green]✓[/green] All documents cleared")


@app.command()
def info() -> None:
    """ℹ️  Show system information."""
    llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:11434")
    llm_model = os.getenv("LLM_MODEL", "qwen2.5vl:72b")
    embed_model = os.getenv("EMBED_MODEL", "mxbai-embed-large")
    
    store = VectorStore()
    
    table = Table(title="System Information")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("LLM Base URL", llm_base_url)
    table.add_row("LLM Model", llm_model)
    table.add_row("Embedding Model", embed_model)
    table.add_row("Documents", str(store.count()))
    
    console.print(table)


if __name__ == "__main__":
    app()
