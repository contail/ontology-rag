# 🧠 Ontology RAG

Simple RAG (Retrieval-Augmented Generation) system for ontology learning and experimentation.

## 🎯 Features

- **Simple CLI**: Easy-to-use command-line interface
- **Vector Search**: ChromaDB for efficient similarity search
- **Local LLM**: Works with Ollama or any OpenAI-compatible API
- **Smart Chunking**: Automatic document splitting with title merging
- **Guardrails**: Prevents hallucination with document-grounded responses

## 📋 Prerequisites

- Python 3.10+
- Ollama (or compatible LLM server)
- Conda or Poetry

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or create project
cd ontology-rag

# Create conda environment
conda create -n ontology-rag python=3.13 -y
conda activate ontology-rag

# Install dependencies
pip install -e .
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
LLM_BASE_URL=http://your-llm-server:11435
LLM_MODEL=qwen2.5vl:72b
EMBED_MODEL=mxbai-embed-large
```

### 3. Basic Usage

```bash
# Add documents to knowledge base
ontology-rag add data/my_document.txt

# Ask questions
ontology-rag query "What is this document about?"

# List all documents
ontology-rag list

# Show system info
ontology-rag info

# Clear knowledge base
ontology-rag clear
```

## 📚 Tutorial

### Step 1: Prepare Your Documents

Create a text file with your knowledge (e.g., `data/ontology_basics.txt`):

```text
What is Ontology?

An ontology is a formal representation of knowledge as a set of concepts within a domain, and the relationships between those concepts.

Types of Ontologies

1. Domain Ontology: Represents concepts specific to a particular domain
2. Upper Ontology: Describes general concepts that are the same across all domains
3. Task Ontology: Describes concepts related to a specific task or activity
```

### Step 2: Add to Knowledge Base

```bash
ontology-rag add data/ontology_basics.txt
```

Output:
```
✓ Added 3 documents to knowledge base
```

### Step 3: Query the System

```bash
ontology-rag query "What are the types of ontologies?"
```

The system will:
1. Convert your question to a vector embedding
2. Search for similar documents in the knowledge base
3. Retrieve top-k most relevant documents
4. Generate an answer based only on retrieved documents

### Step 4: Verify Documents

```bash
ontology-rag list
```

Shows all stored documents with previews.

## 🏗️ Architecture

```
ontology-rag/
├── src/ontology_rag/
│   ├── core/              # RAG engine
│   │   └── rag_engine.py
│   ├── embeddings/        # Embedding generation
│   │   └── client.py
│   ├── storage/           # Vector database
│   │   └── vector_store.py
│   └── cli.py            # Command-line interface
├── examples/             # Example scripts
├── notebooks/            # Jupyter tutorials
├── data/                 # Sample data
└── docs/                 # Documentation
```

## 🔧 Advanced Usage

### Debug Mode

See what documents are retrieved:

```bash
ontology-rag query "your question" --debug
```

### Custom Top-K

Retrieve more documents:

```bash
ontology-rag query "your question" --top-k 5
```

### Programmatic Usage

```python
from ontology_rag.core.rag_engine import RAGEngine
from ontology_rag.embeddings.client import EmbeddingClient
from ontology_rag.storage.vector_store import VectorStore

# Setup
embedder = EmbeddingClient(base_url="http://localhost:11434")
store = VectorStore()
rag = RAGEngine(
    llm_base_url="http://localhost:11434",
    llm_model="qwen2.5vl:72b",
    embedding_client=embedder,
    vector_store=store,
)

# Add documents
rag.add_documents(["Document 1 text", "Document 2 text"])

# Query
answer = rag.query("What is in the documents?")
print(answer)
```

## 📖 How RAG Works

1. **Indexing Phase**:
   - Documents are split into chunks
   - Each chunk is converted to a vector embedding
   - Embeddings are stored in ChromaDB

2. **Query Phase**:
   - User question is converted to embedding
   - Similar documents are retrieved using vector search
   - Retrieved documents are used as context for LLM
   - LLM generates answer based only on context

3. **Guardrails**:
   - System prompts LLM to only use provided documents
   - If no relevant documents found, returns "information not found"
   - Prevents hallucination and ensures grounded responses

## 🎓 Learning Path

1. **Basic RAG**: Start with simple document Q&A
2. **Ontology Integration**: Add OWL/RDF parsing
3. **Advanced Retrieval**: Implement hybrid search (vector + keyword)
4. **Multi-hop Reasoning**: Chain multiple queries
5. **Knowledge Graph**: Combine RAG with graph databases

## 🤝 Contributing

This is a learning project. Feel free to:
- Add new features
- Improve documentation
- Share examples
- Report issues

## 📝 License

MIT License - feel free to use for learning and teaching!

## 🔗 Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Ollama](https://ollama.ai/)
- [RAG Papers](https://arxiv.org/abs/2005.11401)
