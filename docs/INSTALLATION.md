# 📦 Installation Guide

## Prerequisites

- Python 3.10 or higher
- Ollama or compatible LLM server
- Git

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ontology-rag
```

### 2. Create Virtual Environment

#### Option A: Using Conda (Recommended)

```bash
# Create environment
conda create -n ontology-rag python=3.13 -y

# Activate environment
conda activate ontology-rag
```

#### Option B: Using venv

```bash
# Create environment
python -m venv .venv

# Activate environment (macOS/Linux)
source .venv/bin/activate

# Activate environment (Windows)
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install in development mode
pip install -e .

# Or install required packages directly
pip install chromadb httpx typer rich python-dotenv
```

### 4. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file
nano .env  # or use your preferred editor
```

Update the following variables:

```bash
LLM_BASE_URL=http://your-server:11435
LLM_MODEL=qwen2.5vl:72b
EMBED_MODEL=mxbai-embed-large
```

### 5. Verify Installation

```bash
# Check if CLI is available
ontology-rag info

# Should show system information
```

## Troubleshooting

### Issue: `ontology-rag: command not found`

**Solution**: Make sure you installed in editable mode:

```bash
pip install -e .
```

### Issue: `ModuleNotFoundError: No module named 'chromadb'`

**Solution**: Install dependencies:

```bash
pip install chromadb
```

### Issue: Connection error to LLM server

**Solution**: Check your `.env` configuration and ensure the LLM server is running:

```bash
# Test connection
curl http://your-server:11435/api/tags
```

### Issue: Tornado import error

**Solution**: Install tornado:

```bash
pip install tornado
```

## Next Steps

After installation, check out:

- [Quick Start Guide](../README.md#-quick-start)
- [Tutorial Notebook](../notebooks/01_getting_started.ipynb)
- [Example Scripts](../examples/)

## Uninstallation

```bash
# Remove package
pip uninstall ontology-rag

# Remove environment (conda)
conda deactivate
conda env remove -n ontology-rag

# Remove environment (venv)
deactivate
rm -rf .venv
```
