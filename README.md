# AI Services Orchestrator

A unified framework combining RAG (Retrieval Augmented Generation), web search, image generation, and system operations into a single interface.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch web interface
python main.py --mode web

## ✨ Features

- **RAG Pipeline**: Document ingestion, chunking, embedding, and intelligent retrieval
- **Web Search**: Integration with multiple search providers
- **Image Generation**: AI-powered image creation
- **System Operations**: Safe file and system management

## 🛠️ Installation

1. **Clone and navigate**:
   ```bash
   git clone <your-repo>
   cd data-embedding-project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Set up local LLM server** (recommended):
   - Install [llama.cpp](https://github.com/ggerganov/llama.cpp) or [text-generation-webui](https://github.com/oobabooga/text-generation-webui)
   - Download a GGUF model (e.g., Qwen2.5-7B-Instruct)
   - Start server on `http://127.0.0.1:8080`

## 📖 Usage

### Web Interface
```bash
python main.py --mode web
# Open http://localhost:5050
```

### Document Ingestion
```bash
# From command line
python main.py --mode ingest --urls https://example.com --files document.txt

# Or edit config/data_sources.json
```

## 📁 Project Structure

```
├── core/                   # Core interfaces and configuration
│   ├── interfaces.py      # Abstract base classes
│   ├── config.py         # Configuration dataclasses
│   └── __init__.py
├── loaders/               # Data loading modules
│   ├── web_loader.py     # Web scraping
│   ├── file_loader.py    # File processing
│   └── __init__.py
├── processors/            # Text processing
│   ├── chunkers.py       # Text chunking strategies
│   ├── embeddings.py     # Embedding models
│   └── __init__.py
├── storage/               # Vector storage
│   ├── memory_store.py   # In-memory vector store
│   └── __init__.py
├── services/              # AI services
│   ├── image_generation.py
│   ├── web_search.py
│   ├── os_operations.py
│   └── __init__.py
├── templates/             # Web interface
│   ├── index.html
│   ├── style.css
│   └── script.js
├── config/                # All configuration files
│   ├── data_sources.json      # Document sources
│   ├── orchestrator_config.json # Service configuration
│   └── sample_queries.json     # Sample queries by domain
├── orchestrator.py        # Main orchestrator
├── rag_pipeline.py       # RAG implementation
├── web_interface.py      # Flask web app
└── main.py               # Entry point
```

## ⚙️ Configuration

All configuration files are now centralized in the `config/` directory:

### Data Sources
Edit `config/data_sources.json`:
```json
{
  "urls": ["https://example.com"],
  "file_paths": ["documents/paper.pdf"]
}
```

### Service Configuration
Edit `config/orchestrator_config.json` (auto-generated with defaults):
```json
{
  "rag": {
    "enabled": true,
    "embedding_model": "all-MiniLM-L6-v2",
    "chunk_size": 200,
    "chunk_overlap": 50,
    "data_dir": "data"
  },
  "web_search": {
    "enabled": true,
    "provider": "duckduckgo"
  },
  "image_generation": {
    "enabled": true,
    "provider": "local"
  },
  "os_operations": {
    "enabled": true,
    "allow_file_operations": true,
    "allow_process_execution": false
  }
}
```

### Sample Queries
Add domain-specific queries in `config/sample_queries.json`:
```json
{
  "general": ["What are the main concepts?", "Summarize this content"],
  "technology": ["What technologies are discussed?", "Explain the technical concepts"],
  "science": ["What experiments are mentioned?", "What theories are explained?"]
}
```

## 🧹 Maintenance

### Clean Python Cache Files
```bash
# Unix/Linux/macOS
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete

# Windows
for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"
del /s /q *.pyc 2>nul
```

### Reset Data
```bash
rm -rf data/ logs/ output/  # Unix/Linux/macOS
rmdir /s data logs output   # Windows
```

## 🔧 Development

### Adding New Services
1. Create service class in `services/`
2. Implement required interface methods
3. Register in `orchestrator.py` `_init_services()`
4. Add configuration options to `config/orchestrator_config.json`

### Configuration Management
- All configs are JSON files in `config/` directory
- Auto-generated with sensible defaults
- Environment-specific overrides supported

## 🤝 Contributing

1. Keep code modular and well-documented
2. Follow existing patterns for new services
3. Use the unified `config/` directory for all configuration
4. Use type hints consistently
5. Test with different LLM backends

## 📝 License

This project is licensed under the MIT License.
