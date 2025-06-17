from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class ScrapingConfig:
    timeout: int = 10
    retry_count: int = 3
    delay_between_requests: float = 1.0
    user_agent: str = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124'

@dataclass
class ChunkingConfig:
    words_per_chunk: int = 200
    overlap: int = 50
    strategy: str = "sentence"

@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    device: str = "auto"
    normalize_embeddings: bool = True

@dataclass
class VectorStoreConfig:
    storage_type: str = "memory"
    persist_directory: Optional[str] = None
    index_type: str = "flat"

@dataclass
class RAGConfig:
    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    output_directory: str = "data"
