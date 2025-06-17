from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    source: str

@dataclass
class Chunk:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    document_id: Optional[str] = None

@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    rank: int

class DataLoader(ABC):
    @abstractmethod
    def load(self, sources: List[str]) -> List[Document]:
        pass

class TextChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        pass

class EmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        pass
    
    @abstractmethod
    def encode_single(self, text: str) -> List[float]:
        pass

class VectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        pass
