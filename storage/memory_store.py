import json
import os
import numpy as np
from typing import List

from core.interfaces import VectorStore, Chunk, SearchResult
from core.config import VectorStoreConfig

class InMemoryVectorStore(VectorStore):
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.chunks: List[Chunk] = []
        self.embeddings: np.ndarray = None

    def add_chunks(self, chunks: List[Chunk]) -> None:
        self.chunks.extend(chunks)
        embeddings = [chunk.embedding for chunk in chunks if chunk.embedding]
        
        if embeddings:
            new_embeddings = np.array(embeddings)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        query_vector = np.array(query_embedding)
        
        # Handle potential division by zero
        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_vector)
        
        if query_norm == 0:
            return []
        
        # Avoid division by zero for chunk embeddings
        valid_indices = norms > 0
        if not np.any(valid_indices):
            return []
        
        # Calculate cosine similarities only for valid embeddings
        similarities = np.zeros(len(self.chunks))
        similarities[valid_indices] = np.dot(
            self.embeddings[valid_indices], query_vector
        ) / (norms[valid_indices] * query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if idx < len(self.chunks) and similarities[idx] > 0:
                result = SearchResult(
                    chunk=self.chunks[idx],
                    score=float(similarities[idx]),
                    rank=rank
                )
                results.append(result)
        
        return results

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        
        # Save chunks metadata (without embeddings)
        chunks_data = []
        for chunk in self.chunks:
            chunk_dict = {
                'id': chunk.id,
                'content': chunk.content,
                'metadata': chunk.metadata,
                'document_id': chunk.document_id
            }
            chunks_data.append(chunk_dict)
        
        with open(os.path.join(path, 'chunks.json'), 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(os.path.join(path, 'embeddings.npy'), self.embeddings)

    def load(self, path: str) -> None:
        chunks_file = os.path.join(path, 'chunks.json')
        if not os.path.exists(chunks_file):
            return
            
        # Load chunks
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        self.chunks = []
        for chunk_dict in chunks_data:
            chunk = Chunk(
                id=chunk_dict['id'],
                content=chunk_dict['content'],
                metadata=chunk_dict['metadata'],
                document_id=chunk_dict.get('document_id')
            )
            self.chunks.append(chunk)
        
        # Load embeddings
        embeddings_path = os.path.join(path, 'embeddings.npy')
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
            
            # Add embeddings back to chunks
            for i, chunk in enumerate(self.chunks):
                if i < len(self.embeddings):
                    chunk.embedding = self.embeddings[i].tolist()
