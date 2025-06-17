import torch
from typing import List
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from core.interfaces import EmbeddingModel
from core.config import EmbeddingConfig

class SentenceTransformerEmbedding(EmbeddingModel):
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.device = self._get_device()
        
        print(f"Loading embedding model: {config.model_name}")
        self.model = SentenceTransformer(config.model_name)
        
        if self.device != "cpu":
            self.model = self.model.to(self.device)
        else:
            print("Model loaded (CPU mode)")

    def _get_device(self) -> str:
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def encode(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.config.batch_size), desc="Creating embeddings"):
            batch = texts[i:i + self.config.batch_size]
            embeddings = self.model.encode(batch, convert_to_tensor=False)
            all_embeddings.extend(embeddings.tolist())
        
        return all_embeddings

    def encode_single(self, text: str) -> List[float]:
        with torch.no_grad():
            embedding = self.model.encode([text], convert_to_tensor=False)
            return embedding[0].tolist()
    
    def get_dimension(self):
        """Get the dimension of the embeddings produced by the model."""
        return self.model.get_sentence_embedding_dimension()
