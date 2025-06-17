import re
import hashlib
from typing import List
import nltk
try:
    nltk.download('punkt', quiet=True)
except:
    pass

from core.interfaces import TextChunker, Document, Chunk
from core.config import ChunkingConfig

class SentenceChunker(TextChunker):
    def __init__(self, config: ChunkingConfig):
        self.config = config

    def chunk(self, document: Document) -> List[Chunk]:
        text = self._preprocess_text(document.content)
        chunks_text = self._chunk_by_sentences(text)
        
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            chunk_id = hashlib.md5(f"{document.id}_{i}_{chunk_text}".encode()).hexdigest()
            
            chunk = Chunk(
                id=chunk_id,
                content=chunk_text,
                metadata={
                    'chunk_index': i,
                    'word_count': len(re.findall(r'\b\w+\b', chunk_text)),
                    'source_document': document.source,
                    'document_title': document.metadata.get('title', ''),
                },
                document_id=document.id
            )
            chunks.append(chunk)
        
        return chunks

    def _preprocess_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _chunk_by_sentences(self, text: str) -> List[str]:
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            sentence_words = len(re.findall(r'\b\w+\b', sentence))

            if current_word_count + sentence_words > self.config.words_per_chunk:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Overlap handling
                    overlap_start = max(0, len(current_chunk) - self.config.overlap)
                    current_chunk = current_chunk[overlap_start:] + [sentence]
                    current_word_count = sum(len(re.findall(r'\b\w+\b', s)) for s in current_chunk)
                else:
                    chunks.append(sentence)
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

class WordChunker(TextChunker):
    def __init__(self, config: ChunkingConfig):
        self.config = config

    def chunk(self, document: Document) -> List[Chunk]:
        words = document.content.split()
        chunks = []
        
        for i in range(0, len(words), self.config.words_per_chunk - self.config.overlap):
            chunk_words = words[i:i + self.config.words_per_chunk]
            chunk_text = ' '.join(chunk_words)
            
            chunk_id = hashlib.md5(f"{document.id}_{i}_{chunk_text}".encode()).hexdigest()
            
            chunk = Chunk(
                id=chunk_id,
                content=chunk_text,
                metadata={
                    'chunk_index': i // (self.config.words_per_chunk - self.config.overlap),
                    'word_count': len(chunk_words),
                    'source_document': document.source,
                },
                document_id=document.id
            )
            chunks.append(chunk)
        
        return chunks
