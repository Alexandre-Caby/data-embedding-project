import os
import hashlib
from typing import List, Optional
from tqdm import tqdm

from core.interfaces import DataLoader, Document

class FileDataLoader(DataLoader):
    def load(self, file_paths: List[str]) -> List[Document]:
        documents = []
        for path in tqdm(file_paths, desc="Loading files"):
            doc = self._load_file(path)
            if doc:
                documents.append(doc)
        return documents

    def _load_file(self, file_path: str) -> Optional[Document]:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            doc_id = hashlib.md5(file_path.encode()).hexdigest()
            metadata = {
                'filename': os.path.basename(file_path),
                'file_path': file_path,
                'file_size': os.path.getsize(file_path)
            }

            return Document(
                id=doc_id,
                content=content,
                metadata=metadata,
                source=file_path
            )
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None
