import os
import hashlib
from typing import List, Optional
from tqdm import tqdm
import PyPDF2

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
            ext = os.path.splitext(file_path)[1].lower()
            content = ""
            if ext in [".txt", ".md", ".cjsonsv", "."]:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            elif ext == ".pdf":
                try:
                    with open(file_path, "rb") as f:
                        reader = PyPDF2.PdfReader(f)
                        content = "\n".join(page.extract_text() or "" for page in reader.pages)
                except Exception as e:
                    print(f"Error reading PDF {file_path}: {e}")
                    return None
            else:
                print(f"Unsupported file type for {file_path}, skipping.")
                return None

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
