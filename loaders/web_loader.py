import time
import hashlib
import requests
from typing import List, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import numpy as np
from tqdm import tqdm

from core.interfaces import DataLoader, Document
from core.config import ScrapingConfig

class WebDataLoader(DataLoader):
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.headers = {
            'User-Agent': config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        self.last_request_time = 0

    def load(self, urls: List[str]) -> List[Document]:
        documents = []
        for url in tqdm(urls, desc="Loading web content"):
            doc = self._scrape_url(url)
            if doc:
                documents.append(doc)
        return documents

    def _scrape_url(self, url: str) -> Optional[Document]:
        html = self._get_html(url)
        if not html:
            return None

        soup = BeautifulSoup(html, 'html.parser')
        content = self._extract_content(soup)
        title = soup.title.string if soup.title else ''
        
        doc_id = hashlib.md5(url.encode()).hexdigest()
        metadata = {
            'title': title,
            'domain': urlparse(url).netloc,
            'url': url
        }

        return Document(
            id=doc_id,
            content=content,
            metadata=metadata,
            source=url
        )

    def _get_html(self, url: str) -> Optional[str]:
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.config.delay_between_requests:
            time.sleep(self.config.delay_between_requests - time_since_last)

        for attempt in range(self.config.retry_count):
            try:
                response = requests.get(url, headers=self.headers, timeout=self.config.timeout)
                response.raise_for_status()
                self.last_request_time = time.time()
                return response.text
            except Exception as e:
                if attempt < self.config.retry_count - 1:
                    wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                    time.sleep(wait_time)
        return None

    def _extract_content(self, soup: BeautifulSoup) -> str:
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()

        # Try to find main content
        main_content = None
        for container_id in ['content', 'main', 'main-content', 'article', 'post']:
            main_content = soup.find(id=container_id)
            if main_content:
                break

        if not main_content:
            main_content = soup.body

        if main_content:
            paragraphs = []
            for p in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                text = p.get_text().strip()
                if text:
                    paragraphs.append(text)
            return '\n\n'.join(paragraphs)
        else:
            return soup.get_text(separator='\n', strip=True)
