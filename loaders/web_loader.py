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
        try:
            html = self._get_html(url)
            if not html:
                print(f"Failed to get HTML for {url}")
                return None

            soup = BeautifulSoup(html, 'html.parser')
            content = self._extract_content(soup)
            
            if not content or len(content.strip()) < 50:
                print(f"No meaningful content extracted from {url}")
                return None
                
            title = soup.title.string if soup.title else ''
            
            doc_id = hashlib.md5(url.encode()).hexdigest()
            metadata = {
                'title': title,
                'domain': urlparse(url).netloc,
                'url': url
            }

            print(f"Successfully scraped {len(content)} characters from {url}")
            return Document(
                id=doc_id,
                content=content,
                metadata=metadata,
                source=url
            )
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def _get_html(self, url: str) -> Optional[str]:
        """Get HTML content from a URL with improved error handling"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            # Clean the URL first
            url = url.strip().replace('\n', '').replace(' ', '')
            
            response = requests.get(
                url, 
                headers=headers, 
                timeout=self.config.timeout,
                allow_redirects=True,
                verify=False  # Skip SSL verification for problematic sites
            )
            response.raise_for_status()
            return response.text
        except requests.exceptions.SSLError as e:
            print(f"SSL Error for {url}: {e}")
            # Try without SSL verification
            try:
                response = requests.get(url, headers=headers, timeout=self.config.timeout, verify=False)
                response.raise_for_status()
                return response.text
            except Exception as e2:
                print(f"Failed even without SSL verification: {e2}")
                return None
        except Exception as e:
            print(f"Error fetching {url}: {e}")
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
