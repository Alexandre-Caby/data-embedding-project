import requests
import logging
import json
import time
import urllib.parse
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

class WebSearchService:
    """Service for performing web searches."""
    
    def __init__(self, provider: str = "custom", api_key: str = "", results_limit: int = 5):
        self.provider = provider.lower()
        self.api_key = api_key
        self.results_limit = results_limit
        self.logger = logging.getLogger("orchestrator.web_search")
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def search(self, query: str, num_results: int = None) -> List[Dict[str, Any]]:
        """
        Search the web for the given query.
        
        Args:
            query: Search query
            num_results: Number of results to return (overrides instance setting)
            
        Returns:
            List of search results with title, url, and snippet
        """
        if num_results is None:
            num_results = self.results_limit
        
        self.logger.info(f"Searching for '{query}' using {self.provider} provider")
        
        try:
            if self.provider == "google":
                return self._search_google(query, num_results)
            elif self.provider == "bing":
                return self._search_bing(query, num_results)
            elif self.provider == "serp":
                return self._search_serpapi(query, num_results)
            elif self.provider == "duckduckgo":
                return self._search_duckduckgo(query, num_results)
            elif self.provider == "custom":
                return self._search_custom(query, num_results)
            else:
                self.logger.warning(f"Unknown provider '{self.provider}'. Falling back to custom search.")
                return self._search_custom(query, num_results)
        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return []
    
    def _search_google(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API."""
        if not self.api_key:
            self.logger.error("No API key provided for Google Custom Search")
            return []
        
        # Google requires a cx parameter (search engine ID)
        cx = self.api_key.split(":")[1] if ":" in self.api_key else ""
        api_key = self.api_key.split(":")[0] if ":" in self.api_key else self.api_key
        
        if not cx:
            self.logger.error("No search engine ID (cx) provided for Google Custom Search")
            return []
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": api_key,
            "cx": cx,
            "q": query,
            "num": min(num_results, 10)  # Google API limit is 10 per request
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if "items" in data:
            for item in data["items"]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "google"
                })
        
        return results
    
    def _search_bing(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using Bing Search API."""
        if not self.api_key:
            self.logger.error("No API key provided for Bing Search")
            return []
        
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key
        }
        params = {
            "q": query,
            "count": num_results,
            "responseFilter": "Webpages"
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if "webPages" in data and "value" in data["webPages"]:
            for item in data["webPages"]["value"]:
                results.append({
                    "title": item.get("name", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "bing"
                })
        
        return results
    
    def _search_serpapi(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using SerpAPI."""
        if not self.api_key:
            self.logger.error("No API key provided for SerpAPI")
            return []
        
        url = "https://serpapi.com/search"
        params = {
            "api_key": self.api_key,
            "q": query,
            "num": num_results,
            "engine": "google"
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if "organic_results" in data:
            for item in data["organic_results"]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "serpapi"
                })
        
        return results
    
    def _search_duckduckgo(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo HTML endpoint."""
        encoded = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        try:
            resp = requests.get(url, headers=self.headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for element in soup.select(".result")[:num_results]:
                title_el = element.select_one(".result__title")
                url_el   = element.select_one(".result__url")
                snippet_el = element.select_one(".result__snippet")
                if title_el and url_el:
                    link = url_el.get_text()
                    if not link.startswith(("http://", "https://")):
                        link = "https://" + link
                    results.append({
                        "title":   title_el.get_text(),
                        "url":     link,
                        "snippet": snippet_el.get_text() if snippet_el else "",
                        "source":  "duckduckgo"
                    })
            return results
        except Exception as e:
            self.logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    def _search_custom(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """
        Custom search implementation that scrapes search results.
        This is a fallback method that doesn't require API keys.
        """
        self.logger.info("Using custom search implementation")
        
        # Format query for URL
        encoded_query = urllib.parse.quote(query)
        url = f"https://www.google.com/search?q={encoded_query}&num={num_results}"
        
        try:
            # Make request with delay to avoid rate limiting
            time.sleep(1)
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")
            search_results = []
            
            # Extract results - this selector may need updates as Google changes their HTML
            result_elements = soup.select("div.g")
            
            for element in result_elements[:num_results]:
                try:
                    title_element = element.select_one("h3")
                    link_element = element.select_one("a")
                    snippet_element = element.select_one("div.VwiC3b")
                    
                    if title_element and link_element and "href" in link_element.attrs:
                        title = title_element.get_text()
                        link = link_element["href"]
                        snippet = snippet_element.get_text() if snippet_element else ""
                        
                        # Filter out non-web results
                        if link.startswith("http") and not link.startswith("https://webcache.googleusercontent.com"):
                            search_results.append({
                                "title": title,
                                "url": link,
                                "snippet": snippet,
                                "source": "custom_search"
                            })
                except Exception as e:
                    self.logger.warning(f"Error parsing search result: {e}")
            
            return search_results
        
        except Exception as e:
            self.logger.error(f"Custom search error: {e}")
            
            # Try a simpler fallback method
            return self._search_fallback(query, num_results)
    
    def _search_fallback(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Ultra simple fallback search."""
        self.logger.info("Using fallback search method")
        
        # Use DuckDuckGo as fallback
        encoded_query = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            search_results = []
            
            # Extract results
            result_elements = soup.select(".result")
            
            for element in result_elements[:num_results]:
                try:
                    title_element = element.select_one(".result__title")
                    link_element = element.select_one(".result__url")
                    snippet_element = element.select_one(".result__snippet")
                    
                    if title_element and link_element:
                        # Extract actual link
                        link = link_element.get_text()
                        if not link.startswith(('http://', 'https://')):
                            link = f"https://{link}"
                        
                        search_results.append({
                            "title": title_element.get_text(),
                            "url": link,
                            "snippet": snippet_element.get_text() if snippet_element else "",
                            "source": "duckduckgo"
                        })
                except Exception as e:
                    self.logger.warning(f"Error parsing fallback search result: {e}")
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Fallback search error: {e}")
            return []