import os
import sys
from typing import Dict, Any, List, Optional, Union
import json
import logging
from concurrent.futures import ThreadPoolExecutor

# Import your existing components
from rag_pipeline import create_rag_pipeline, RAGPipeline
from services.image_generation import HuggingFaceImageService, ImageGenerationRequest
from services.web_search import WebSearchService
from services.os_operations import OSOperationsService
from core.interfaces import Document

class AIServicesOrchestrator:
    """
    Central orchestrator for coordinating various AI services:
    - RAG (Retrieval Augmented Generation)
    - Web scraping
    - Image generation
    - Web search
    - OS operations
    """
    
    def __init__(self, config_path: str = "config/orchestrator_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Initialize services
        self.services = {}
        self._init_services()
        
        self.temp_web_documents = []  # Store temporary web documents
        self.temp_web_query = ""      # Store the query that generated these documents
        
        self.logger.info("Orchestrator initialized with services: " + 
                        ", ".join(self.services.keys()))
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                # Create default config
                default_config = {
                    "rag": {
                        "enabled": True,
                        "embedding_model": "all-MiniLM-L6-v2",
                        "chunk_size": 200,
                        "chunk_overlap": 50,
                        "data_dir": "data"
                    },
                    "image_generation": {
                        "enabled": True,
                        "provider": "local", 
                        "model": "stable-diffusion-xl-base-1.0",
                        "api_key": ""
                    },
                    "web_search": {
                        "enabled": True,
                        "provider": "duckduckgo",
                        "api_key": "",
                        "results_limit": 5
                    },
                    "os_operations": {
                        "enabled": True,
                        "allow_file_operations": True,
                        "allow_process_execution": False,
                        "restricted_paths": []
                    }
                }
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                
                # Save default config
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                
                return default_config
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            return {
                "rag": {"enabled": True},
                "image_generation": {"enabled": False},
                "web_search": {"enabled": False},
                "os_operations": {"enabled": True}
            }
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the orchestrator."""
        logger = logging.getLogger("orchestrator")
        logger.setLevel(logging.INFO)
        
        # Create handler if not exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
            # Add file handler
            os.makedirs("logs", exist_ok=True)
            file_handler = logging.FileHandler("logs/orchestrator.log")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _init_services(self):
        """Initialize all enabled services."""
        # Initialize RAG
        if self.config.get("rag", {}).get("enabled", True):
            try:
                rag_config = self.config.get("rag", {})
                self.services["rag"] = create_rag_pipeline(
                    embedding_model=rag_config.get("embedding_model", "all-MiniLM-L6-v2"),
                    chunk_size=rag_config.get("chunk_size", 200),
                    chunk_overlap=rag_config.get("chunk_overlap", 50),
                    output_dir=rag_config.get("data_dir", "data")
                )
                
                # Load existing data if available
                data_dir = rag_config.get("data_dir", "data")
                if os.path.exists(data_dir):
                    self.services["rag"].load(data_dir)
                    self.logger.info(f"Loaded RAG data from {data_dir}")
            except Exception as e:
                self.logger.error(f"Failed to initialize RAG service: {e}")

        # Image Generation via HF Inference (free tier)
        img_conf = self.config.get("image_generation", {})
        if img_conf.get("enabled", False):
            api_key = img_conf.get("api_key", "").strip()
            if not api_key:
                self.logger.warning("Pas de clé HF, désactivation du service image_generation")
            else:
                try:
                    svc = HuggingFaceImageService(api_key=api_key)
                    self.services["image_generation"] = svc
                    self.logger.info("Service image_generation initialisé via HF Inference (free tier)")
                except Exception as e:
                    self.logger.error(f"Échec de l'init du service image_generation : {e}")

        # Web Search
        ws_conf = self.config.get("web_search", {})
        if ws_conf.get("enabled", False):
            try:
                search_config = self.config.get("web_search", {})
                self.services["web_search"] = WebSearchService(
                    provider=search_config.get("provider", "custom"),
                    api_key=search_config.get("api_key", ""),
                    results_limit=search_config.get("results_limit", 5)
                )
                self.logger.info(f"Initialized Web Search service with provider: {search_config.get('provider')}")
            except Exception as e:
                self.logger.error(f"Failed to initialize Web Search service: {e}")
        
        # Initialize OS Operations
        if self.config.get("os_operations", {}).get("enabled", True):
            try:
                os_config = self.config.get("os_operations", {})
                self.services["os_operations"] = OSOperationsService(
                    allow_file_operations=os_config.get("allow_file_operations", True),
                    allow_process_execution=os_config.get("allow_process_execution", False),
                    restricted_paths=os_config.get("restricted_paths", [])
                )
                self.logger.info("Initialized OS Operations service")
            except Exception as e:
                self.logger.error(f"Failed to initialize OS Operations service: {e}")
    
    def get_service(self, service_name: str) -> Any:
        """Get a specific service by name."""
        if service_name not in self.services:
            self.logger.warning(f"Service '{service_name}' not found or not enabled")
            return None
        return self.services[service_name]
    
    def process_query(self, query: str, quality: str = 'high') -> Dict[str, Any]:
        """Process a user query using appropriate services with high quality by default."""
        results = {"success": True, "query": query, "results": {}}
        
        # Check for specialized service indicators in the query
        target_service = None
        query_lower = query.lower().strip()

        # Image generation
        if query_lower.startswith(('generate image', 'create image', 'draw')):
            target_service = "image_generation"
            # Extract the prompt
            for prefix in ["generate image:", "generate image", "create image:", "create image", "draw:"]:
                if query.lower().startswith(prefix):
                    prompt = query[len(prefix):].strip()
                    if prompt:
                        if "image_generation" in self.services:
                            try:
                                # Correction : toujours passer par self.generate_image pour garantir le bon type d'objet
                                image_result = self.generate_image(prompt)
                                results["results"]["image_generation"] = image_result["data"]
                                results["success"] = image_result["success"]
                            except Exception as e:
                                self.logger.error(f"Image generation error: {e}")
                                results["results"]["image_generation"] = {"error": str(e)}
                                results["success"] = False

        # Web search
        elif query_lower.startswith(('search web', 'search for')):
            target_service = "web_search"
            # Extract the search query
            for prefix in ["search web for:", "search web for", "search web:", "search web", "search for:", "search for"]:
                if query_lower.startswith(prefix):
                    search_query = query[len(prefix):].strip()
                    if search_query:
                        if "web_search" in self.services:
                            try:
                                web_results = self.services["web_search"].search(search_query)
                                
                                # Process web results to extract knowledge
                                self.temp_web_documents = self.process_web_search_results(web_results)
                                self.temp_web_query = search_query
                                
                                # Create context from web documents
                                web_context = ""
                                for doc in self.temp_web_documents:
                                    web_context += f"\n\n--- From {doc.metadata.get('title', 'Web')} ---\n"
                                    web_context += doc.content[:1500] + "..."
                                
                                # Generate a response with web context if RAG is available
                                if "rag" in self.services and web_context:
                                    rag_result = self.services["rag"]._handle_complex_query(
                                        query=search_query, 
                                        web_context=web_context,
                                        top_k=3
                                    )
                                    results["results"]["rag"] = rag_result
                                
                                results["results"]["web_search"] = web_results
                                results["success"] = True
                            except Exception as e:
                                self.logger.error(f"Web search error: {e}")
                                results["results"]["web_search"] = {"error": str(e)}
                                results["success"] = False

        # System operations
        elif any(keyword in query_lower for keyword in [
            'system info', 'system information', 'computer info',
            'disk space', 'storage info', 'memory usage', 'ram usage',
            'cpu usage', 'processor usage', 'list files', 'list directory',
            'read file', 'show content', 'display file', 
            
            # Enhanced file creation keywords
            'write file', 'create file', 'delete file', 'remove file', 'copy file',
            'create a file', 'write a file', 'make a file', 'make file',
            'new file', 'create a text file', 'create text file', 'hello.txt',
            'text file with', 'create a file with', 'write hello world',
            
            # French patterns
            "créer fichier", "créer un fichier", "écrire fichier", 
            "nouveau fichier", "sauvegarder fichier", "créer fichier texte",
            "faire un fichier", "fichier avec", "avec", "contenant",
            "bonjour monde"
        ]):
            target_service = "os_operations"
            if "os_operations" in self.services:
                try:
                    self.logger.info(f"Routing command to OS operations: {query}")
                    os_result = self.services["os_operations"].process_command(query)
                    results["results"]["os_operations"] = os_result
                    results["success"] = True
                    return results  # Return immediately for OS operations
                except Exception as e:
                    self.logger.error(f"OS operations error: {e}")
                    results["results"]["os_operations"] = {"error": str(e)}
                    results["success"] = False

        # For general queries or if specific service processing failed, use RAG with high quality
        if "rag" in self.services and (not target_service or not results["success"] or not results["results"]):
            try:
                # Set top_k parameter based on quality
                top_k = 8 if quality == 'high' else 5

                if quality == 'high':
                    # Use the generate_response method directly for high quality
                    rag_result = self.services["rag"].generate_response(query, top_k=top_k)
                else:
                    # Use the fast response method for lower quality/faster response
                    rag_result = self.services["rag"].generate_response_fast(query, top_k=top_k)

                results["results"]["rag"] = rag_result
                # If RAG was the fallback but we had an error, mark as successful if RAG worked
                if not results["success"] and rag_result.get("context"):
                    results["success"] = True
            except Exception as e:
                self.logger.error(f"Error in RAG service: {e}")
                results["results"]["rag"] = {"error": str(e), "context": f"Error processing: {str(e)}"}
                results["success"] = False
        
        return results
    
    def ingest_data(self, urls: List[str] = None, file_paths: List[str] = None) -> Dict[str, Any]:
        """
        Ingest data into the RAG pipeline.
        
        Args:
            urls: List of URLs to scrape
            file_paths: List of file paths to process
        
        Returns:
            Dict containing ingestion results
        """
        result = {"success": True, "message": "", "chunks": 0}
        
        if "rag" not in self.services:
            result["success"] = False
            result["message"] = "RAG service not available"
            return result
        
        try:
            chunks = self.services["rag"].ingest_data(urls=urls, file_paths=file_paths)
            result["chunks"] = len(chunks)
            result["message"] = f"Successfully ingested {len(chunks)} chunks"
            
            # Save the pipeline state
            self.services["rag"].save()
        except Exception as e:
            result["success"] = False
            result["message"] = f"Error ingesting data: {str(e)}"
            self.logger.error(f"Data ingestion error: {e}")
        
        return result
    
    def generate_image(self, prompt: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate an image from a text prompt.
        
        Args:
            prompt: Text description of the image to generate
            options: Additional options for image generation
        
        Returns:
            Dict containing the generated image information
        """
        if "image_generation" not in self.services:
            return {
                "success": False,
                "message": "Image generation service not available",
                "data": None
            }
        
        try:
            # Correction: toujours passer un objet ImageGenerationRequest
            if options is None:
                options = {}
            # Crée l'objet request à partir du prompt et des options
            request = ImageGenerationRequest(
                prompt=prompt,
                negative_prompt=options.get("negative_prompt"),
                width=options.get("width", 1024),
                height=options.get("height", 1024),
                num_inference_steps=options.get("num_inference_steps", 30),
                guidance_scale=options.get("guidance_scale", 7.5),
                seed=options.get("seed")
            )
            result = self.services["image_generation"].generate_image(request)
            return {
                "success": True,
                "message": "Image generated successfully",
                "data": result
            }
        except Exception as e:
            self.logger.error(f"Image generation error: {e}")
            return {
                "success": False,
                "message": f"Error generating image: {str(e)}",
                "data": None
            }
    
    def search_web(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            num_results: Number of results to return
        
        Returns:
            Dict containing search results
        """
        if "web_search" not in self.services:
            return {
                "success": False,
                "message": "Web search service not available",
                "results": []
            }
        
        try:
            results = self.services["web_search"].search(query, num_results)
            return {
                "success": True,
                "message": f"Found {len(results)} results",
                "results": results
            }
        except Exception as e:
            self.logger.error(f"Web search error: {e}")
            return {
                "success": False,
                "message": f"Error searching web: {str(e)}",
                "results": []
            }
    
    def execute_os_command(self, command: str) -> Dict[str, Any]:
        """
        Execute an OS operation.
        
        Args:
            command: Command to execute
        
        Returns:
            Dict containing execution result
        """
        if "os_operations" not in self.services:
            return {
                "success": False,
                "message": "OS operations service not available",
                "result": None
            }
        
        try:
            result = self.services["os_operations"].process_command(command)
            return {
                "success": True,
                "message": "Command executed successfully",
                "result": result
            }
        except Exception as e:
            self.logger.error(f"OS operation error: {e}")
            return {
                "success": False,
                "message": f"Error executing command: {str(e)}",
                "result": None
            }
    
    def reset_services(self) -> Dict[str, bool]:
        """Reset all stateful services."""
        results = {}
        
        # Reset RAG conversation history
        if "rag" in self.services:
            try:
                self.services["rag"].reset_conversation()
                results["rag"] = True
            except Exception as e:
                self.logger.error(f"Error resetting RAG service: {e}")
                results["rag"] = False
        
        # Add resets for other stateful services as needed
        
        return results
    
    def shutdown(self):
        """Shutdown all services properly."""
        for service_name, service in self.services.items():
            try:
                if hasattr(service, "save"):
                    service.save()
                if hasattr(service, "shutdown"):
                    service.shutdown()
                    self.logger.info(f"Service {service_name} shut down properly")
            except Exception as e:
                self.logger.error(f"Error shutting down {service_name}: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from all services"""
        stats = {}

        if "rag" in self.services:
            try:
                stats["rag"] = self.services["rag"].get_performance_stats()
            except Exception as e:
                self.logger.error(f"Error getting RAG stats: {e}")
                stats["rag"] = {"error": str(e)}
        
        return stats
    
    def process_web_search_results(self, search_results: List[Dict[str, Any]]) -> List[Document]:
        """
        Process web search results by fetching and converting them to Document objects.
        """
        if not search_results:
            return []
        
        self.logger.info(f"Processing {len(search_results)} web search results")
        
        # Extract URLs from search results and clean them
        urls = []
        for result in search_results:
            url = result.get('url')
            if url and isinstance(url, str):
                # Clean the URL - remove whitespace and newlines
                cleaned_url = url.strip().replace('\n', '').replace(' ', '')
                if cleaned_url.startswith(('http://', 'https://')):
                    urls.append(cleaned_url)
        
        if not urls:
            self.logger.warning("No valid URLs found in search results")
            return []
        
        # Log the URLs we're trying to scrape
        self.logger.info(f"Cleaned URLs to scrape: {urls[:3]}")
        
        # Use WebDataLoader to scrape content from URLs
        try:
            from loaders.web_loader import WebDataLoader
            from core.config import ScrapingConfig
            
            # Create a config with reasonable scraping settings
            scraping_config = ScrapingConfig(
                delay_between_requests=1.0,
                timeout=15,
                retry_count=2
            )
            
            web_loader = WebDataLoader(scraping_config)
            documents = web_loader.load(urls[:3])  # Try fewer URLs first
            
            # Log each document's content length
            for i, doc in enumerate(documents):
                self.logger.info(f"Document {i+1}: {len(doc.content)} characters from {doc.metadata.get('url', 'unknown')}")
            
            self.logger.info(f"Successfully scraped {len(documents)} documents from web search results")
            
            self.logger.info(f"Ingesting {len(documents)} web documents into RAG...")
            if "rag" in self.services and documents:
                try:
                    # On récupère la liste des URLs des documents web
                    urls_to_ingest = [doc.metadata.get('url') for doc in documents if doc.metadata.get('url')]
                    self.services["rag"].ingest_data(
                        urls=urls_to_ingest,
                        file_paths=None
                    )
                    self.logger.info(f"Ingested {len(urls_to_ingest)} web documents into RAG successfully")
                except Exception as e:
                    self.logger.error(f"Error ingesting web documents into RAG: {e}")
                    
            return documents
        except Exception as e:
            self.logger.error(f"Error processing web search results: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []
