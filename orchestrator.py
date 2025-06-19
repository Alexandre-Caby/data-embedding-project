import os
import sys
from typing import Dict, Any, List, Optional, Union
import json
import logging
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

# Import your existing components
from rag_pipeline import create_rag_pipeline, RAGPipeline
from services.image_generation import HuggingFaceImageService, ImageGenerationRequest
from services.web_search import WebSearchService
from services.os_operations import OSOperationsService

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
    
    def process_query(self, query: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a user query using appropriate services.
        """
        options = options or {}
        results = {"success": True, "query": query, "results": {}}
        
        # Determine if this is a command for a specific service
        service_commands = {
            "generate image": "image_generation",
            "create image": "image_generation",
            "make image": "image_generation",
            "search web": "web_search",
            "search for": "web_search",
            "find online": "web_search",
            "web search": "web_search",
            "file operations": "os_operations",
            "system info": "os_operations",
            "system information": "os_operations"
        }
        
        target_service = None
        query_lower = query.lower()

        # Check for specific service commands
        for command, service in service_commands.items():
            if command in query_lower and service in self.services:
                target_service = service
                break

        # Process with the specific service if identified
        if target_service:
            try:
                if target_service == "image_generation":
                    # Extract the image description
                    prompt = query
                    for prefix in ["generate image", "create image", "make image"]:
                        if prompt.lower().startswith(prefix):
                            prompt = prompt[len(prefix):].strip()
                            break

                    if not prompt:
                        prompt = "A creative and interesting image"

                    image_result = self.services["image_generation"].generate_image(prompt)
                    results["results"]["image_generation"] = image_result

                elif target_service == "web_search":
                    # Extract the search query
                    search_query = query
                    for prefix in ["search web for", "search web", "search for", "find online", "web search"]:
                        if search_query.lower().startswith(prefix):
                            search_query = search_query[len(prefix):].strip()
                            break

                    search_query = search_query.lstrip(':').strip()

                    if not search_query:
                        search_query = "latest news"

                    search_results = self.services["web_search"].search(search_query)
                    results["results"]["web_search"] = search_results

                elif target_service == "os_operations":
                    # Process OS operations command
                    os_result = self.services["os_operations"].process_command(query)
                    results["results"]["os_operations"] = os_result

            except Exception as e:
                self.logger.error(f"Error in {target_service} service: {e}")
                results["results"][target_service] = {"error": str(e)}
                results["success"] = False

        # For general queries or if specific service processing failed, use RAG
        # But only if no specific service was successfully invoked
        if "rag" in self.services and (not target_service or not results["success"] or not results["results"]):
            try:
                rag_result = self.services["rag"].generate_response(query)
                results["results"]["rag"] = rag_result
                # If RAG was the fallback but we had an error, mark as successful if RAG worked
                if not results["success"] and rag_result.get("context"):
                    results["success"] = True
            except Exception as e:
                self.logger.error(f"Error in RAG service: {e}")
                results["results"]["rag"] = {"error": str(e)}
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
            result = self.services["image_generation"].generate_image(prompt, options)
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