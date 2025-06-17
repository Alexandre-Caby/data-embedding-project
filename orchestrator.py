import os
import json
import logging
from typing import Any, Dict, List
from concurrent.futures import ThreadPoolExecutor

from rag_pipeline import create_rag_pipeline, RAGPipeline
from services.image_generation import HuggingFaceImageService, ImageGenerationRequest
from services.web_search import WebSearchService
from services.os_operations import OSOperationsService


class AIServicesOrchestrator:
    """
    Central orchestrator for coordinating various AI services:
    - RAG (Retrieval Augmented Generation)
    - Image generation via HF Inference (free tier)
    - Web search
    - OS operations
    """

    def __init__(self, config_path: str = "config/orchestrator_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.services: Dict[str, Any] = {}
        self._init_services()
        self.logger.info("Orchestrator initialized with services: " + ", ".join(self.services.keys()))

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file, or create defaults."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)

        default = {
            "rag": {
                "enabled": True,
                "embedding_model": "all-MiniLM-L6-v2",
                "chunk_size": 200,
                "chunk_overlap": 50,
                "data_dir": "data"
            },
            "image_generation": {
                "enabled": True,
                "provider": "hf-inference",
                "api_key": "",  # à remplir
                "model": "stable-diffusion-v1-5",
                "available_models": [
                    "stable-diffusion-v1-5",
                    "stable-diffusion-xl-base-1.0",
                    "stable-diffusion-3-medium"
                ]
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
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(default, f, indent=2)
        return default

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("orchestrator")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            ch = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            ch.setFormatter(fmt)
            logger.addHandler(ch)
            os.makedirs("logs", exist_ok=True)
            fh = logging.FileHandler("logs/orchestrator.log")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        return logger

    def _init_services(self):
        # RAG
        rag_conf = self.config.get("rag", {})
        if rag_conf.get("enabled", False):
            try:
                rag = create_rag_pipeline(
                    embedding_model=rag_conf.get("embedding_model"),
                    chunk_size=rag_conf.get("chunk_size"),
                    chunk_overlap=rag_conf.get("chunk_overlap"),
                    output_dir=rag_conf.get("data_dir")
                )
                if os.path.exists(rag_conf.get("data_dir", "")):
                    rag.load(rag_conf["data_dir"])
                    self.logger.info(f"Loaded RAG data from {rag_conf['data_dir']}")
                self.services["rag"] = rag
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
                ws = WebSearchService(
                    provider=ws_conf.get("provider"),
                    api_key=ws_conf.get("api_key"),
                    results_limit=ws_conf.get("results_limit")
                )
                self.services["web_search"] = ws
                self.logger.info(f"Initialized Web Search service: {ws_conf.get('provider')}")
            except Exception as e:
                self.logger.error(f"Failed to init Web Search service: {e}")

        # OS Operations
        os_conf = self.config.get("os_operations", {})
        if os_conf.get("enabled", False):
            try:
                os_op = OSOperationsService(
                    allow_file_operations=os_conf.get("allow_file_operations"),
                    allow_process_execution=os_conf.get("allow_process_execution"),
                    restricted_paths=os_conf.get("restricted_paths")
                )
                self.services["os_operations"] = os_op
                self.logger.info("Initialized OS Operations service")
            except Exception as e:
                self.logger.error(f"Failed to init OS Operations service: {e}")

    def process_query(self, query: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        options = options or {}
        response = {"success": True, "query": query, "results": {}}
        ql = query.lower()

        # Map keywords to services
        cmd_map = {
            "generate image": "image_generation",
            "search web": "web_search",
            "system info": "os_operations"
        }
        target = next((s for k, s in cmd_map.items() if k in ql and s in self.services), None)

        # Invoke specific service
        if target == "image_generation":
            prompt = query
            for prefix in ["generate image", "create image", "make image"]:
                if prompt.lower().startswith(prefix):
                    prompt = prompt[len(prefix):].strip()
            if not prompt:
                prompt = "A creative and interesting image"
            result = self.services["image_generation"].generate_image_simple(prompt, options)
            response["results"]["image_generation"] = result

        elif target == "web_search":
            sq = query
            for prefix in ["search web for", "search web"]:
                if sq.lower().startswith(prefix):
                    sq = sq[len(prefix):].strip()
            if not sq:
                sq = "latest news"
            response["results"]["web_search"] = self.services["web_search"].search(sq)

        elif target == "os_operations":
            response["results"]["os_operations"] = self.services["os_operations"].process_command(query)

        # Fallback to RAG if needed
        if "rag" in self.services and (not target or not response["results"]):
            try:
                rag_out = self.services["rag"].generate_response(query)
                response["results"]["rag"] = rag_out
            except Exception as e:
                self.logger.error(f"RAG error: {e}")
                response["success"] = False

        return response

    def reset_services(self) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        if "rag" in self.services:
            try:
                self.services["rag"].reset_conversation()
                results["rag"] = True
            except:
                results["rag"] = False
        return results

    def shutdown(self):
        for name, svc in self.services.items():
            try:
                if hasattr(svc, "save"):
                    svc.save()
                if hasattr(svc, "shutdown"):
                    svc.shutdown()
                self.logger.info(f"Service {name} shut down")
            except Exception as e:
                self.logger.error(f"Error shutting down {name}: {e}")
