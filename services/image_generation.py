import os
import time
import base64
import io
import requests
import logging
from typing import Dict, Any, Optional, List
from PIL import Image
from dataclasses import dataclass, field
from enum import Enum
from huggingface_hub import InferenceClient

class ServiceStatus(Enum):
    """Status des services"""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class ImageGenerationRequest:
    """Requête de génération d'image"""
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    model: Optional[str] = None


@dataclass
class ImageGenerationResult:
    """Résultat de génération d'image"""
    success: bool
    filepath: Optional[str] = None
    prompt: Optional[str] = None
    error_message: Optional[str] = None
    generation_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HuggingFaceImageService:
    """Service de génération d'images via l'API Hugging Face"""

    # Modèles populaires disponibles sur HF
    AVAILABLE_MODELS = {
        "stable-diffusion-xl": "stabilityai/stable-diffusion-xl-base-1.0",
        "stable-diffusion-2.1": "stabilityai/stable-diffusion-2-1",
        "dalle-mini": "dalle-mini/dalle-mini",
        "openjourney": "prompthero/openjourney-v4",
        "dreamlike-anime": "dreamlike-art/dreamlike-anime-1.0",
        "realistic-vision": "SG161222/Realistic_Vision_V2.0",
        "anything-v4": "andite/anything-v4.0",
        "stable-diffusion-3.5-large": "stabilityai/stable-diffusion-3.5-large"  # ajouté
    }
    def __init__(self, api_key: str, default_model: str = "stable-diffusion-3.5-large"):
        self.api_key = api_key
        self.default_model = default_model
        self.logger = logging.getLogger("orchestrator.huggingface_image")
        self._status = ServiceStatus.IDLE

        # Initialiser le client HF officiel
        self.client = InferenceClient(
            api_key=self.api_key,
            model=self.AVAILABLE_MODELS[self.default_model]
        )

        self.output_dir = os.path.join("output", "images")
        os.makedirs(self.output_dir, exist_ok=True)

    def _validate_api_key(self):
        """Valide la clé API Hugging Face"""
        try:
            test_url = f"{self.base_url}/{self.AVAILABLE_MODELS[self.default_model]}"
            response = requests.get(test_url, headers=self.headers, timeout=10)
            if response.status_code == 401:
                self.logger.error("Invalid Hugging Face API key")
                self._status = ServiceStatus.ERROR
            elif response.status_code == 200:
                self.logger.info("Hugging Face API key validated successfully")
                self._status = ServiceStatus.IDLE
            else:
                self.logger.warning(f"API validation returned status {response.status_code}")
        except Exception as e:
            self.logger.error(f"Error validating API key: {e}")
            self._status = ServiceStatus.ERROR

    @property
    def service_name(self) -> str:
        return "huggingface_image_generation"

    @property
    def status(self) -> ServiceStatus:
        return self._status

    @property
    def supported_models(self) -> List[str]:
        """Retourne la liste des modèles supportés"""
        return list(self.AVAILABLE_MODELS.keys())

    @property
    def max_resolution(self) -> tuple:
        """Résolution maximale supportée"""
        return (1024, 1024)

    @property
    def supported_formats(self) -> List[str]:
        """Formats d'image supportés"""
        return ["PNG", "JPEG", "JPG"]

    def supports_size(self, width: int, height: int) -> bool:
        """Vérifie si la taille est supportée"""
        max_w, max_h = self.max_resolution
        return width <= max_w and height <= max_h and width > 0 and height > 0

    def health_check(self) -> Dict[str, Any]:
        """Vérification de santé du service"""
        try:
            model_url = f"{self.base_url}/{self.AVAILABLE_MODELS[self.default_model]}"
            response = requests.get(model_url, headers=self.headers, timeout=5)

            return {
                "service": self.service_name,
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "api_accessible": response.status_code == 200,
                "default_model": self.default_model,
                "response_time": response.elapsed.total_seconds()
            }
        except Exception as e:
            return {
                "service": self.service_name,
                "status": "unhealthy",
                "error": str(e)
            }

    def _wait_for_model(self, model_url: str, max_wait: int = 60) -> bool:
        """
        Attend que le modèle soit prêt (certains modèles HF ont un temps de démarrage)

        Args:
            model_url: URL du modèle
            max_wait: Temps d'attente maximum en secondes

        Returns:
            True si le modèle est prêt, False sinon
        """
        start_time = time.time()

        while time.time() - start_time < max_wait:
            try:
                response = requests.post(
                    model_url,
                    headers=self.headers,
                    json={"inputs": "test"},
                    timeout=10
                )

                if response.status_code == 200:
                    return True
                elif response.status_code == 503:
                    # Modèle en cours de chargement
                    self.logger.info("Model is loading, waiting...")
                    time.sleep(5)
                    continue
                else:
                    self.logger.warning(f"Unexpected response: {response.status_code}")
                    return False

            except Exception as e:
                self.logger.error(f"Error checking model status: {e}")
                time.sleep(5)

        return False

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        start_time = time.time()
        self._status = ServiceStatus.PROCESSING

        try:
            model_key = request.model or self.default_model
            if model_key not in self.AVAILABLE_MODELS:
                model_key = self.default_model
                self.logger.warning(f"Model {request.model} not found, using {model_key}")

            model_id = self.AVAILABLE_MODELS[model_key]
            self.client.model = model_id  # changer le modèle si besoin

            if not self.supports_size(request.width, request.height):
                max_w, max_h = self.max_resolution
                raise ValueError(f"Size {request.width}x{request.height} not supported. Max: {max_w}x{max_h}")

            self.logger.info(f"Generating image with model {model_key}: {request.prompt[:50]}...")

            # Appel à la méthode text_to_image
            image: Image.Image = self.client.text_to_image(
                request.prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale
                # Note: negative_prompt n'est pas encore supporté dans text_to_image officielle, à vérifier
            )

            timestamp = int(time.time())
            filename = f"hf_image_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            image.save(filepath, format="PNG")

            generation_time = time.time() - start_time

            self._status = ServiceStatus.IDLE
            self.logger.info(f"Image generated successfully in {generation_time:.2f}s: {filepath}")

            return ImageGenerationResult(
                success=True,
                filepath=filepath,
                prompt=request.prompt,
                generation_time=generation_time,
                metadata={
                    "model": model_key,
                    "model_name": model_id,
                    "parameters": {
                        "width": request.width,
                        "height": request.height,
                        "num_inference_steps": request.num_inference_steps,
                        "guidance_scale": request.guidance_scale,
                        "seed": request.seed,
                    },
                    "file_size": os.path.getsize(filepath)
                }
            )

        except Exception as e:
            self._status = ServiceStatus.ERROR
            error_msg = str(e)
            self.logger.error(f"Image generation failed: {error_msg}")
            return ImageGenerationResult(
                success=False,
                error_message=error_msg,
                generation_time=time.time() - start_time,
                prompt=request.prompt
            )
    def generate_image_simple(self, prompt: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Version simplifiée pour compatibilité avec l'ancien code

        Args:
            prompt: Description de l'image
            options: Options supplémentaires

        Returns:
            Dictionnaire avec les résultats
        """
        options = options or {}

        # Créer une requête à partir des paramètres
        request = ImageGenerationRequest(
            prompt=prompt,
            negative_prompt=options.get("negative_prompt"),
            width=options.get("width", 1024),
            height=options.get("height", 1024),
            num_inference_steps=options.get("num_inference_steps", 30),
            guidance_scale=options.get("guidance_scale", 7.5),
            seed=options.get("seed"),
            model=options.get("model")
        )

        # Générer l'image
        result = self.generate_image(request)

        # Convertir en format compatible
        if result.success:
            return {
                "success": True,
                "filepath": result.filepath,
                "prompt": result.prompt,
                "provider": "huggingface",
                "model": result.metadata.get("model", "unknown"),
                "generation_time": result.generation_time
            }
        else:
            return {
                "success": False,
                "error": result.error_message,
                "elapsed_time": result.generation_time
            }

    def list_available_models(self) -> Dict[str, str]:
        """Retourne la liste des modèles disponibles"""
        return self.AVAILABLE_MODELS.copy()

    def set_default_model(self, model_key: str) -> bool:
        """Change le modèle par défaut"""
        if model_key in self.AVAILABLE_MODELS:
            self.default_model = model_key
            self.logger.info(f"Default model changed to {model_key}")
            return True
        else:
            self.logger.error(f"Model {model_key} not available")
            return False

    def shutdown(self) -> bool:
        """Arrêt propre du service"""
        try:
            self._status = ServiceStatus.DISABLED
            self.logger.info("Hugging Face image service shutdown")
            return True
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return False


# Fonction helper pour créer le service facilement
def create_huggingface_image_service(api_key: str, model: str = "stable-diffusion-xl") -> HuggingFaceImageService:
    """
    Crée et configure un service Hugging Face

    Args:
        api_key: Clé API Hugging Face
        model: Modèle par défaut à utiliser

    Returns:
        Instance du service configurée
    """
    return HuggingFaceImageService(api_key=api_key, default_model=model)