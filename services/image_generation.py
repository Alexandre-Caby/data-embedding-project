import os
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from PIL import Image
from huggingface_hub import InferenceClient

class ServiceStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    DISABLED = "disabled"

@dataclass
class ImageGenerationRequest:
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    model: Optional[str] = None

@dataclass
class ImageGenerationResult:
    success: bool
    filepath: Optional[str] = None
    prompt: Optional[str] = None
    error_message: Optional[str] = None
    generation_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class HuggingFaceImageService:
    """Génération d’images via HF Inference (free tier)."""

    # On ne précise pas de provider ici : InferenceClient choisira automatiquement le service gratuit.
    def __init__(self, api_key: str):
        self.api_key       = api_key
        self.logger        = logging.getLogger("orchestrator.huggingface_image")
        self._status       = ServiceStatus.IDLE
        # Pas de modèle hardcodé : HF Inference utilisera un modèle gratuit par défaut.
        self.client = InferenceClient(token=self.api_key)

        self.output_dir = os.path.join("output", "images")
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def status(self) -> ServiceStatus:
        return self._status

    def supports_size(self, width: int, height: int) -> bool:
        # HF Inference accepte typiquement jusqu’à 1024×1024
        return 0 < width <= 1024 and 0 < height <= 1024

    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        start = time.time()
        self._status = ServiceStatus.PROCESSING

        try:
            if not self.supports_size(request.width, request.height):
                raise ValueError(f"Taille {request.width}x{request.height} non supportée")

            # Appel simple à text_to_image : HF Inference choisit un modèle gratuit
            image: Image.Image = self.client.text_to_image(
                request.prompt,
                width=request.width,
                height=request.height,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale
            )

            filename = f"hf_image_{int(start)}.png"
            path     = os.path.join(self.output_dir, filename)
            image.save(path, format="PNG")

            self._status = ServiceStatus.IDLE
            return ImageGenerationResult(
                success=True,
                filepath=path,
                prompt=request.prompt,
                generation_time=time.time() - start,
                metadata={"provider": "hf-inference"}
            )

        except Exception as e:
            self._status = ServiceStatus.ERROR
            msg = str(e)
            self.logger.error(f"Image generation failed: {msg}")
            return ImageGenerationResult(
                success=False,
                error_message=msg,
                generation_time=time.time() - start
            )

    def generate_image_simple(self, prompt: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Interface simplifiée pour l’orchestrateur."""
        opts = options or {}
        req = ImageGenerationRequest(
            prompt=prompt,
            negative_prompt=opts.get("negative_prompt"),
            width=opts.get("width", 512),
            height=opts.get("height", 512),
            num_inference_steps=opts.get("num_inference_steps", 30),
            guidance_scale=opts.get("guidance_scale", 7.5),
            seed=opts.get("seed")
        )
        res = self.generate_image(req)
        return {
            "success": res.success,
            "filepath": res.filepath,
            "prompt": res.prompt,
            "error": res.error_message,
            "generation_time": res.generation_time
        }
