import os
import time
import base64
import io
from typing import Dict, Any, Optional, Union
from PIL import Image
import requests
import logging

class ImageGenerationService:
    """Service for generating images from text descriptions."""
    
    def __init__(self, provider: str = "local", model: str = "stable-diffusion-xl-base-1.0", api_key: str = ""):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.logger = logging.getLogger("orchestrator.image_generation")
        
        # Set up output directory
        self.output_dir = os.path.join("output", "images")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self._init_provider()
    
    def _init_provider(self):
        """Initialize the selected image generation provider."""
        if self.provider == "local":
            try:
                # Try to import necessary libraries for local generation
                import torch
                from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
                
                # Determine which model architecture to use
                if "xl" in self.model.lower():
                    self.pipe = StableDiffusionXLPipeline.from_pretrained(
                        self.model,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        variant="fp16"
                    )
                else:
                    self.pipe = StableDiffusionPipeline.from_pretrained(
                        self.model,
                        torch_dtype=torch.float16,
                        use_safetensors=True
                    )
                
                # Move to GPU if available
                if torch.cuda.is_available():
                    self.pipe = self.pipe.to("cuda")
                    self.logger.info(f"Using GPU for image generation with model {self.model}")
                else:
                    self.logger.info(f"Using CPU for image generation with model {self.model}")
                
            except ImportError:
                self.logger.warning("Required packages for local image generation not found. "
                                   "Install with: pip install torch diffusers transformers accelerate")
                self.pipe = None
            except Exception as e:
                self.logger.error(f"Error initializing local image generation: {e}")
                self.pipe = None
        
        elif self.provider == "openai":
            if not self.api_key:
                self.logger.warning("No API key provided for OpenAI DALL-E")
            # OpenAI client will be initialized when needed
        
        elif self.provider == "stability":
            if not self.api_key:
                self.logger.warning("No API key provided for Stability AI")
            # Stability client will be initialized when needed
        
        else:
            self.logger.warning(f"Unknown provider '{self.provider}'. Falling back to local.")
            self.provider = "local"
            self._init_provider()
    
    def generate_image(self, prompt: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate an image based on the text prompt.
        
        Args:
            prompt: Text description of the image to generate
            options: Additional options for image generation
            
        Returns:
            Dict containing information about the generated image
        """
        options = options or {}
        start_time = time.time()
        
        # Set default parameters
        width = options.get("width", 1024)
        height = options.get("height", 1024)
        negative_prompt = options.get("negative_prompt", "blurry, bad quality, distorted")
        
        # Generate timestamp for filename
        timestamp = int(time.time())
        filename = f"image_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            # Generate with appropriate provider
            if self.provider == "local":
                return self._generate_local(prompt, negative_prompt, width, height, filepath)
            elif self.provider == "openai":
                return self._generate_openai(prompt, width, height, filepath)
            elif self.provider == "stability":
                return self._generate_stability(prompt, negative_prompt, width, height, filepath)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        
        except Exception as e:
            self.logger.error(f"Image generation error: {e}")
            return {
                "success": False,
                "error": str(e),
                "elapsed_time": time.time() - start_time
            }
    
    def _generate_local(self, prompt, negative_prompt, width, height, filepath):
        """Generate image using local diffusers pipeline."""
        if not self.pipe:
            raise ValueError("Local image generation is not properly initialized")
        
        # Generate the image
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
        # Save the image
        image.save(filepath)
        
        return {
            "success": True,
            "filepath": filepath,
            "prompt": prompt,
            "provider": "local",
            "model": self.model
        }
    
    def _generate_openai(self, prompt, width, height, filepath):
        """Generate image using OpenAI's DALL-E API."""
        from openai import OpenAI
        
        client = OpenAI(api_key=self.api_key)
        
        # Map dimensions to DALL-E size parameter
        if width == height == 1024:
            size = "1024x1024"
        elif width == 1792 and height == 1024:
            size = "1792x1024"
        elif width == 1024 and height == 1792:
            size = "1024x1792"
        else:
            size = "1024x1024"  # Default
        
        # Generate image
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality="standard",
            n=1,
        )
        
        # Get image URL
        image_url = response.data[0].url
        
        # Download and save the image
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(image_response.content)
        
        return {
            "success": True,
            "filepath": filepath,
            "prompt": prompt,
            "provider": "openai",
            "model": "dall-e-3"
        }
    
    def _generate_stability(self, prompt, negative_prompt, width, height, filepath):
        """Generate image using Stability AI API."""
        import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
        from stability_sdk import client
        
        # Initialize stability client
        stability_api = client.StabilityInference(
            key=self.api_key,
            verbose=True,
        )
        
        # Generate the image
        answers = stability_api.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            samples=1
        )
        
        # Process and save the result
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.type == generation.ARTIFACT_IMAGE:
                    # Save the image
                    with open(filepath, "wb") as f:
                        f.write(artifact.binary)
                    
                    return {
                        "success": True,
                        "filepath": filepath,
                        "prompt": prompt,
                        "provider": "stability",
                        "model": self.model
                    }
        
        raise ValueError("No image was generated by Stability AI")
    
    def shutdown(self):
        """Clean up resources."""
        if hasattr(self, 'pipe') and self.pipe is not None:
            try:
                del self.pipe
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                self.logger.error(f"Error during shutdown: {e}")