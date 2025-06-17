from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ServiceStatus(Enum):
    """Status des services"""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    DISABLED = "disabled"


class ContentType(Enum):
    """Types de contenu supportés"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"


@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    source: str
    content_type: ContentType = ContentType.TEXT
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validation après initialisation"""
        if not self.id or not self.content:
            raise ValueError("Document must have id and content")


@dataclass
class Chunk:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    document_id: Optional[str] = None
    chunk_index: int = 0

    def __post_init__(self):
        """Validation après initialisation"""
        if not self.id or not self.content:
            raise ValueError("Chunk must have id and content")


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    rank: int
    relevance_explanation: Optional[str] = None

    def __post_init__(self):
        """Validation du score"""
        if not 0 <= self.score <= 1:
            raise ValueError("Score must be between 0 and 1")


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


@dataclass
class ImageGenerationResult:
    """Résultat de génération d'image"""
    success: bool
    filepath: Optional[str] = None
    prompt: Optional[str] = None
    error_message: Optional[str] = None
    generation_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Interfaces de base
class DataLoader(ABC):
    """Interface pour le chargement de données"""

    @abstractmethod
    def load(self, sources: List[str]) -> List[Document]:
        """Charge des documents depuis des sources"""
        pass

    @abstractmethod
    def supports(self, source: str) -> bool:
        """Vérifie si la source est supportée"""
        pass


class TextChunker(ABC):
    """Interface pour le découpage de texte"""

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Découpe un document en chunks"""
        pass

    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """Taille des chunks"""
        pass

    @property
    @abstractmethod
    def chunk_overlap(self) -> int:
        """Chevauchement entre chunks"""
        pass


class EmbeddingModel(ABC):
    """Interface pour les modèles d'embedding"""

    @abstractmethod
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode plusieurs textes"""
        pass

    @abstractmethod
    def encode_single(self, text: str) -> List[float]:
        """Encode un seul texte"""
        pass

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Dimension des embeddings"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Nom du modèle"""
        pass


class VectorStore(ABC):
    """Interface pour le stockage vectoriel"""

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Ajoute des chunks au store"""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        """Recherche par similarité vectorielle"""
        pass

    @abstractmethod
    def search_by_text(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Recherche par texte (avec embedding automatique)"""
        pass

    @abstractmethod
    def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """Supprime des chunks"""
        pass

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Récupère un chunk par ID"""
        pass

    @abstractmethod
    def count(self) -> int:
        """Nombre de chunks stockés"""
        pass

    @abstractmethod
    def save(self, path: str) -> bool:
        """Sauvegarde le store"""
        pass

    @abstractmethod
    def load(self, path: str) -> bool:
        """Charge le store"""
        pass

    @property
    @abstractmethod
    def chunks(self) -> List[Chunk]:
        """Accès en lecture aux chunks"""
        pass


# Nouvelles interfaces pour les services d'agents
class BaseService(ABC):
    """Interface de base pour tous les services"""

    @property
    @abstractmethod
    def service_name(self) -> str:
        """Nom du service"""
        pass

    @property
    @abstractmethod
    def status(self) -> ServiceStatus:
        """Status actuel du service"""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Vérification de santé du service"""
        pass

    @abstractmethod
    def shutdown(self) -> bool:
        """Arrêt propre du service"""
        pass


class ImageGenerationService(BaseService):
    """Interface pour la génération d'images"""

    @abstractmethod
    def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResult:
        """Génère une image à partir d'une requête"""
        pass

    @abstractmethod
    def supports_size(self, width: int, height: int) -> bool:
        """Vérifie si la taille est supportée"""
        pass

    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """Formats d'image supportés"""
        pass

    @property
    @abstractmethod
    def max_resolution(self) -> tuple:
        """Résolution maximale (width, height)"""
        pass


class WebSearchService(BaseService):
    """Interface pour la recherche web"""

    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """Effectue une recherche web"""
        pass

    @abstractmethod
    def search_with_filters(self, query: str, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Recherche avec filtres"""
        pass

    @property
    @abstractmethod
    def supported_providers(self) -> List[str]:
        """Fournisseurs de recherche supportés"""
        pass


class OSOperationsService(BaseService):
    """Interface pour les opérations système"""

    @abstractmethod
    def execute_command(self, command: str, safe_mode: bool = True) -> Dict[str, Any]:
        """Exécute une commande système"""
        pass

    @abstractmethod
    def read_file(self, filepath: str) -> Dict[str, Any]:
        """Lit un fichier"""
        pass

    @abstractmethod
    def write_file(self, filepath: str, content: str) -> Dict[str, Any]:
        """Écrit dans un fichier"""
        pass

    @abstractmethod
    def list_directory(self, path: str) -> Dict[str, Any]:
        """Liste le contenu d'un répertoire"""
        pass

    @property
    @abstractmethod
    def allowed_operations(self) -> List[str]:
        """Opérations autorisées"""
        pass


class ConversationManager(ABC):
    """Interface pour la gestion des conversations"""

    @abstractmethod
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> str:
        """Ajoute un message à la conversation"""
        pass

    @abstractmethod
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Récupère l'historique de conversation"""
        pass

    @abstractmethod
    def clear_conversation(self) -> bool:
        """Efface l'historique"""
        pass

    @abstractmethod
    def get_context_for_query(self, query: str) -> str:
        """Récupère le contexte pertinent pour une requête"""
        pass


class AgentOrchestrator(ABC):
    """Interface pour l'orchestrateur d'agents"""

    @abstractmethod
    def register_service(self, service: BaseService) -> bool:
        """Enregistre un nouveau service"""
        pass

    @abstractmethod
    def get_service(self, service_name: str) -> Optional[BaseService]:
        """Récupère un service par nom"""
        pass

    @abstractmethod
    def process_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Traite une requête utilisateur"""
        pass

    @abstractmethod
    def route_query(self, query: str) -> List[str]:
        """Détermine quels services utiliser pour une requête"""
        pass

    @property
    @abstractmethod
    def available_services(self) -> List[str]:
        """Liste des services disponibles"""
        pass

    @property
    @abstractmethod
    def service_health(self) -> Dict[str, ServiceStatus]:
        """État de santé de tous les services"""
        pass