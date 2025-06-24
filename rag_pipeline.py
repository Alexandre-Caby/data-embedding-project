import os
import sys
import torch
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI

from core.config import RAGConfig
from core.interfaces import Document, Chunk, SearchResult
from loaders import WebDataLoader, FileDataLoader
from processors import SentenceChunker, WordChunker, SentenceTransformerEmbedding, PerformanceOptimizer, timing_decorator
from storage import InMemoryVectorStore
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class RAGPipeline:
    def __init__(self, config: RAGConfig, local_model_path: Optional[str] = None, prompt_config_path: Optional[str] = None):
        self.config = config
        self.local_model_path = local_model_path
        self.conversation_history = []
        
        # Model path for reference
        self.gguf_model_path = os.path.join(os.path.dirname(__file__), "core", "qwen2.5-7b-instruct-q3_k_m.gguf")
        
        # Initialize OpenAI client for local API
        self.openai_client = OpenAI(
            api_key='1234', 
            base_url='http://127.0.0.1:8080/'
        )
        self.openai_model = "qwen2.5-7b-instruct-q3_k_m"
        
        # Initialize Performance Optimizer
        self.speed_optimizer = PerformanceOptimizer(
            cache_dir=os.path.join(config.output_directory, "cache"),
            max_workers=4
        )
        
        self.prompt_config = self._load_prompt_config(prompt_config_path)
        self._initialize_components()

    def _load_prompt_config(self, prompt_config_path):
        # Default prompt if no config file is found
        default_prompt = {
            "system_message": (
                "Vous êtes un assistant IA expert et précis. Votre rôle est de fournir des réponses complètes et utiles en utilisant les documents fournis.\n"
                "INSTRUCTIONS IMPORTANTES :\n"
                "1. Analyse approfondie : Lisez attentivement tous les documents fournis\n"
                "2. Réponse directe : Commencez par répondre directement à la question posée\n"
                "3. Citations précises : Mentionnez les sources (ex: \"D'après le Document 1...\")\n"
                "4. Contexte enrichi : Ajoutez des informations contextuelles pertinentes des documents\n"
                "5. Clarté : Organisez votre réponse de manière claire et structurée\n"
                "6. Limites : Si les documents ne contiennent pas assez d'informations, dites-le clairement\n"
                "STYLE DE RÉPONSE :\n"
                "- Commencez par la réponse principale\n"
                "- Ajoutez des détails pertinents\n"
                "- Mentionnez les sources utilisées\n"
                "- Utilisez un ton informatif mais accessible\n"
                "- Répondez en français sauf indication contraire\n"
                "GESTION DES QUESTIONS DIFFICILES :\n"
                "- Pour les questions factuelles précises (dates, mesures, etc.), cherchez des informations spécifiques\n"
                "- Pour les questions complexes, décomposez la réponse en plusieurs parties\n"
                "- Si une information manque, suggérez une recherche web pour compléter"
            )
        }
        if prompt_config_path is None:
            prompt_config_path = os.path.join(os.path.dirname(__file__), "config", "prompt_config.json")
        try:
            if os.path.exists(prompt_config_path):
                with open(prompt_config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load prompt config: {e}")
        return default_prompt

    def _initialize_components(self):
        # Initialize data loaders
        self.web_loader = WebDataLoader(self.config.scraping)
        self.file_loader = FileDataLoader()
        
        # Initialize chunker
        if self.config.chunking.strategy == "sentence":
            self.chunker = SentenceChunker(self.config.chunking)
        elif self.config.chunking.strategy == "word":
            self.chunker = WordChunker(self.config.chunking)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.config.chunking.strategy}")
        
        # Initialize embedding model
        self.embedder = SentenceTransformerEmbedding(self.config.embedding)
        
        # Initialize vector store
        if self.config.vector_store.storage_type == "memory":
            self.vector_store = InMemoryVectorStore(self.config.vector_store)
        else:
            raise ValueError(f"Unknown vector store type: {self.config.vector_store.storage_type}")
        
        # Print model path information
        print(f"Local GGUF model path: {self.gguf_model_path}")
        if not os.path.exists(self.gguf_model_path):
            print(f"WARNING: GGUF model file not found at {self.gguf_model_path}")
            print("Make sure your local API server is properly configured with the model.")
        
        # Try to test API connection
        try:
            test_response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": "Hello, are you working?"}],
                max_tokens=10
            )
            print("✓ Successfully connected to local OpenAI-compatible API")
        except Exception as e:
            print(f"⚠ Could not connect to local API: {str(e)}")
            print("Make sure your local server is running at http://127.0.0.1:8080/")
            print("You can start it with llama.cpp, text-generation-webui, or other compatible servers")
            print(f"Point the server to your model at: {self.gguf_model_path}")
            
        # Keep the HF model initialization for backward compatibility
        try:
            # Skip loading HF model if we're using local API
            if self.local_model_path:
                # Automatic device selection: 0 for GPU, -1 for CPU
                device = 0 if torch.cuda.is_available() else -1
                hf_model_id = self.local_model_path

                print(f"Loading HF model `{hf_model_id}` on device {device}")
                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    hf_model_id,
                    device_map="auto" if device == 0 else None,
                    trust_remote_code=True
                )
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.model = self.model.to(self.device)

                # Keep backward compatibility with pipeline
                self.llm = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=device,
                    truncation=True,
                    max_new_tokens=256,
                    return_full_text=False,
                )
            else:
                self.llm = None
                self.model = None
                self.tokenizer = None
                print("Skipping HF model loading (using local API instead)")

        except Exception as e:
            print(f"Error loading HF model: {str(e)}")
            print("Will use local OpenAI API exclusively.")
            self.llm = None
            self.model = None
            self.tokenizer = None

    def ingest_data(self, urls: Optional[List[str]] = None, file_paths: Optional[List[str]] = None) -> List[Chunk]:
        """Ingest data from URLs and/or files."""
        documents = []
        
        # Load from URLs
        if urls:
            web_docs = self.web_loader.load(urls)
            documents.extend(web_docs)
        
        # Load from files
        if file_paths:
            file_docs = self.file_loader.load(file_paths)
            documents.extend(file_docs)
        
        if not documents:
            raise ValueError("No documents loaded")
        
        # Process documents into chunks
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)
        
        # Generate embeddings
        self._generate_embeddings(all_chunks)
        
        # Add to vector store
        self.vector_store.add_chunks(all_chunks)

        return all_chunks

    def _generate_embeddings(self, chunks: List[Chunk]) -> None:
        """Generate embeddings for chunks."""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedder.encode(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for relevant chunks."""
        query_embedding = self.embedder.encode_single(query)
        return self.vector_store.search(query_embedding, top_k)

    def _enhance_query(self, query: str) -> str:
        """Enhance query with more sophisticated analysis."""
        # Tokenize the query for better analysis
        words = query.lower().split()
        
        # Basic spell correction for common errors
        corrections = {           
            'hauteure': 'hauteur', 'longeur': 'longueur', 'largeure': 'largeur',
            'conbien': 'combien', 'quand': 'quand', 'ou': 'où', 'comment': 'comment'
        }
        
        # Apply corrections
        corrected_words = [corrections.get(word, word) for word in words]
        return ' '.join(corrected_words)

    def _categorize_query(self, query: str) -> str:
        """Categorize the query using the category keywords from config."""
        query_lower = query.lower()
        
        # Get category keywords from config
        category_keywords = self.prompt_config.get("category_keywords", {})
        
        # Count matches for each category
        category_scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            category_scores[category] = score
        
        # Get the category with the highest score
        if category_scores:
            max_score = max(category_scores.values())
            if max_score > 0:
                for category, score in category_scores.items():
                    if score == max_score:
                        return category
        
        # Default categorization based on typical patterns
        if any(word in query_lower for word in ['taille', 'hauteur', 'largeur', 'dimension', 'mesure']):
            return "scientifique"
        elif any(word in query_lower for word in ['date', 'quand', 'année', 'siècle', 'époque']):
            return "historique"
        elif any(word in query_lower for word in ['où', 'lieu', 'endroit', 'localisation']):
            return "géographique"
        elif any(word in query_lower for word in ['qui', 'personne', 'personnage', 'nom']):
            return "historique" 
        elif any(word in query_lower for word in ['comment', 'méthode', 'processus', 'fonctionnement']):
            return "technologique"
        elif any(word in query_lower for word in ['pourquoi', 'raison', 'cause', 'motif']):
            return "philosophique"
        elif any(word in query_lower for word in ['combien', 'nombre', 'quantité', 'statistique']):
            return "scientifique"
        
        return "default"

    @timing_decorator
    def generate_response_fast(self, query: str, top_k: int = 8, use_history: bool = True) -> Dict[str, Any]:
        """Fast response generation with performance optimizations"""
        
        # Check cache first
        cached = self.speed_optimizer.cached_response(query)
        if cached:
            print("✓ Returning cached response")
            return cached
        
        # Quick intent detection
        intent = self.speed_optimizer.detect_intent_quickly(query)
        
        # Route non-RAG queries quickly
        if intent != "rag_query":
            return self._handle_non_rag_query(query, intent)
        
        # Handle simple queries with reduced processing
        if self.speed_optimizer.is_simple_query(query):
            result = self._handle_simple_query(query, top_k=3)
        else:
            # Full processing for complex queries
            result = self._handle_complex_query(query, top_k, use_history)
        
        # Cache the result
        self.speed_optimizer.cache_response(query, result)
        
        return result
    
    def _handle_non_rag_query(self, query: str, intent: str) -> Dict[str, Any]:
        """Handle non-RAG queries quickly"""
        quick_responses = {
            "image_generation": f"Pour générer une image de '{query}', utilisez le bouton 'Generate Image'.",
            "web_search": f"Pour rechercher '{query}' sur le web, utilisez le bouton 'Web Search'.",
            "system_info": "Pour obtenir des informations système, utilisez le bouton 'System Info'."
        }
        
        return {
            "context": quick_responses.get(intent, "Veuillez utiliser les boutons appropriés."),
            "retrieved_chunks": [],
            "intent": intent,
            "processing_mode": "quick_route"
        }
    
    def _handle_simple_query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Handle simple queries with minimal processing"""
        try:
            # Quick search with fewer chunks
            results = self.search(query, top_k=top_k)
            optimized_chunks = self.speed_optimizer.optimize_chunk_selection(results, max_chunks=2)
            
            if not optimized_chunks:
                return {
                    "context": f"Je n'ai pas trouvé d'information sur '{query}'. Essayez de reformuler ou utilisez la recherche web.",
                    "retrieved_chunks": [],
                    "processing_mode": "fast_no_data"
                }
            
            # Fast prompt creation
            fast_prompt = self.speed_optimizer.create_fast_prompt(query, optimized_chunks, max_context_length=800)
            
            # Quick generation with optimized parameters
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "Répondez de manière concise et précise en utilisant les documents fournis."},
                    {"role": "user", "content": fast_prompt}
                ],
                temperature=0.1,  # Very low for speed and precision
                max_tokens=200,   # Limit tokens for speed
                stream=False
            )
            
            result = {
                "context": response.choices[0].message.content.strip(),
                "retrieved_chunks": [
                    {
                        "content": chunk.chunk.content, 
                        "score": chunk.score,
                        "source": chunk.chunk.metadata.get("source_document", "unknown") if chunk.chunk.metadata else "unknown"
                    } 
                    for chunk in optimized_chunks
                ],
                "processing_mode": "fast"
            }
            
            return result
            
        except Exception as e:
            return {
                "context": f"Erreur lors du traitement rapide de '{query}': {str(e)}",
                "retrieved_chunks": [],
                "processing_mode": "fast_error"
            }
    
    def _handle_complex_query(self, query: str, top_k: int = 8, use_history: bool = True, web_context: str = None) -> Dict[str, Any]:
        """Handle complex queries with standard processing"""
        try:
            # If web_context is provided, prioritize it
            if web_context:
                system_message = """Vous êtes un assistant IA expert. Analysez les informations provenant de recherches web et répondez 
                de manière précise et complète à la question posée. Utilisez les informations fournies pour créer une réponse 
                synthétique et informative. Ne mentionnez pas explicitement que vous utilisez des recherches web. 
                Répondez en français et soyez précis."""
                
                user_message = f"""Question: {query}

Informations obtenues du web:
{web_context}

Analysez ces informations et répondez à la question de manière complète et précise."""
                
                # Generate response using web context
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.7,  # Higher for creative synthesis
                    max_tokens=800,
                    stream=False
                )
                
                return {
                    "context": response.choices[0].message.content.strip(),
                    "retrieved_chunks": [],
                    "processing_mode": "web_synthesis",
                    "web_processed": True
                }
            
            # Use standard processing but with some optimizations
            enhanced_query = self._enhance_query(query)
            category = self._categorize_query(query)
            
            # Search with more chunks for complex queries
            results = self.search(enhanced_query, top_k=top_k)
            optimized_chunks = self.speed_optimizer.optimize_chunk_selection(results, max_chunks=4)
            
            if not optimized_chunks:
                no_context_response = self.prompt_config.get("no_context_response", "Je n'ai pas trouvé d'information pertinente...")
                return {
                    "context": no_context_response.replace("{query}", query),
                    "retrieved_chunks": [],
                    "processing_mode": "standard_no_data"
                }
            
            # Build context from optimized chunks
            context = "\n\n".join([f"Document {i+1}: {chunk.chunk.content}" for i, chunk in enumerate(optimized_chunks)])
            
            # If web_context is provided, include it in the prompt
            user_message = ""
            if web_context:
                system_message = """Vous êtes un assistant IA expert. Analysez les informations provenant de recherches web et répondez 
                de manière précise et complète à la question posée. Utilisez les informations fournies mais ne les citez pas 
                directement. Synthétisez l'information pour donner une réponse cohérente. Ne mentionnez pas que vous utilisez 
                des recherches web dans votre réponse. Si les informations sont insuffisantes, dites-le clairement."""
                
                user_message = f"""Question: {query}

    Information obtenue du web:
    {web_context}

    Répondez à la question en vous basant sur ces informations. Soyez précis et informatif."""
                
                # Use higher temperature for creative synthesis of web information
                temperature = 0.7
                max_tokens = 800
            else:
                # Get category-specific parameters
                temperature = self.prompt_config.get("temperature_by_category", {}).get(category, 0.3)
                max_tokens = self.prompt_config.get("max_tokens_by_category", {}).get(category, 600)
                
                # Standard generation with optimizations
                system_message = self.prompt_config.get("system_message", "Vous êtes un assistant IA expert...")
                user_message_template = self.prompt_config.get("user_message_template", 
                    "## DOCUMENTS DE RÉFÉRENCE :\n{context}\n\n## QUESTION :\n{query}")
                
                user_message = user_message_template.replace("{context}", context)
                user_message = user_message.replace("{query}", query)
                user_message = user_message.replace("{category}", category)
                
                messages = [{"role": "system", "content": system_message}]
                
                # Add limited conversation history for speed
                if use_history and self.conversation_history:
                    for entry in self.conversation_history[-1:]:  # Only last exchange
                        messages.append({"role": "user", "content": entry[0]})
                        messages.append({"role": "assistant", "content": entry[1]})
                
                messages.append({"role": "user", "content": user_message})
        
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Update conversation history
            if use_history:
                self.conversation_history.append((query, response_text))
                # Keep only last 2 exchanges for speed
                self.conversation_history = self.conversation_history[-2:]
            
            result = {
                "context": response_text,
                "retrieved_chunks": [
                    {
                        "content": chunk.chunk.content, 
                        "score": float(chunk.score),
                        "source": chunk.chunk.metadata.get("source_document", "unknown") if chunk.chunk.metadata else "unknown",
                        "metadata": chunk.chunk.metadata if chunk.chunk.metadata else {}
                    } 
                    for chunk in optimized_chunks
                ],
                "processing_mode": "standard",
                "category": category
            }
            
            return result
            
        except Exception as e:
            return {
                "context": f"Erreur lors du traitement complexe: {str(e)}. Essayez la recherche web pour '{query}'.",
                "retrieved_chunks": [],
                "processing_mode": "standard_error"
            }

    def generate_response(self, query: str, top_k: int = 8, use_history: bool = True) -> Dict[str, Any]:
        """Generate a high-quality response by default using the prompt configuration."""
        
        query_lower = query.lower().strip()
        
        # Handle conversational phrases directly with predefined responses
        conversational_phrases = self.prompt_config.get("conversational_phrases", [])
        conversational_responses = self.prompt_config.get("conversational_responses", {})
        
        # Check for exact matches in common phrases
        if query_lower in conversational_responses:
            return {
                "context": conversational_responses[query_lower],
                "retrieved_chunks": [],
                "processing_mode": "conversational"
            }
        
        # Check for partial matches in common phrases
        for phrase in conversational_phrases:
            if phrase in query_lower and len(query_lower.split()) <= 5:
                # For longer conversational queries, use the closest matching response
                closest_phrase = phrase
                return {
                    "context": conversational_responses.get(closest_phrase, 
                              "Bonjour ! Comment puis-je vous aider aujourd'hui ?"),
                    "retrieved_chunks": [],
                    "processing_mode": "conversational"
                }
        
        # Check for greeting queries
        greeting_words = ['hello', 'hey', 'hi', 'bonjour', 'salut', 'bonsoir']
        simple_queries = ['qui es-tu', 'what are you', 'que peux-tu faire', 'what can you do', 'aide', 'help']
        
        # Return greeting response from config if appropriate
        if any(word in query_lower for word in greeting_words) and len(query.split()) <= 5:
            greeting_response = self.prompt_config.get("greeting_response", "Bonjour ! Je suis votre assistant IA.")
            return {
                "context": greeting_response,
                "retrieved_chunks": []
            }
        
        # Return help response if appropriate
        if any(phrase in query_lower for phrase in simple_queries):
            return {
                "context": self.prompt_config.get("greeting_response", "Je peux vous aider de plusieurs façons..."),
                "retrieved_chunks": []
            }
        
        # Check if this is likely a general knowledge question
        general_knowledge_patterns = self.prompt_config.get("general_knowledge_queries", [])
        is_general_knowledge = any(pattern in query_lower for pattern in general_knowledge_patterns)
        
        # Search for relevant chunks
        enhanced_query = self._enhance_query(query)
        search_results = self.search(enhanced_query, top_k=top_k)
        
        # Check if we have relevant results (higher threshold for general knowledge questions)
        relevance_threshold = 0.5 if is_general_knowledge else 0.2
        relevant_chunks = [result for result in search_results[:5] if result.score > relevance_threshold]
        
        # If this is likely general knowledge and we don't have highly relevant chunks, suggest web search
        if is_general_knowledge and (not relevant_chunks or len(relevant_chunks) < 2):
            no_context_response = self.prompt_config.get("no_context_response", "Je n'ai pas trouvé d'information pertinente...")
            return {
                "context": no_context_response.replace("{query}", query),
                "retrieved_chunks": [],
                "suggested_web_search": query
            }
        
        # Continue with regular processing for non-general knowledge or if we have relevant chunks
        return self.generate_response_fast(query, top_k, use_history)

    def chat(self, message: str, top_k: int = 5) -> str:
        """Simple chat interface that returns just the response text."""
        response = self.generate_response(message, top_k, use_history=True)
        return response["context"]
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        return "Conversation has been reset."
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.speed_optimizer.get_cache_stats()
    
    def clear_cache(self):
        """Clear performance cache"""
        self.speed_optimizer.clear_cache()

    def save(self, path: Optional[str] = None) -> None:
        """Save the pipeline state."""
        save_path = path or self.config.output_directory
        self.vector_store.save(save_path)
        print(f"Pipeline saved to {save_path}")

    def load(self, path: Optional[str] = None) -> None:
        """Load a saved pipeline state."""
        load_path = path or self.config.output_directory
        
        # Load vector store data
        self.vector_store.load(load_path)
        
        # Check if embedding model dimensions match
        if hasattr(self.vector_store, 'embeddings') and len(self.vector_store.embeddings) > 0:
            stored_dim = self.vector_store.embeddings[0].shape[0]
            model_dim = self.embedder.get_dimension()
            
            if stored_dim != model_dim:
                print(f"WARNING: Embedding dimension mismatch! Stored: {stored_dim}, Model: {model_dim}")
                print(f"Recreating embedder with matching model...")
                
                # Try to find a model with matching dimensions
                if stored_dim == 1024:
                    self.config.embedding.model_name = "sentence-transformers/all-mpnet-base-v2"  # 1024 dimensions
                elif stored_dim == 768:
                    self.config.embedding.model_name = "sentence-transformers/all-distilroberta-v1"  # 768 dimensions
                elif stored_dim == 384:
                    self.config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions
                
                # Reinitialize the embedder
                self.embedder = SentenceTransformerEmbedding(self.config.embedding)
                
        print(f"Pipeline loaded from {load_path}")
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'speed_optimizer'):
            self.speed_optimizer.cleanup()

# Factory function for easy pipeline creation
def create_rag_pipeline(
    embedding_model: str = "all-MiniLM-L6-v2",
    chunk_size: int = 200,
    chunk_overlap: int = 50,
    output_dir: str = "data",
    local_model_path: Optional[str] = None,
    openai_api_base: str = "http://127.0.0.1:8080/",
    openai_api_key: str = "1234",
    openai_model: str = "qwen2.5-7b-instruct-q3_k_m"
) -> RAGPipeline:
    """Factory function to create a RAG pipeline with common configurations."""
    config = RAGConfig()
    config.embedding.model_name = embedding_model
    config.chunking.words_per_chunk = chunk_size
    config.chunking.overlap = chunk_overlap
    config.output_directory = output_dir
    
    pipeline = RAGPipeline(config, local_model_path)
    
    # Configure OpenAI client
    pipeline.openai_client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base
    )
    pipeline.openai_model = openai_model
    
    return pipeline