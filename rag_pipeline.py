import os
import sys
import torch
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI

from core.config import RAGConfig
from core.interfaces import Document, Chunk, SearchResult
from loaders import WebDataLoader, FileDataLoader
from processors import SentenceChunker, WordChunker, SentenceTransformerEmbedding
from storage import InMemoryVectorStore
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class RAGPipeline:
    def __init__(self, config: RAGConfig, local_model_path: Optional[str] = None, prompt_config_path: Optional[str] = None):
        self.config = config
        self.local_model_path = local_model_path
        self.conversation_history = []
        
        # Model path for reference
        self.gguf_model_path = os.path.join(os.path.dirname(__file__), "core", "qwen3-30b.gguf")
        
        # Initialize OpenAI client for local API
        self.openai_client = OpenAI(
            api_key='1234', 
            base_url='http://127.0.0.1:8080/'
        )
        self.openai_model = "qwen2.5-7b-instruct-q3_k_m"
        
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

    def generate_response(self, query: str, top_k: int = 5, use_history: bool = True) -> Dict[str, Any]:
        """Generate a response using the RAG pipeline with enhanced query handling."""
        
        # Handle greetings and simple queries without requiring context first
        greeting_words = ['hello', 'hey', 'hi', 'bonjour', 'salut', 'bonsoir', 'comment allez-vous', 'comment ça va']
        simple_queries = ['qui es-tu', 'what are you', 'que peux-tu faire', 'what can you do', 'aide', 'help']
        
        query_lower = query.lower().strip()
        
        # Check for greetings
        if any(word in query_lower for word in greeting_words) and len(query.split()) <= 5:
            return {
                "context": "Bonjour ! Je suis votre assistant IA. Je peux répondre à vos questions en me basant sur les documents que j'ai en mémoire, faire des recherches web, générer des images, ou vous donner des informations système. Comment puis-je vous aider ?",
                "retrieved_chunks": []
            }
        
        # Check for help/capability queries
        if any(phrase in query_lower for phrase in simple_queries):
            return {
                "context": "Je peux vous aider de plusieurs façons :\n• Répondre à vos questions en me basant sur mes documents\n• Faire des recherches sur le web\n• Générer des images à partir de descriptions\n• Fournir des informations système\n\nPosez-moi une question ou utilisez des commandes comme 'search web for...', 'generate image...', etc.",
                "retrieved_chunks": []
            }
        
        # Improve query with spell correction and expansion
        enhanced_query = self._enhance_query(query)
        
        # Search for relevant chunks with both original and enhanced query
        search_results = self.search(enhanced_query, top_k=top_k)
        if not search_results or all(result.score < 0.2 for result in search_results[:3]):
            # Try with original query if enhanced didn't work
            search_results = self.search(query, top_k=top_k)
        
        # Extract the content from search results
        context_chunks = [result.chunk.content for result in search_results[:3] if result.score > 0.2]
        context = "\n\n".join([f"Document {i+1}: {content}" for i, content in enumerate(context_chunks)])
        
        # If no relevant chunks found, provide a helpful response
        if not context_chunks or all(result.score < 0.3 for result in search_results[:3]):
            return {
                "context": "Je n'ai pas trouvé d'information pertinente dans ma base de documents pour répondre à votre question. Vous pouvez :\n• Reformuler votre question\n• Utiliser 'search web for [votre question]' pour chercher sur internet\n• Me demander de l'aide avec 'help' pour voir mes capacités",
                "retrieved_chunks": []
            }

        # Prepare messages for OpenAI API
        messages = []
        
        # Use prompt from config
        system_message = self.prompt_config.get("system_message", "")
        messages.append({"role": "system", "content": system_message})
        
        # Add conversation history if enabled
        if use_history and self.conversation_history:
            for entry in self.conversation_history[-2:]:
                messages.append({"role": "user", "content": entry[0]})
                messages.append({"role": "assistant", "content": entry[1]})
        
        # Add current context and query
        user_message = f"""## DOCUMENTS DE RÉFÉRENCE :
{context}

## QUESTION À ANALYSER :
Question originale : {query}
Question améliorée : {enhanced_query}

## INSTRUCTIONS SPÉCIALES :
- Cette question semble porter sur : {self._categorize_query(query)}
- Recherchez des informations spécifiques dans les documents
- Si vous trouvez des données pertinentes, présentez-les clairement
- Si les informations sont incomplètes, mentionnez ce qui manque

Répondez de manière complète et structurée :"""
        
        messages.append({"role": "user", "content": user_message})
        
        # Generate response using local OpenAI API
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0.2,  # Lower temperature for more precise responses
                max_tokens=600,   # More tokens for detailed responses
                stream=False
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Post-process response to add web search suggestion if needed
            if "je n'ai pas" in response_text.lower() or "information manque" in response_text.lower():
                if needs_current_info:
                    response_text += f"\n\n💡 **Suggestion**: Pour obtenir des informations plus récentes ou précises, essayez une recherche web : `search web for {query}`"
            
            # Update conversation history
            if use_history:
                self.conversation_history.append((query, response_text))
                
            return {
                "context": response_text,
                "retrieved_chunks": [
                    {"content": result.chunk.content, 
                     "score": float(result.score), 
                     "source": result.chunk.metadata.get("source_document", "unknown") if result.chunk.metadata else "unknown",
                     "metadata": result.chunk.metadata if result.chunk.metadata else {}}
                    for result in search_results[:5]  # Include more chunks for reference
                ]
            }
        except Exception as e:
            print(f"Error with OpenAI API: {str(e)}")
            
            # Enhanced fallback response
            return {
                "context": f"Je rencontre une difficulté technique pour analyser votre question '{query}'. Pour cette question qui semble nécessiter des informations précises, je recommande d'utiliser la recherche web : `search web for {query}`",
                "retrieved_chunks": [
                    {"content": result.chunk.content, 
                     "score": float(result.score), 
                     "source": result.chunk.metadata.get("source_document", "unknown") if result.chunk.metadata else "unknown",
                     "metadata": result.chunk.metadata if result.chunk.metadata else {}}
                    for result in search_results
                ]
            }

    def chat(self, message: str, top_k: int = 5) -> str:
        """Simple chat interface that returns just the response text."""
        response = self.generate_response(message, top_k, use_history=True)
        return response["context"]
    
    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []
        return "Conversation has been reset."

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
