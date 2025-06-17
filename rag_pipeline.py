import os
import sys
import torch
from typing import List, Dict, Any, Optional
from openai import OpenAI

from core.config import RAGConfig
from core.interfaces import Document, Chunk, SearchResult
from loaders import WebDataLoader, FileDataLoader
from processors import SentenceChunker, WordChunker, SentenceTransformerEmbedding
from storage import InMemoryVectorStore
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class RAGPipeline:
    def __init__(self, config: RAGConfig, local_model_path: Optional[str] = None):
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
        
        self._initialize_components()

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
        """Generate a response using the RAG pipeline with local OpenAI API."""
        # Search for relevant chunks
        search_results = self.search(query, top_k=top_k)
        
        # Extract the content from search results
        context_chunks = [result.chunk.content for result in search_results[:3]]
        context = "\n\n".join([f"Document {i+1}: {content}" for i, content in enumerate(context_chunks)])
        
        if not context_chunks:
            return {
                "context": "Je n'ai pas trouvé d'information pertinente pour répondre à votre question.",
                "retrieved_chunks": []
            }
        
        # Prepare messages for OpenAI API
        messages = []
        
        # System message with instructions
        system_message = """<no_thinking>
You are a precise, helpful, and knowledgeable assistant with expertise in retrieving and analyzing information. 
Your answers must be based exclusively on the provided context. Follow these guidelines:

1. Answer the user's question using ONLY the information in the reference documents.
2. If the information needed is not in the references, respond with: "Je n'ai pas suffisamment d'informations pour répondre à cette question."
3. Do not use prior knowledge or make assumptions beyond what's explicitly stated in the references.
4. Cite the specific Document number(s) you used in your response (e.g., "D'après le Document 2...").
5. Be concise but thorough, organizing complex information in a clear structure.
6. Maintain a neutral, informative tone.
7. Respond in French unless specified otherwise."""
        
        messages.append({"role": "system", "content": system_message})
        
        # Add conversation history if enabled
        if use_history and self.conversation_history:
            for entry in self.conversation_history[-3:]:  # Include up to 3 most recent exchanges
                messages.append({"role": "user", "content": entry[0]})
                messages.append({"role": "assistant", "content": entry[1]})
        
        # Add current context and query
        user_message = f"""## REFERENCE INFORMATION:
{context}

Please answer the following question based ONLY on the information provided above:
{query}"""
        
        messages.append({"role": "user", "content": user_message})
        
        # Generate response using local OpenAI API
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                stream=False  # Disable streaming for faster response
            )
            
            response_text = response.choices[0].message.content.strip()
            
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
                    for result in search_results
                ]
            }
        except Exception as e:
            print(f"Error with OpenAI API: {str(e)}")
            print("Falling back to HuggingFace model if available")
            
            # Fallback to HuggingFace model if available
            if self.llm:
                prompt = f"""You are a precise, helpful assistant. Answer based only on this information:

## REFERENCE INFORMATION:
{context}

## QUESTION:
{query}

## ANSWER:"""
                
                response = self.llm(prompt, max_length=512, 
                                   do_sample=True, temperature=0.7)
                
                response_text = response[0]['generated_text']
                
                # Extract only the model's answer (remove the prompt)
                if "## ANSWER:" in response_text:
                    response_text = response_text.split("## ANSWER:")[1].strip()
                
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
                        for result in search_results
                    ]
                }
            
            return {
                "context": f"Erreur lors de la génération de la réponse: {str(e)}",
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
