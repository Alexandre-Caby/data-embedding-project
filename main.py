import json
import os
import re
import random
import argparse
from collections import Counter
from orchestrator import AIServicesOrchestrator

def load_data_sources(config_file="config_data_source/data_sources.json"):
    """Load data sources from a JSON configuration file."""
    # Create default config if file doesn't exist
    if not os.path.exists(config_file):
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        default_config = {
            "urls": [],
            "file_paths": []
        }
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2)
        print(f"Created default configuration at {config_file}")
        return default_config["urls"], default_config["file_paths"]
    
    # Load existing config
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get("urls", []), config.get("file_paths", [])
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return [], []

def detect_content_type(chunks):
    """Detect the dominant topic/domain of the content based on keyword frequency."""
    # Define domain-specific keywords
    domains = {
        "science": ["science", "scientific", "research", "experiment", "laboratory", "theory", 
                   "physics", "chemistry", "biology", "astronomy", "neuroscience", "hypothesis"],
        "technology": ["technology", "computer", "software", "hardware", "algorithm", "programming", 
                      "artificial intelligence", "machine learning", "data", "internet", "digital", "code"],
        "politics": ["politics", "government", "policy", "election", "democracy", "republican", 
                    "democrat", "parliament", "congress", "senate", "legislation", "vote"],
        "gaming": ["game", "gaming", "player", "console", "PlayStation", "Xbox", "Nintendo", 
                  "gameplay", "character", "level", "achievement", "multiplayer"],
        "business": ["business", "economy", "market", "finance", "investment", "company", 
                    "corporation", "startup", "entrepreneur", "profit", "industry", "stock"],
        "health": ["health", "medicine", "medical", "disease", "treatment", "therapy", 
                  "doctor", "patient", "hospital", "diagnosis", "symptom", "cure"]
    }
    
    # Combine all text content
    all_text = " ".join([chunk.content.lower() for chunk in chunks if hasattr(chunk, 'content')])
    
    # Count occurrences of domain-specific keywords
    domain_scores = {}
    for domain, keywords in domains.items():
        domain_scores[domain] = sum(all_text.count(keyword.lower()) for keyword in keywords)
    
    # Get the dominant domain
    if not domain_scores or max(domain_scores.values()) == 0:
        return "general"
    
    return max(domain_scores.items(), key=lambda x: x[1])[0]

def get_sample_query(content_type):
    # Get sample queries based on content type from the json file
    queries_file = "config_data_source/sample_queries.json"
    if os.path.exists(queries_file):
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        return random.choice(queries.get(content_type, queries["general"]))
    else:
        # Default queries if file doesn't exist
        default_queries = {
            "general": ["What are the main concepts discussed?", "Summarize this information."],
            "science": ["What scientific principles are explained?", "What experiments are mentioned?"],
            "technology": ["What technologies are discussed?", "Explain the technical concepts mentioned."],
            "politics": ["What political issues are discussed?", "What policies are mentioned?"],
            "gaming": ["What games are mentioned?", "What gameplay mechanics are discussed?"],
            "business": ["What business concepts are covered?", "What market trends are mentioned?"],
            "health": ["What health conditions are discussed?", "What treatments are mentioned?"]
        }
        return random.choice(default_queries.get(content_type, default_queries["general"]))

def main():
    parser = argparse.ArgumentParser(description="AI Services Orchestrator")
    parser.add_argument("--mode", choices=["ingest", "chat", "web", "cli"], default=None,
                        help="Mode to run (ingest, chat, web, cli)")
    parser.add_argument("--urls", nargs="+", help="URLs to scrape")
    parser.add_argument("--files", nargs="+", help="File paths to process")
    
    args = parser.parse_args()
    
    print("=== AI Services Orchestrator ===")
    
    # Initialize orchestrator
    orchestrator = AIServicesOrchestrator()
    
    # If mode is specified, run that mode directly
    if args.mode:
        if args.mode == "ingest":
            # Load URLs and file paths from configuration if not provided
            urls = args.urls
            file_paths = args.files
            
            if not urls and not file_paths:
                urls, file_paths = load_data_sources()
                print(f"Loaded {len(urls)} URLs and {len(file_paths)} file paths from configuration")
            
            # Process data
            result = orchestrator.ingest_data(
                urls=urls if urls else None,
                file_paths=file_paths if file_paths else None
            )
            
            if result["success"]:
                print(f"\nSuccessfully ingested {result['chunks']} chunks")
            else:
                print(f"\nError: {result['message']}")
        
        elif args.mode == "chat":
            print("\nLaunching command line chatbox...")
            import orchestrator_cli
            orchestrator_cli.run_chat_session(orchestrator)
        
        elif args.mode == "web":
            print("\nLaunching web interface on http://127.0.0.1:5050...")
            import web_interface
            web_interface.app.run(debug=True, port=5050)
        
        elif args.mode == "cli":
            print("\nLaunching CLI...")
            import orchestrator_cli
            orchestrator_cli.main()
        
        # Shutdown orchestrator
        orchestrator.shutdown()
        return
    
    # If no mode specified, run the traditional flow
    
    # Load URLs and file paths from configuration
    urls, file_paths = load_data_sources()
    print(f"Loaded {len(urls)} URLs and {len(file_paths)} file paths from configuration")
    
    # Process data
    try:
        ingest_result = orchestrator.ingest_data(
            urls=urls if urls else None,
            file_paths=file_paths if file_paths else None
        )
        
        if ingest_result["success"]:
            print(f"\nProcessed {ingest_result['chunks']} chunks")
            
            # Get a service to detect content type (can be RAG's chunks)
            rag_service = orchestrator.get_service("rag")
            if rag_service:
                chunks = rag_service.vector_store.chunks
                content_type = detect_content_type(chunks)
                query = get_sample_query(content_type)
                print(f"\nDetected content type: {content_type.upper()}")
            else:
                query = "What are the main concepts in this content?"
                print("\nNo content detected. Using generic query.")
            
            # Search and display results by test query
            print("\n=== Search Example ===")
            print(f"Using query: \"{query}\"")
            
            search_result = orchestrator.process_query(query)
            
            if "rag" in search_result["results"]:
                rag_result = search_result["results"]["rag"]
                print(f"\nResponse: {rag_result.get('context', 'No response')}")
                
                # Show sources
                chunks = rag_result.get("retrieved_chunks", [])
                for i, chunk in enumerate(chunks[:3], 1):
                    print(f"\n{i}. Score: {chunk.get('score', 0):.4f}")
                    print(f"Content: {chunk.get('content', '')[:200]}...")
                    if chunk.get("metadata") and "source_document" in chunk["metadata"]:
                        print(f"Source: {chunk['metadata']['source_document']}")
        else:
            print(f"\nError ingesting data: {ingest_result['message']}")
    except Exception as e:
        print(f"\nError: {str(e)}")
    
    # Ask user if they want to launch a chatbot
    launch_choice = input("\nDo you want to launch a chatbot now? (y/n): ").lower().strip()
    
    if launch_choice == 'y':
        interface_choice = input("Choose interface: (1) Command line (2) Web browser: ").strip()
        
        if interface_choice == '1':
            print("\nLaunching command line chatbox...")
            import orchestrator_cli
            orchestrator_cli.run_chat_session(orchestrator)
        elif interface_choice == '2':
            print("\nLaunching web interface on http://127.0.0.1:5050...")
            import web_interface
            web_interface.app.run(debug=True, port=5050)
        else:
            print("Invalid choice. Exiting...")
    
    # Shutdown orchestrator
    orchestrator.shutdown()

if __name__ == "__main__":
    main()
