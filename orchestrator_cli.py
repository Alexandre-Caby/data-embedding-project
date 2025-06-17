import os
import sys
import argparse
from orchestrator import AIServicesOrchestrator

def main():
    parser = argparse.ArgumentParser(description="AI Services Orchestrator CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest data into the RAG pipeline")
    ingest_parser.add_argument("--urls", nargs="+", help="URLs to scrape")
    ingest_parser.add_argument("--files", nargs="+", help="File paths to process")
    
    # Image generation command
    image_parser = subparsers.add_parser("image", help="Generate an image")
    image_parser.add_argument("prompt", help="Image description")
    image_parser.add_argument("--width", type=int, default=1024, help="Image width")
    image_parser.add_argument("--height", type=int, default=1024, help="Image height")
    
    # Web search command
    search_parser = subparsers.add_parser("search", help="Search the web")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--results", type=int, default=5, help="Number of results")
    
    # OS command
    os_parser = subparsers.add_parser("os", help="Execute an OS operation")
    os_parser.add_argument("operation", help="Operation to execute")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize orchestrator
    print("Initializing orchestrator...")
    orchestrator = AIServicesOrchestrator()
    
    # Execute command
    if args.command == "chat":
        run_chat_session(orchestrator)
    elif args.command == "ingest":
        run_ingest(orchestrator, args.urls, args.files)
    elif args.command == "image":
        run_image_generation(orchestrator, args.prompt, args.width, args.height)
    elif args.command == "search":
        run_web_search(orchestrator, args.query, args.results)
    elif args.command == "os":
        run_os_operation(orchestrator, args.operation)
    else:
        parser.print_help()
    
    # Shutdown orchestrator
    orchestrator.shutdown()

def run_chat_session(orchestrator):
    """Run an interactive chat session."""
    print("\nChat with the AI services orchestrator! Type 'quit' to exit.\n")
    
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not user_query.strip():
            continue
            
        print("\nThinking...")
        
        try:
            # Check for specific commands first
            if any(cmd in user_query.lower() for cmd in ['recherche', 'search web', 'search for', 'find online']):
                # This is a web search request
                search_query = user_query
                for prefix in ["recherche sur le web", "recherche", "search web for", "search web", "search for", "find online"]:
                    if search_query.lower().startswith(prefix):
                        search_query = search_query[len(prefix):].strip()
                        break
                
                search_query = search_query.lstrip(',').strip()
                
                if "web_search" in orchestrator.services:
                    print(f"\nSearching the web for: {search_query}")
                    web_results = orchestrator.services["web_search"].search(search_query)
                    
                    if web_results:
                        print(f"\nFound {len(web_results)} web results:")
                        for i, res in enumerate(web_results[:5], 1):
                            print(f"\n{i}. {res.get('title', 'No title')}")
                            print(f"   URL: {res.get('url', 'No URL')}")
                            print(f"   {res.get('snippet', 'No snippet')[:150]}...")
                    else:
                        print("\nNo web results found.")
                else:
                    print("\nWeb search service not available.")
                continue
            
            # For general queries, use the orchestrator
            result = orchestrator.process_query(user_query)
            
            # Handle RAG responses
            if "rag" in result["results"]:
                rag_result = result["results"]["rag"]
                response_text = rag_result.get("context", "No answer found")
                
                # Don't show the default "insufficient information" message for simple greetings
                if "suffisamment d'informations" in response_text and any(greeting in user_query.lower() for greeting in ['hello', 'hi', 'bonjour', 'salut']):
                    print("\nAI Assistant:")
                    print("Bonjour ! Je suis votre assistant IA. Comment puis-je vous aider aujourd'hui ?")
                else:
                    print("\nAI Assistant:")
                    print(response_text)
                
                # Only show sources if they're actually relevant (not the default response)
                chunks = rag_result.get("retrieved_chunks", [])
                if chunks and not "suffisamment d'informations" in response_text:
                    sources = set()
                    for chunk in chunks:
                        source = chunk.get("source", chunk.get("metadata", {}).get("source_document", ""))
                        if source and source != "unknown":
                            sources.add(source)
                    
                    if sources:
                        print(f"\nSources: {', '.join(list(sources)[:3])}")
            
            # Handle image generation
            if "image_generation" in result["results"]:
                img_result = result["results"]["image_generation"]
                if img_result.get("success") and "filepath" in img_result:
                    print(f"\nImage generated: {img_result['filepath']}")
            
            # Handle web search results
            if "web_search" in result["results"]:
                web_results = result["results"]["web_search"]
                if isinstance(web_results, list) and web_results:
                    print(f"\nFound {len(web_results)} web results:")
                    for i, res in enumerate(web_results[:3], 1):
                        print(f"{i}. {res.get('title', 'No title')}")
                        print(f"   URL: {res.get('url', 'No URL')}")
                        print(f"   {res.get('snippet', 'No snippet')[:100]}...")
            
            # Handle OS operations
            if "os_operations" in result["results"]:
                os_result = result["results"]["os_operations"]
                print(f"\nOS operation result: {os_result.get('message', 'No message')}")
        
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try another question or type 'quit' to exit.")

def run_ingest(orchestrator, urls, files):
    """Ingest data into the RAG pipeline."""
    print("\nIngesting data...")
    
    try:
        result = orchestrator.ingest_data(urls=urls, file_paths=files)
        
        if result["success"]:
            print(f"\nSuccess: {result['message']}")
        else:
            print(f"\nError: {result['message']}")
    except Exception as e:
        print(f"\nError: {str(e)}")

def run_image_generation(orchestrator, prompt, width, height):
    """Generate an image."""
    print(f"\nGenerating image for: {prompt}")
    
    try:
        options = {"width": width, "height": height}
        result = orchestrator.generate_image(prompt, options)
        
        if result["success"] and "data" in result and "filepath" in result["data"]:
            print(f"\nImage generated: {result['data']['filepath']}")
        else:
            print(f"\nError: {result.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"\nError: {str(e)}")

def run_web_search(orchestrator, query, num_results):
    """Search the web."""
    print(f"\nSearching for: {query}")
    
    try:
        result = orchestrator.search_web(query, num_results)
        
        if result["success"] and "results" in result:
            web_results = result["results"]
            print(f"\nFound {len(web_results)} results:")
            
            for i, res in enumerate(web_results, 1):
                print(f"{i}. {res.get('title', 'No title')}")
                print(f"   URL: {res.get('url', 'No URL')}")
                print(f"   {res.get('snippet', 'No snippet')[:100]}...")
        else:
            print(f"\nError: {result.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"\nError: {str(e)}")

def run_os_operation(orchestrator, operation):
    """Execute an OS operation."""
    print(f"\nExecuting OS operation: {operation}")
    
    try:
        result = orchestrator.execute_os_command(operation)
        
        if result["success"]:
            print(f"\nSuccess: {result['message']}")
            if "result" in result and result["result"]:
                print("\nResult:")
                print(result["result"])
        else:
            print(f"\nError: {result.get('message', 'Unknown error')}")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()