import os
import sys
import argparse
from orchestrator import AIServicesOrchestrator

def run_chat_session(orchestrator_instance=None):
    """Run an interactive chat session."""
    if orchestrator_instance:
        orchestrator = orchestrator_instance
    else:
        print("Initializing AI Services Orchestrator...")
        orchestrator = AIServicesOrchestrator()
    
    print("\n" + "="*60)
    print("🤖 AI Services Orchestrator - Interactive Chat")
    print("="*60)
    print("Available commands:")
    print("• Just ask questions normally")
    print("• 'search web for [query]' - Web search")
    print("• 'generate image [description]' - Image generation")
    print("• 'system info' - System information")
    print("• 'help' - Show this help")
    print("• 'reset' - Clear conversation history")
    print("• 'quit' or 'exit' - Exit chat")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye!")
                break
                
            if user_input.lower() == 'reset':
                orchestrator.reset_services()
                print("\n🔄 Conversation history reset!")
                continue
                
            if user_input.lower() in ['help', 'aide']:
                print("\n📋 Available services:")
                print("• RAG: Ask questions about loaded documents")
                print("• Web Search: 'search web for [query]'")
                print("• Image Generation: 'generate image [description]'")
                print("• System Info: 'system info'")
                continue
            
            print("\n🤔 Thinking...")
            
            # Process the query through the orchestrator
            result = orchestrator.process_query(user_input)
            
            if not result.get("success", True):
                print(f"\n❌ Error: {result.get('message', 'Unknown error')}")
                continue
            
            # Handle different types of results
            response_shown = False
            
            # RAG response
            if "rag" in result.get("results", {}):
                rag_result = result["results"]["rag"]
                if "context" in rag_result:
                    print(f"\n🤖 AI Assistant:\n{rag_result['context']}")
                    
                    # Show sources if available
                    chunks = rag_result.get("retrieved_chunks", [])
                    if chunks and not rag_result['context'].startswith("Je n'ai pas"):
                        print(f"\n📚 Sources utilisées ({len(chunks)} documents):")
                        for i, chunk in enumerate(chunks[:3], 1):
                            source = chunk.get('metadata', {}).get('source_document', 'Source inconnue')
                            score = chunk.get('score', 0)
                            print(f"  {i}. {source} (score: {score:.3f})")
                    response_shown = True
            
            # Web search results
            if "web_search" in result.get("results", {}):
                web_results = result["results"]["web_search"]
                if isinstance(web_results, list) and web_results:
                    print(f"\n🔍 Résultats de recherche web ({len(web_results)} résultats):")
                    for i, item in enumerate(web_results[:5], 1):
                        title = item.get('title', 'Sans titre')
                        url = item.get('url', '')
                        snippet = item.get('snippet', 'Pas de description')
                        print(f"\n{i}. {title}")
                        print(f"   URL: {url}")
                        print(f"   Description: {snippet[:150]}...")
                    response_shown = True
            
            # Image generation results
            if "image_generation" in result.get("results", {}):
                img_result = result["results"]["image_generation"]
                if img_result.get("success"):
                    filepath = img_result.get("filepath", "")
                    prompt = img_result.get("prompt", "")
                    print(f"\n🎨 Image générée avec succès!")
                    print(f"Prompt: {prompt}")
                    print(f"Fichier: {filepath}")
                else:
                    print(f"\n❌ Erreur lors de la génération d'image: {img_result.get('message', 'Erreur inconnue')}")
                response_shown = True
            
            # System operations results
            if "os_operations" in result.get("results", {}):
                os_result = result["results"]["os_operations"]
                if "message" in os_result:
                    print(f"\n💻 Informations système:\n{os_result['message']}")
                else:
                    print(f"\n💻 Résultat système: {os_result}")
                response_shown = True
            
            # If no specific result was shown, show a generic message
            if not response_shown:
                print(f"\n🤖 Réponse reçue mais aucun contenu à afficher.")
                print(f"Résultats disponibles: {', '.join(result.get('results', {}).keys())}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"\n❌ Erreur inattendue: {str(e)}")
            print("Tapez 'quit' pour quitter ou continuez à poser des questions.")

def main():
    """Main CLI entry point with subcommands."""
    parser = argparse.ArgumentParser(description="AI Services Orchestrator CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Start interactive chat session')
    
    # Quick commands
    search_parser = subparsers.add_parser('search', help='Quick web search')
    search_parser.add_argument('query', nargs='+', help='Search query')
    
    image_parser = subparsers.add_parser('image', help='Generate image')
    image_parser.add_argument('description', nargs='+', help='Image description')
    
    os_parser = subparsers.add_parser('os', help='System operations')
    os_parser.add_argument('command', nargs='+', help='OS command')
    
    args = parser.parse_args()
    
    # If no command specified, default to chat
    if not args.command:
        run_chat_session()
        return
    
    # Initialize orchestrator for other commands
    orchestrator = AIServicesOrchestrator()
    
    if args.command == 'chat':
        run_chat_session(orchestrator)
        
    elif args.command == 'search':
        query = ' '.join(args.query)
        print(f"🔍 Searching for: {query}")
        result = orchestrator.search_web(query)
        
        if result["success"]:
            results = result["results"]
            print(f"\n✅ Found {len(results)} results:")
            for i, item in enumerate(results[:5], 1):
                print(f"\n{i}. {item.get('title', 'No title')}")
                print(f"   {item.get('url', '')}")
                print(f"   {item.get('snippet', 'No description')[:150]}...")
        else:
            print(f"❌ Search failed: {result['message']}")
            
    elif args.command == 'image':
        description = ' '.join(args.description)
        print(f"🎨 Generating image: {description}")
        result = orchestrator.generate_image(description)
        
        if result["success"]:
            filepath = result["data"].get("filepath", "")
            print(f"✅ Image generated successfully!")
            print(f"File: {filepath}")
        else:
            print(f"❌ Image generation failed: {result['message']}")
            
    elif args.command == 'os':
        command = ' '.join(args.command)
        print(f"💻 Executing: {command}")
        result = orchestrator.execute_os_command(command)
        
        if result["success"]:
            print(f"✅ Command executed:")
            print(result["result"])
        else:
            print(f"❌ Command failed: {result['message']}")

if __name__ == "__main__":
    main()