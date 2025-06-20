import json
import os
import argparse
from orchestrator import AIServicesOrchestrator

def load_data_sources(config_file="config/data_sources.json"):
    """Load data sources from configuration file."""
    if not os.path.exists(config_file):
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        default_config = {"urls": [], "file_paths": []}
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2)
        print(f"Created default configuration at {config_file}")
        return [], []
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get("urls", []), config.get("file_paths", [])
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return [], []

def main():
    parser = argparse.ArgumentParser(description="AI Services Orchestrator")
    parser.add_argument("--mode", choices=["ingest", "web"], 
                       help="Mode to run", required=False)
    parser.add_argument("--urls", nargs="+", help="URLs to scrape")
    parser.add_argument("--files", nargs="+", help="File paths to process")
    
    args = parser.parse_args()
    
    print("=== AI Services Orchestrator ===")
    orchestrator = AIServicesOrchestrator()
    
    try:
        if args.mode == "ingest":
            urls = args.urls or []
            files = args.files or []
            
            if not urls and not files:
                urls, files = load_data_sources()
            
            result = orchestrator.ingest_data(urls or None, files or None)
            print(f"Result: {result['message']}")
            
        elif args.mode == "web":
            print("Starting web interface on http://127.0.0.1:5050...")
            import web_interface
            web_interface.app.run(debug=False, port=5050)
            
        else:
            # Interactive mode
            urls, files = load_data_sources()
            
            if urls or files:
                result = orchestrator.ingest_data(urls or None, files or None)
                print(f"Ingested: {result.get('chunks', 0)} chunks")
            
            choice = input("\nChoose interface: (1) Web browser (3) Exit: ").strip()
            
            if choice == '1':
                print("Starting web interface on http://127.0.0.1:5050...")
                import web_interface
                web_interface.app.run(debug=False, port=5050)
            
    finally:
        orchestrator.shutdown()

if __name__ == "__main__":
    main()
