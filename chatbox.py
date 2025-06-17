import os
import sys
from rag_pipeline import create_rag_pipeline

def format_response(response_text):
    """Format the response text to be more readable."""
    # Remove extra whitespace and newlines
    formatted = response_text.strip()
    # Ensure sentences end with proper punctuation and spacing
    return formatted

def main():
    print("=== RAG Chatbox ===")
    
    # Define the load path explicitly
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        print("Did you run the main.py script first to embed your documents?")
        return
    
    print("\nLoading the RAG pipeline...")
    
    try:
        # Create the pipeline with simpler setup - avoid local model issues
        pipeline = create_rag_pipeline()
        
        # Load with explicit path
        pipeline.load(data_dir)
        
        print("\nChat with your documents! Type 'quit' to exit.\n")
        
        while True:
            user_query = input("\nYou: ")
            if user_query.lower() in ['quit', 'exit', 'q']:
                break
                
            if not user_query.strip():
                continue
                
            print("\nThinking...")
            
            try:
                # Generate response using existing RAG pipeline
                response = pipeline.generate_response(user_query)
                
                # Format and print the answer
                formatted_response = format_response(response['context'])
                print("\nAI Assistant:")
                print(formatted_response)
                
                # Show sources
                sources = set([chunk['source'] for chunk in response['retrieved_chunks']])
                if sources:
                    print("\nSources:", ", ".join(sources))
            
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try another question or type 'quit' to exit.")
    
    except Exception as e:
        print(f"Failed to initialize RAG pipeline: {str(e)}")
        print("Please make sure you've run main.py first to process your documents.")
        sys.exit(1)

if __name__ == "__main__":
    main()