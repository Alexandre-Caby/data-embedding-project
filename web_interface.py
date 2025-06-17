from flask import Flask, render_template, request, jsonify, session, url_for, send_from_directory
import os
import base64
from orchestrator import AIServicesOrchestrator

app = Flask(__name__)
app.config['SECRET_KEY'] = '1234'

# Initialize the orchestrator
orchestrator = None
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

def init_orchestrator():
    global orchestrator
    try:
        orchestrator = AIServicesOrchestrator()
        return True
    except Exception as e:
        print(f"Error initializing orchestrator: {e}")
        return False

@app.route('/')
def index():
    # Reset conversation for new sessions
    if not orchestrator:
        init_orchestrator()
    if orchestrator and 'conversation_started' not in session:
        orchestrator.reset_services()
        session['conversation_started'] = True
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if not orchestrator:
        if not init_orchestrator():
            return jsonify({"error": "Failed to initialize orchestrator"}), 500
            
    data = request.json
    message = data.get('message', '')
    if not message:
        return jsonify({"error": "No message provided"}), 400
    
    try:
        # Process query through orchestrator
        result = orchestrator.process_query(message)
        
        # Prepare the response
        response = {
            "context": "I couldn't process your request.",
            "retrieved_chunks": [],
            "images": [],
            "web_results": [],
            "system_info": {}
        }
        
        # Handle RAG responses
        if "rag" in result["results"]:
            rag_result = result["results"]["rag"]
            response["context"] = rag_result.get("context", "No answer found")
            response["retrieved_chunks"] = rag_result.get("retrieved_chunks", [])
        
        # Handle image generation results
        if "image_generation" in result["results"]:
            img_result = result["results"]["image_generation"]
            if img_result.get("success") and "filepath" in img_result:
                # Get relative path for the image
                rel_path = os.path.relpath(img_result["filepath"], os.path.dirname(__file__))
                img_url = url_for('static', filename=rel_path.replace('\\', '/')) if "static" in rel_path else f"/output/{os.path.basename(img_result['filepath'])}"
                response["images"].append({
                    "url": img_url,
                    "prompt": img_result.get("prompt", "")
                })
                response["context"] = f"I've generated an image based on your request: '{img_result.get('prompt', '')}'"
        
        # Handle web search results
        if "web_search" in result["results"]:
            web_results = result["results"]["web_search"]
            if isinstance(web_results, list):
                response["web_results"] = web_results
                response["context"] = f"I found {len(web_results)} results from the web for your query."
        
        # Handle OS operations
        if "os_operations" in result["results"]:
            os_result = result["results"]["os_operations"]
            response["system_info"] = os_result
            if "message" in os_result:
                response["context"] = os_result["message"]
        
        return jsonify(response)
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    if not orchestrator:
        init_orchestrator()
    
    if orchestrator:
        result = orchestrator.reset_services()
        return jsonify({"message": "Conversation and services reset successfully", "details": result})
    return jsonify({"error": "Orchestrator not initialized"}), 500

@app.route('/output/<path:filename>')
def serve_output(filename):
    """Serve files from the output directory."""
    return send_from_directory(output_dir, filename)

@app.route('/generate_image', methods=['POST'])
def generate_image():
    """Direct endpoint for image generation."""
    if not orchestrator:
        if not init_orchestrator():
            return jsonify({"error": "Failed to initialize orchestrator"}), 500
    
    data = request.json
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    options = data.get('options', {})
    
    try:
        result = orchestrator.generate_image(prompt, options)
        if result["success"] and "data" in result and "filepath" in result["data"]:
            # Get relative path for the image
            filename = os.path.basename(result["data"]["filepath"])
            return jsonify({
                "success": True,
                "image_url": f"/output/{filename}",
                "message": result["message"]
            })
        else:
            return jsonify({
                "success": False,
                "message": result.get("message", "Unknown error")
            })
    except Exception as e:
        print(f"Error generating image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/search_web', methods=['POST'])
def search_web():
    """Direct endpoint for web search."""
    if not orchestrator:
        if not init_orchestrator():
            return jsonify({"error": "Failed to initialize orchestrator"}), 500
    
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    num_results = data.get('num_results', 5)
    
    try:
        result = orchestrator.search_web(query, num_results)
        return jsonify(result)
    except Exception as e:
        print(f"Error searching web: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_orchestrator()
    # Run on port 5050 to avoid conflicts with the OpenAI API server on port 8080
    app.run(debug=True, port=5050)