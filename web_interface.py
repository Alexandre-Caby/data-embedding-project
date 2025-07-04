from flask import Flask, render_template, request, jsonify, session, url_for, send_from_directory
import os
import logging
import traceback
from orchestrator import AIServicesOrchestrator

app = Flask(__name__)
app.config['SECRET_KEY'] = '1234'

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("web_interface")

# Initialize the orchestrator
orchestrator = None
output_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)

def init_orchestrator(force=False):
    """Initialize the orchestrator with more robust error handling"""
    global orchestrator
    
    if orchestrator is not None and not force:
        logger.info("Using existing orchestrator instance")
        return True
    
    try:
        logger.info("Initializing orchestrator...")
        orchestrator = AIServicesOrchestrator()
        
        # Verify services are available
        services = list(orchestrator.services.keys())
        # logger.info(f"Orchestrator initialized with services: {', '.join(services)}")
        
        # Verify RAG service specifically
        if "rag" in orchestrator.services:
            rag = orchestrator.services["rag"]
            if hasattr(rag, 'vector_store') and hasattr(rag.vector_store, 'chunks'):
                num_chunks = len(rag.vector_store.chunks)
                logger.info(f"RAG loaded with {num_chunks} document chunks")
            else:
                logger.warning("RAG service initialized but no chunks loaded")
        
        # Verify web search service
        if "web_search" in orchestrator.services:
            logger.info(f"Web search service available using provider: {orchestrator.services['web_search'].provider}")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing orchestrator: {e}")
        logger.error(traceback.format_exc())
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
    """Enhanced chat endpoint with better error handling and higher quality responses by default"""
    global orchestrator
    
    # Ensure orchestrator is initialized
    if orchestrator is None:
        success = init_orchestrator()
        if not success:
            logger.error("Failed to initialize orchestrator")
            return jsonify({
                "context": "Sorry, I'm having trouble connecting to the required services. Please try again later.",
                "error": "Orchestrator initialization failed"
            }), 500
    
    # Get message from request
    try:
        data = request.json
        message = data.get('message', '').strip()
        if not message:
            return jsonify({"error": "No message provided"}), 400
    except Exception as e:
        logger.error(f"Error parsing request: {e}")
        return jsonify({"error": "Invalid request format"}), 400
    
    # Process the message
    try:
        logger.info(f"Processing query: {message[:50]}{'...' if len(message) > 50 else ''}")
        
        # Check for system operations first
        query_lower = message.lower().strip()
        if any(keyword in query_lower for keyword in [
            'system info', 'system information', 'computer info',
            'disk space', 'storage info', 'memory usage', 'ram usage',
            'cpu usage', 'processor usage', 'list files', 'list directory',
            'read file', 'show content', 'display file'
        ]):
            if "os_operations" in orchestrator.services:
                try:
                    os_result = orchestrator.services["os_operations"].process_command(message)
                    
                    # Format the OS result for display
                    if os_result.get("success"):
                        if os_result.get("command_type") == "system_info":
                            context = f"**System Information:**\n\n"
                            sys_info = os_result.get("system_info", {})
                            context += f"- **System:** {sys_info.get('system', 'N/A')}\n"
                            context += f"- **Node:** {sys_info.get('node', 'N/A')}\n"
                            context += f"- **Release:** {sys_info.get('release', 'N/A')}\n"
                            context += f"- **Machine:** {sys_info.get('machine', 'N/A')}\n"
                            context += f"- **CPU Count:** {sys_info.get('cpu_count', 'N/A')}\n"
                            context += f"- **Total Memory:** {sys_info.get('memory_total_human', 'N/A')}\n"
                        
                        elif os_result.get("command_type") == "list_directory":
                            context = f"**Directory listing for:** {os_result.get('path', 'Unknown')}\n\n"
                            contents = os_result.get("contents", {})
                            directories = contents.get("directories", [])
                            files = contents.get("files", [])
                            
                            if directories:
                                context += "**Directories:**\n"
                                for dir_item in directories:
                                    context += f"📁 {dir_item['name']}\n"
                                context += "\n"
                            
                            if files:
                                context += "**Files:**\n"
                                for file_item in files:
                                    context += f"📄 {file_item['name']} ({file_item.get('size_human', 'Unknown size')})\n"
                            
                            context += f"\n**Total items:** {len(directories) + len(files)}"
                        
                        elif os_result.get("command_type") == "memory_usage":
                            context = "**Memory Usage:**\n\n"
                            mem_info = os_result.get("memory_info", {})
                            context += f"- **Total:** {mem_info.get('total_human', 'N/A')}\n"
                            context += f"- **Used:** {mem_info.get('used_human', 'N/A')} ({mem_info.get('percent_used', 'N/A')}%)\n"
                            context += f"- **Available:** {mem_info.get('available_human', 'N/A')}\n"
                            if mem_info.get('swap_total_human'):
                                context += f"- **Swap Total:** {mem_info.get('swap_total_human', 'N/A')}\n"
                                context += f"- **Swap Used:** {mem_info.get('swap_used_human', 'N/A')} ({mem_info.get('swap_percent_used', 'N/A')}%)\n"
                        
                        elif os_result.get("command_type") == "cpu_usage":
                            context = "**CPU Usage:**\n\n"
                            cpu_info = os_result.get("cpu_info", {})
                            context += f"- **Overall Usage:** {cpu_info.get('percent', 'N/A')}%\n"
                            if cpu_info.get('frequency'):
                                freq = cpu_info['frequency']
                                context += f"- **Current Frequency:** {freq.get('current', 'N/A')} MHz\n"
                        
                        elif os_result.get("command_type") == "disk_space":
                            context = "**Disk Space:**\n\n"
                            for disk in os_result.get("disk_info", []):
                                context += f"**{disk.get('device', 'Unknown')}** ({disk.get('filesystem', 'Unknown')})\n"
                                context += f"- **Total:** {disk.get('total_human', 'N/A')}\n"
                                context += f"- **Used:** {disk.get('used_human', 'N/A')} ({disk.get('percent_used', 'N/A')}%)\n"
                                context += f"- **Free:** {disk.get('free_human', 'N/A')}\n\n"
                        
                        elif os_result.get("command_type") == "write_file":
                            context = f"**File Created Successfully!**\n\n"
                            context += f"📄 **Filename:** `{os.path.basename(os_result.get('path', 'Unknown'))}`\n"
                            context += f"📁 **Path:** `{os_result.get('path', 'Unknown')}`\n"
                            context += f"📏 **Size:** {os_result.get('size_human', 'Unknown')}\n\n"
                            context += f"The file has been created with your specified content."
                        else:
                            context = os_result.get("message", "Operation completed successfully")
                    else:
                        context = f"Error: {os_result.get('message', 'Unknown error occurred')}"
                    
                    return jsonify({
                        "context": context,
                        "retrieved_chunks": [],
                        "system_info": os_result
                    })
                    
                except Exception as e:
                    logger.error(f"OS operations error: {e}")
                    return jsonify({
                        "context": f"Error executing system operation: {str(e)}",
                        "retrieved_chunks": []
                    })
        
        # Handle web search
        if message.lower().startswith(('search web', 'search for', 'find online')):
            logger.info("Detected web search query")
            
            # Extract the search query
            search_query = message
            for prefix in ["search web for", "search web", "search for", "find online"]:
                if search_query.lower().startswith(prefix):
                    search_query = search_query[len(prefix):].strip()
                    break
            
            # Remove any remaining colons or extra text
            search_query = search_query.lstrip(':').strip()
            
            # Direct call to web search for reliability
            if "web_search" in orchestrator.services:
                logger.info(f"Calling web_search service directly with query: {search_query}")
                web_results = orchestrator.services["web_search"].search(search_query)
                
                # Process web search results to extract knowledge
                web_documents = orchestrator.process_web_search_results(web_results)
                
                # Create context from web documents OR search snippets as fallback
                web_context = ""
                if web_documents:
                    # Use scraped content
                    for i, doc in enumerate(web_documents):
                        web_context += f"\n\n--- Source {i+1}: {doc.metadata.get('title', 'Web Source')} ---\n"
                        web_context += doc.content[:2000]
                    logger.info(f"Using scraped content from {len(web_documents)} documents")
                else:
                    # Fallback: use search result snippets
                    logger.info("Web scraping failed, using search snippets as fallback")
                    for i, result in enumerate(web_results[:3]):
                        web_context += f"\n\n--- Résultat {i+1}: {result.get('title', 'Sans titre')} ---\n"
                        web_context += result.get('snippet', 'Pas de description disponible')
                
                # Generate a response based on the web content using RAG
                if "rag" in orchestrator.services and web_context:
                    try:
                        # Use the RAG service to generate a comprehensive response
                        rag_result = orchestrator.services["rag"]._handle_complex_query(
                            query=search_query,
                            web_context=web_context,
                            top_k=3
                        )
                        
                        return jsonify({
                            "context": rag_result.get("context", f"Basé sur les résultats de recherche pour '{search_query}':"),
                            "web_results": web_results,
                            "retrieved_chunks": []
                        })
                    except Exception as e:
                        logger.error(f"Error processing web content with RAG: {e}")
                
                # Final fallback: return search results with basic summary
                summary = f"J'ai trouvé {len(web_results)} résultats pour '{search_query}'. "
                if web_context:
                    # Create a basic summary from the available context
                    summary += "Voici ce que j'ai trouvé : " + web_context[:500] + "..."
                
                return jsonify({
                    "context": summary,
                    "web_results": web_results,
                    "retrieved_chunks": []
                })
        
        # Process through orchestrator for other queries
        result = orchestrator.process_query(message, quality='high')
        
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
            logger.info(f"RAG found {len(response['retrieved_chunks'])} relevant chunks")
        
        # Handle other types of results (image, web search, etc.)
        if "image_generation" in result["results"]:
            # Process image results...
            img_result = result["results"]["image_generation"]
            # Correction ici : accès attributs objet, pas .get()
            if getattr(img_result, "success", False) and getattr(img_result, "filepath", None):
                # Get relative path for the image
                filename = os.path.basename(img_result.filepath)
                response["images"].append({
                    "url": f"/output/{filename}",
                    "prompt": getattr(img_result, "prompt", "")
                })
                response["context"] = f"I've generated an image based on your request"
        
        # Handle web search results
        if "web_search" in result["results"]:
            web_results = result["results"]["web_search"]
            if isinstance(web_results, list):
                response["web_results"] = web_results
                response["context"] = f"I found {len(web_results)} results from the web for your query."
                logger.info(f"Web search found {len(web_results)} results")
        
        # Handle OS operations
        if "os_operations" in result["results"]:
            os_result = result["results"]["os_operations"]
            response["system_info"] = os_result
            if "message" in os_result:
                response["context"] = os_result["message"]

        # Handle image generation results
        if 'image_generation' in result['results']:
            img_res = result['results']['image_generation']
            if getattr(img_res, "success", False) and getattr(img_res, "filepath", None):
                # Build URL to the generated image in output/images
                filename = os.path.basename(img_res.filepath)
                img_url = url_for('serve_output', filename=f'images/{filename}')
                response['images'].append({
                    'url': img_url,
                    'prompt': getattr(img_res, "prompt", "")
                })
                response['context'] = "Voici l'image générée :"

        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        logger.error(traceback.format_exc())
        
        # Try to reinitialize the orchestrator on failure
        try:
            init_orchestrator(force=True)
        except:
            pass
        
        return jsonify({
            "context": "Sorry, I encountered an error processing your request. Please try again.",
            "error": str(e)
        }), 500

@app.route('/direct_search', methods=['POST'])
def direct_search():
    """Simplified endpoint that only does web search"""
    global orchestrator
    
    if not orchestrator:
        init_orchestrator()
    
    try:
        data = request.json
        query = data.get('query', '')
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Use the web search service directly
        if "web_search" in orchestrator.services:
            results = orchestrator.services["web_search"].search(query)
            return jsonify({
                "success": True, 
                "results": results,
                "message": f"Found {len(results)} results"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Web search service not available"
            })
    
    except Exception as e:
        logger.error(f"Error in direct search: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    """Reset conversation and services"""
    if not orchestrator:
        init_orchestrator()
    
    if orchestrator:
        result = orchestrator.reset_services()
        return jsonify({"message": "Conversation and services reset successfully", "details": result})
    return jsonify({"error": "Orchestrator not initialized"}), 500

@app.route('/reset_conversation', methods=['POST'])
def reset_conversation():
    """Reset the conversation history."""
    if not orchestrator:
        init_orchestrator()
    
    if orchestrator:
        result = orchestrator.reset_services()
        return jsonify({"message": "Conversation reset successfully", "details": result})
    return jsonify({"error": "Orchestrator not initialized"}), 500

@app.route('/output/<path:filename>')
def serve_output(filename):
    """Serve files from the output directory."""
    return send_from_directory(output_dir, filename)

@app.route('/styles.css')
def serve_css():
    return send_from_directory('templates', 'style.css', mimetype='text/css')

@app.route('/script.js')
def serve_js():
    return send_from_directory('templates', 'script.js', mimetype='application/javascript')

@app.route('/status', methods=['GET'])
def status():
    """Get status of orchestrator and services"""
    if not orchestrator:
        success = init_orchestrator()
        if not success:
            return jsonify({"status": "error", "message": "Failed to initialize orchestrator"}), 500
    
    services = {}
    for name, service in orchestrator.services.items():
        if name == "rag" and hasattr(service.vector_store, "chunks"):
            services[name] = {
                "status": "active",
                "chunks": len(service.vector_store.chunks)
            }
        else:
            services[name] = {"status": "active"}
    
    return jsonify({
        "status": "ok",
        "services": services
    })

if __name__ == "__main__":
    init_orchestrator()
    # Run on port 5050 to avoid conflicts with the OpenAI API server on port 8080
    app.run(debug=True, port=5050)