document.addEventListener('DOMContentLoaded', function() {
    const messagesContainer = document.getElementById('messagesContainer');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const clearButton = document.getElementById('clearChat');
    const emptyState = document.getElementById('emptyState');
    
    let isWaitingForResponse = false;
    
    // Track selected service
    let selectedService = null;
    
    // Auto-resize the textarea
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        
        // Enable send button if there's text or a service is selected
        sendButton.disabled = !this.value.trim() && !selectedService;
    });
    
    // Send message when Enter key is pressed (without Shift)
    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Send message when button is clicked
    sendButton.addEventListener('click', sendMessage);
    
    // Clear chat history
    clearButton.addEventListener('click', function() {
        // Send a request to reset the conversation
        fetch('/reset_conversation', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            messagesContainer.innerHTML = '';
            showEmptyState();
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
    
    // Update event listeners for action buttons
    document.getElementById('imageGenBtn').addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation(); // Prevent event bubbling
        toggleService('image');
    });
    
    document.getElementById('webSearchBtn').addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation(); // Prevent event bubbling
        toggleService('web_search');
    });
    
    document.getElementById('systemInfoBtn').addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation(); // Prevent event bubbling
        toggleService('system');
    });
    
    // Create separate direct search function for web search button
    function performWebSearch() {
        // Get the input field value
        const query = messageInput.value.trim() || "";
        
        // Add appropriate web search prefix if needed
        const searchQuery = selectedService === 'web_search' && query
            ? query
            : `Search the web for: ${query || "latest news"}`;
        
        // Set the message input to the search query
        messageInput.value = searchQuery;
        
        // Send the message
        sendMessage();
    }
    
    // Toggle service selection
    function toggleService(service) {
        console.log(`Toggling service: ${service}, currently selected: ${selectedService}`);
        
        const buttonMap = {
            'image': 'imageGenBtn',
            'web_search': 'webSearchBtn',
            'system': 'systemInfoBtn'
        };
        
        // Get all service buttons
        const buttons = {
            'image': document.getElementById('imageGenBtn'),
            'web_search': document.getElementById('webSearchBtn'),
            'system': document.getElementById('systemInfoBtn')
        };
        
        // If clicking the same service, deselect it
        if (selectedService === service) {
            console.log(`Deselecting ${service}`);
            selectedService = null;
            
            // Remove 'selected' class from the button
            buttons[service].classList.remove('selected');
            
            // Reset placeholder
            updatePlaceholder();
            return;
        }
        
        // Clear all selections first
        Object.values(buttons).forEach(btn => {
            if (btn) btn.classList.remove('selected');
        });
        
        // Select the new service
        selectedService = service;
        
        // Add 'selected' class to the selected button
        if (buttons[service]) {
            buttons[service].classList.add('selected');
            console.log(`Selected ${service} button`);
        }
        
        // Focus the input field after selecting a service
        messageInput.focus();
        
        // Update placeholder to indicate the selected service
        updatePlaceholder();
    }
    
    // Update placeholder based on selected service
    function updatePlaceholder() {
        const placeholderMap = {
            'image': 'Describe the image you want to generate...',
            'web_search': 'Enter your web search query...',
            'system': 'Ask for system information...',
            null: 'Type your question here'
        };
        
        messageInput.placeholder = placeholderMap[selectedService];
    }
    
    function sendMessage() {
        const message = messageInput.value.trim();

        if ((!message && !selectedService) || isWaitingForResponse) {
            return;
        }

        hideEmptyState();

        let processedMessage = message;

        // Adjust message based on selected service if needed
        if (selectedService) {
            console.log(`Sending with selected service: ${selectedService}`);
            
            switch(selectedService) {
                case 'image':
                    processedMessage = message.startsWith('Generate an image') 
                        ? message 
                        : `Generate an image of: ${message || 'a creative abstract design'}`;
                    break;
                    
                case 'web_search':
                    processedMessage = message.startsWith('Search web for') || message.startsWith('Search the web for')
                        ? message
                        : `Search web for: ${message || 'latest technology trends'}`;
                    break;
                    
                case 'system':
                    processedMessage = message || 'Get system information';
                    break;
            }
        }

        // Add user message to chat
        addUserMessage(processedMessage);

        // Clear input and reset selection
        messageInput.value = '';
        messageInput.style.height = '60px';

        // Deselect the current service if one is selected
        if (selectedService) {
            const currentService = selectedService;
            toggleService(currentService); // This will deselect since it's the same service
        }

        // Show typing indicator
        addTypingIndicator();

        // Disable input while waiting
        isWaitingForResponse = true;
        sendButton.disabled = true;

        // Send message to backend
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: processedMessage })
        })
        .then(response => response.json())
        .then(data => {
            // Remove typing indicator
            removeTypingIndicator();
            
            // Add assistant message with all result types
            addAssistantMessage(
                data.context, 
                data.retrieved_chunks, 
                data.images || [], 
                data.web_results || []
            );
            
            // Enable input
            isWaitingForResponse = false;
            sendButton.disabled = false;
            messageInput.focus();
        })
        .catch(error => {
            console.error('Error:', error);
            removeTypingIndicator();
            addAssistantMessage("Sorry, I encountered an error processing your request.", [], [], []);
            isWaitingForResponse = false;
            sendButton.disabled = false;
        });
    }
    
    function addUserMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message user-message';
        messageElement.innerHTML = `
            <div class="message-content">${escapeHtml(message)}</div>
        `;
        messagesContainer.appendChild(messageElement);
        scrollToBottom();
    }
    
    function addAssistantMessage(message, chunks, images, webResults) {
        const messageElement = document.createElement('div');
        messageElement.className = 'message assistant-message';
        
        let sourcesHtml = '';
        let sourcesDropdownHtml = '';
        
        // Process sources for RAG results
        if (chunks && chunks.length > 0) {
            // Create source pills
            const uniqueSources = new Set();
            chunks.forEach(chunk => {
                if (chunk.metadata && chunk.metadata.source_document) {
                    uniqueSources.add(chunk.metadata.source_document);
                }
            });
            
            if (uniqueSources.size > 0) {
                sourcesHtml = '<div class="message-sources">';
                Array.from(uniqueSources).forEach((source, index) => {
                    sourcesHtml += `<span class="source-item" data-message-id="${Date.now()}">${escapeHtml(source)}</span>`;
                });
                sourcesHtml += '</div>';
            }
            
            // Create detailed sources dropdown
            sourcesDropdownHtml = `
                <div class="sources-dropdown" id="sources-${Date.now()}">
                    <div class="source-header">Sources</div>
            `;
            
            chunks.slice(0, 3).forEach(chunk => {
                sourcesDropdownHtml += `
                    <div class="source-content">
                        ${escapeHtml(chunk.content.substring(0, 200))}${chunk.content.length > 200 ? '...' : ''}
                        <div class="source-score">Relevance: ${Math.round(chunk.score * 100)}%</div>
                    </div>
                `;
            });
            
            sourcesDropdownHtml += '</div>';
        }
        
        // Process images
        let imagesHtml = '';
        if (images && images.length > 0) {
            images.forEach(image => {
                imagesHtml += `
                    <div class="image-result">
                        <img src="${image.url}" alt="${escapeHtml(image.prompt)}" />
                    </div>
                `;
            });
        }
        
        // Process web results
        let webResultsHtml = '';
        if (webResults && webResults.length > 0) {
            webResultsHtml = `
                <div class="web-results">
                    <div class="web-results-header">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="11" cy="11" r="8"></circle>
                            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                        </svg>
                        <span>Search Results (${webResults.length})</span>
                    </div>
            `;
            
            webResults.forEach(result => {
                // Clean the URL before using it
                const cleanUrl = result.url.trim().replace(/\s+/g, '');
                const domain = cleanUrl.split('/')[2] || '';
                
                webResultsHtml += `
                    <div class="web-result-item">
                        <div class="web-result-title">
                            <a href="${cleanUrl}" target="_blank" rel="noopener noreferrer" data-original-url="${cleanUrl}">
                                ${escapeHtml(result.title)}
                            </a>
                        </div>
                        <div class="web-result-url">
                            ${escapeHtml(domain)}
                        </div>
                        <div class="web-result-snippet">
                            ${escapeHtml(result.snippet)}
                        </div>
                    </div>
                `;
            });
            
            webResultsHtml += `</div>`;
        }
        
        messageElement.innerHTML = `
            <div>
                <div class="message-content">${formatMessage(message)}</div>
                ${sourcesHtml}
                ${sourcesDropdownHtml}
                ${imagesHtml}
                ${webResultsHtml}
            </div>
        `;
        
        messagesContainer.appendChild(messageElement);
        
        // Add event listeners to source items
        if (chunks && chunks.length > 0) {
            const sourceItems = messageElement.querySelectorAll('.source-item');
            const sourcesDropdown = messageElement.querySelector('.sources-dropdown');
            
            sourceItems.forEach(item => {
                item.addEventListener('click', function() {
                    sourcesDropdown.classList.toggle('active');
                });
            });
        }
        
        scrollToBottom();
    }
    
    function addTypingIndicator() {
        const typingElement = document.createElement('div');
        typingElement.className = 'typing-indicator';
        typingElement.id = 'typingIndicator';
        typingElement.innerHTML = `
            <div class="typing-bubble"></div>
            <div class="typing-bubble"></div>
            <div class="typing-bubble"></div>
            <span>Assistant is typing</span>
        `;
        messagesContainer.appendChild(typingElement);
        scrollToBottom();
    }
    
    function removeTypingIndicator() {
        const typingElement = document.getElementById('typingIndicator');
        if (typingElement) {
            typingElement.remove();
        }
    }
    
    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    function hideEmptyState() {
        if (emptyState) {
            emptyState.style.display = 'none';
        }
    }
    
    function showEmptyState() {
        if (emptyState) {
            emptyState.style.display = 'flex';
        }
    }
    
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    function formatMessage(message) {
        // Convert URLs to links
        const urlRegex = /(https?:\/\/[^\s]+)/g;
        let formattedMessage = escapeHtml(message).replace(urlRegex, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
        
        // Convert line breaks to <br>
        formattedMessage = formattedMessage.replace(/\n/g, '<br>');
        
        return formattedMessage;
    }
});

// Handle URL clicks for web results
document.addEventListener('click', function(e) {
    const target = e.target.closest('.web-result-title a');
    if (target) {
        e.preventDefault();
        
        // Get URL from data attribute
        let url = target.getAttribute('data-original-url');
        
        // Clean the URL by removing spaces and encoding issues
        url = url.trim()
                .replace(/\s+/g, '')
                .replace(/&amp;/g, '&')
                .replace(/&lt;/g, '<')
                .replace(/&gt;/g, '>')
                .replace(/&quot;/g, '"')
                .replace(/&#039;/g, "'")
                .replace(/%20+/g, '');
        
        // Ensure URL has proper protocol
        if (!url.startsWith('http://') && !url.startsWith('https://')) {
            url = 'https://' + url.replace(/^[\/\\]+/, '');
        }
        
        // Open the URL in a new tab
        window.open(url, '_blank', 'noopener,noreferrer');
    }
});