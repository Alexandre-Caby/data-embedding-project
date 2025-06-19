document.addEventListener('DOMContentLoaded', function() {
    const messagesContainer = document.getElementById('messagesContainer');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const clearButton = document.getElementById('clearChat');
    const emptyState = document.getElementById('emptyState');
    
    let isWaitingForResponse = false;
    let selectedService = null;
    
    // Check if elements exist
    if (!messagesContainer || !messageInput || !sendButton) {
        console.error('Required DOM elements not found');
        return;
    }
    
    // Auto-resize the textarea
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        
        // Enable send button if there's text
        updateSendButton();
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
    if (clearButton) {
        clearButton.addEventListener('click', function() {
            fetch('/reset_conversation', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                messagesContainer.innerHTML = '';
                showEmptyState();
            })
            .catch(error => {
                console.error('Error resetting conversation:', error);
            });
        });
    }
    
    // Service button event listeners
    const serviceButtons = {
        'imageGenBtn': 'image',
        'webSearchBtn': 'web_search',
        'systemInfoBtn': 'system'
    };
    
    Object.keys(serviceButtons).forEach(buttonId => {
        const button = document.getElementById(buttonId);
        if (button) {
            button.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                toggleService(serviceButtons[buttonId]);
            });
        }
    });
    
    function updateSendButton() {
        const hasText = messageInput.value.trim().length > 0;
        sendButton.disabled = !hasText && !selectedService;
    }
    
    function toggleService(service) {
        const buttonMap = {
            'image': 'imageGenBtn',
            'web_search': 'webSearchBtn', 
            'system': 'systemInfoBtn'
        };
        
        // If same service clicked, deselect
        if (selectedService === service) {
            selectedService = null;
            document.getElementById(buttonMap[service]).classList.remove('selected');
            updatePlaceholder();
            updateSendButton();
            return;
        }
        
        // Clear all selections
        Object.values(buttonMap).forEach(btnId => {
            const btn = document.getElementById(btnId);
            if (btn) btn.classList.remove('selected');
        });
        
        // Select new service
        selectedService = service;
        const targetButton = document.getElementById(buttonMap[service]);
        if (targetButton) {
            targetButton.classList.add('selected');
        }
        
        updatePlaceholder();
        updateSendButton();
        messageInput.focus();
    }
    
    function updatePlaceholder() {
        const placeholders = {
            'image': 'Describe the image you want to generate...',
            'web_search': 'Enter your web search query...',
            'system': 'Ask for system information...',
            null: 'Type your question here'
        };
        
        messageInput.placeholder = placeholders[selectedService] || placeholders[null];
    }
    
    function sendMessage() {
        const message = messageInput.value.trim();
        
        // Store the message for potential enhancement later
        messageInput.lastQuery = message;
        
        if (!message && !selectedService) {
            return;
        }
        
        if (isWaitingForResponse) {
            return;
        }
        
        hideEmptyState();
        
        let processedMessage = message;
        
        // Process message based on selected service
        if (selectedService) {
            switch(selectedService) {
                case 'image':
                    processedMessage = message || 'Generate a creative image';
                    if (!processedMessage.toLowerCase().includes('generate')) {
                        processedMessage = `Generate image: ${processedMessage}`;
                    }
                    break;
                    
                case 'web_search':
                    processedMessage = message || 'latest news';
                    if (!processedMessage.toLowerCase().includes('search')) {
                        processedMessage = `Search web for: ${processedMessage}`;
                    }
                    break;
                    
                case 'system':
                    processedMessage = message || 'system info';
                    break;
            }
            
            // Clear service selection
            const buttonMap = {
                'image': 'imageGenBtn',
                'web_search': 'webSearchBtn',
                'system': 'systemInfoBtn'
            };
            
            Object.values(buttonMap).forEach(btnId => {
                const btn = document.getElementById(btnId);
                if (btn) btn.classList.remove('selected');
            });
            
            selectedService = null;
            updatePlaceholder();
        }
        
        // Add user message
        addUserMessage(processedMessage);
        
        // Clear input
        messageInput.value = '';
        messageInput.style.height = 'auto';
        updateSendButton();
        
        // Show typing indicator
        addTypingIndicator();
        
        // Set waiting state
        isWaitingForResponse = true;
        sendButton.disabled = true;
        
        // Send to backend
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: processedMessage })
        })
        .then(response => response.json())
        .then(data => {
            removeTypingIndicator();
            addAssistantMessage(
                data.context || "No response received", 
                data.retrieved_chunks || [], 
                data.images || [], 
                data.web_results || []
            );
        })
        .catch(error => {
            console.error('Error:', error);
            removeTypingIndicator();
            addAssistantMessage("Sorry, I encountered an error processing your request.", [], [], []);
        })
        .finally(() => {
            isWaitingForResponse = false;
            updateSendButton();
            messageInput.focus();
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
        
        let content = `<div class="message-content">${formatMessage(message)}</div>`;

        // Add sources if available and relevant (not for default responses)
        if (chunks && chunks.length > 0 && !message.includes('suffisamment d\'informations')) {
            const uniqueSources = [...new Set(chunks.map(chunk => 
                chunk.metadata?.source_document || chunk.source || ''
            ).filter(source => source && source !== 'unknown'))];
            
            if (uniqueSources.length > 0) {
                content += '<div class="message-sources">';
                content += '<strong>Sources:</strong><br>';
                uniqueSources.forEach(source => {
                    content += `<span class="source-item">${escapeHtml(source)}</span>`;
                });
                content += '</div>';
            }
        }
        
        // Add images if available
        if (images && images.length > 0) {
            images.forEach(image => {
                content += `
                    <div class="image-result">
                        <img src="${image.url}" alt="${escapeHtml(image.prompt || 'Generated image')}" />
                    </div>
                `;
            });
        }
        
        // Add web results if available
        if (webResults && webResults.length > 0) {
            content += `
                <div class="web-results">
                    <div class="web-results-header">
                        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="11" cy="11" r="8"></circle>
                            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                        </svg>
                        <span>Résultats de recherche (${webResults.length})</span>
                    </div>
            `;
            
            webResults.forEach(result => {
                let url = result.url?.trim() || '#';
                const title = result.title || 'Sans titre';
                const snippet = result.snippet || 'Aucune description disponible';
                
                // Clean and fix the URL
                url = cleanUrl(url);
                
                content += `
                    <div class="web-result-item">
                        <div class="web-result-title">
                            <a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer" data-clean-url="${escapeHtml(url)}">
                                ${escapeHtml(title)}
                            </a>
                        </div>
                        <div class="web-result-url">${escapeHtml(displayUrl(url))}</div>
                        <div class="web-result-snippet">${escapeHtml(snippet)}</div>
                    </div>
                `;
            });
            
            content += '</div>';
        }
        
        messageElement.innerHTML = content;
        messagesContainer.appendChild(messageElement);
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
            <span>Assistant is typing...</span>
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
        if (typeof unsafe !== 'string') return '';
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    function formatMessage(message) {
        if (typeof message !== 'string') return '';
        
        // Configure marked options for better rendering
        marked.setOptions({
            breaks: true,           // Convert line breaks to <br>
            gfm: true,             // GitHub flavored markdown
            headerIds: false,      // Don't add IDs to headers
            mangle: false,         // Don't mangle autolinks
            sanitize: false,       // Allow HTML (be careful with user input)
            smartLists: true,      // Use smarter list behavior
            smartypants: false     // Don't use smart quotes
        });
        
        try {
            let formatted = marked.parse(message);
            
            // Clean up any extra whitespace
            formatted = formatted.replace(/\n\s*\n/g, '\n');
            
            // Convert standalone URLs to links (that aren't already in <a> tags)
            const urlRegex = /(?<!href=["'])(https?:\/\/[^\s<>"']+)(?![^<]*<\/a>)/g;
            formatted = formatted.replace(urlRegex, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
            
            return formatted;
        } catch (error) {
            console.error('Error formatting message:', error);
            // Fallback to basic formatting
            return escapeHtml(message).replace(/\n/g, '<br>');
        }
    }
    
    function cleanUrl(url) {
        if (!url || url === '#') return '#';
        
        // Remove HTML entities and extra spaces
        let cleanedUrl = url
            .replace(/&amp;/g, '&')
            .replace(/&lt;/g, '<')
            .replace(/&gt;/g, '>')
            .replace(/&quot;/g, '"')
            .replace(/&#039;/g, "'")
            .replace(/%20+/g, '') // Remove encoded spaces
            .replace(/\s+/g, '') // Remove any remaining spaces
            .trim();
        
        // If URL doesn't start with protocol, add https://
        if (!cleanedUrl.startsWith('http://') && !cleanedUrl.startsWith('https://')) {
            // Remove any leading slashes or weird characters
            cleanedUrl = cleanedUrl.replace(/^[\/\\]+/, '');
            cleanedUrl = 'https://' + cleanedUrl;
        }
        
        // Additional cleaning for malformed URLs
        cleanedUrl = cleanedUrl.replace(/https?:\/\/\s+/, 'https://');
        
        return cleanedUrl;
    }
    
    function displayUrl(url) {
        // For display purposes, show a cleaner version
        return url.replace(/^https?:\/\//, '').replace(/\/$/, '');
    }

    // Initialize
    updateSendButton();
});

// Enhanced URL click handler
document.addEventListener('click', function(e) {
    const target = e.target.closest('.web-result-title a');
    if (target) {
        e.preventDefault();
        
        // Get the cleaned URL
        let url = target.getAttribute('data-clean-url') || target.getAttribute('href');
        
        // Additional cleaning just in case
        url = url.trim()
            .replace(/^https?:\/\/\s+/, 'https://')
            .replace(/%20+/g, '')
            .replace(/\s+/g, '');
        
        // Validate URL format
        try {
            new URL(url);
            console.log('Opening cleaned URL:', url);
            window.open(url, '_blank', 'noopener,noreferrer');
        } catch (error) {
            console.error('Invalid URL:', url, error);
            // Try to construct a valid URL
            const fallbackUrl = url.replace(/^[^a-zA-Z]*/, 'https://');
            try {
                new URL(fallbackUrl);
                console.log('Opening fallback URL:', fallbackUrl);
                window.open(fallbackUrl, '_blank', 'noopener,noreferrer');
            } catch (fallbackError) {
                console.error('Could not open URL:', url);
                alert('Sorry, this URL appears to be malformed and cannot be opened.');
            }
        }
    }
});