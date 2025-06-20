document.addEventListener('DOMContentLoaded', function() {
  const messagesContainer = document.getElementById('messagesContainer');
  const messageInput      = document.getElementById('messageInput');
  const sendButton        = document.getElementById('sendButton');
  const clearButton       = document.getElementById('clearChat');
  const emptyState        = document.getElementById('emptyState');

  let isWaitingForResponse = false;
  let selectedService      = null;

  // Vérification des éléments
  if (!messagesContainer || !messageInput || !sendButton) {
    console.error('Required DOM elements not found');
    return;
  }

  // Redimensionnement automatique du textarea
  messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    updateSendButton();
  });

  // Envoi par Enter (sans Shift)
  messageInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Envoi par clic
  sendButton.addEventListener('click', sendMessage);

  // Reset du chat
  if (clearButton) {
    clearButton.addEventListener('click', function() {
      fetch('/reset_conversation', { method: 'POST' })
        .then(res => res.json())
        .then(() => {
          messagesContainer.innerHTML = '';
          showEmptyState();
        })
        .catch(err => console.error('Error resetting conversation:', err));
    });
  }

  // Boutons de service
  const serviceButtons = {
    'imageGenBtn':    'image',
    'webSearchBtn':   'web_search',
    'systemInfoBtn':  'system'
  };
  Object.entries(serviceButtons).forEach(([btnId, svc]) => {
    const btn = document.getElementById(btnId);
    if (btn) {
      btn.addEventListener('click', e => {
        e.preventDefault(); e.stopPropagation();
        toggleService(svc);
      });
    }
  });

  // Activation / désactivation du bouton envoyer
  function updateSendButton() {
    const hasText = messageInput.value.trim().length > 0;
    sendButton.disabled = !hasText && !selectedService;
  }

  function toggleService(service) {
    const btnMap = { image: 'imageGenBtn', web_search: 'webSearchBtn', system: 'systemInfoBtn' };
    if (selectedService === service) {
      document.getElementById(btnMap[service]).classList.remove('selected');
      selectedService = null;
      updatePlaceholder();
      updateSendButton();
      return;
    }
    Object.values(btnMap).forEach(id => document.getElementById(id)?.classList.remove('selected'));
    selectedService = service;
    document.getElementById(btnMap[service])?.classList.add('selected');
    updatePlaceholder();
    updateSendButton();
    messageInput.focus();
  }

  function updatePlaceholder() {
    const placeholders = {
      image:     'Describe the image you want to generate…',
      web_search:'Enter your web search query…',
      system:    'Ask for system information…',
      null:      'Type your question here'
    };
    messageInput.placeholder = placeholders[selectedService] || placeholders.null;
  }

  // Envoi du message
  function sendMessage() {
    const text = messageInput.value.trim();
    if ((!text && !selectedService) || isWaitingForResponse) return;

    // Store the message for potential enhancement later
    messageInput.lastQuery = text;

    hideEmptyState();
    let msg = text;

    if (selectedService) {
      if (selectedService === 'image') {
        msg = text || 'Generate a creative image';
        if (!msg.toLowerCase().includes('generate')) msg = `Generate image: ${msg}`;
      }
      if (selectedService === 'web_search') {
        msg = text || 'latest news';
        if (!msg.toLowerCase().includes('search')) msg = `Search web for: ${msg}`;
      }
      if (selectedService === 'system') {
        msg = text || 'system info';
      }
      // Reset service buttons
      ['imageGenBtn','webSearchBtn','systemInfoBtn'].forEach(id => document.getElementById(id)?.classList.remove('selected'));
      selectedService = null;
      updatePlaceholder();
    }

    addUserMessage(msg);
    messageInput.value = '';
    messageInput.style.height = 'auto';
    updateSendButton();
    addTypingIndicator();

    isWaitingForResponse = true;
    sendButton.disabled    = true;

    fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg })
    })
    .then(r => r.json())
    .then(data => {
      removeTypingIndicator();
      addAssistantMessage(
        data.context || "No response received",
        data.retrieved_chunks || [],
        data.images || [],
        data.web_results || []
      );
    })
    .catch(err => {
      console.error('Error:', err);
      removeTypingIndicator();
      addAssistantMessage("Sorry, I encountered an error processing your request.", [], [], []);
    })
    .finally(() => {
      isWaitingForResponse = false;
      updateSendButton();
      messageInput.focus();
    });
  }

  // Affichage des messages utilisateur / assistant
  function addUserMessage(text) {
    const el = document.createElement('div');
    el.className = 'message user-message';
    el.innerHTML = `<div class="message-content">${escapeHtml(text)}</div>`;
    messagesContainer.appendChild(el);
    scrollToBottom();
  }

  function addAssistantMessage(message, chunks, images, webResults) {
    const el = document.createElement('div');
    el.className = 'message assistant-message';
    let html = `<div class="message-content">${formatMessage(message)}</div>`;

    // Images générées
    if (images.length) {
      images.forEach(img => {
        let url = img.url;
        if (!url.startsWith('http') && !url.startsWith('/output/')) {
          url = `/output/images/${url}`;
        }
        html += `
          <div class="image-result">
            <img src="${url}"
                 alt="${escapeHtml(img.prompt||'Generated image')}"
                 onerror="this.onerror=null;this.src='data:image/svg+xml;base64,PHN2ZyB4...';this.style.opacity='0.5';" />
            ${img.prompt ? `<div class="image-caption">"${escapeHtml(img.prompt)}"</div>` : ''}
          </div>
        `;
      });
    }

    // Sources - Amélioration avec condition sur les réponses par défaut
    if (chunks.length && !message.includes('suffisamment d\'informations')) {
      const uniq = [...new Set(chunks.map(c=>c.metadata?.source_document||c.source||'').filter(s=>s&&s!=='unknown'))];
      if (uniq.length) {
        html += '<div class="message-sources"><strong>Sources:</strong><br>';
        uniq.forEach(src => html += `<span class="source-item">${escapeHtml(src)}</span>`);
        html += '</div>';
      }
    }

    // Résultats web
    if (webResults.length) {
      html += `
        <div class="web-results">
          <div class="web-results-header">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
            <span>Résultats de recherche (${webResults.length})</span>
          </div>
      `;
      webResults.forEach(r => {
        const url     = cleanUrl(r.url?.trim()||'#');
        const title   = r.title   || 'Sans titre';
        const snippet = r.snippet || 'Aucune description disponible';
        html += `
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
      html += '</div>';
    }

    el.innerHTML = html;
    messagesContainer.appendChild(el);
    scrollToBottom();
  }

  // Indicateur de saisie
  function addTypingIndicator() {
    const ind = document.createElement('div');
    ind.className = 'typing-indicator'; ind.id = 'typingIndicator';
    ind.innerHTML = '<div class="typing-bubble"></div>'.repeat(3) + '<span>Assistant is typing…</span>';
    messagesContainer.appendChild(ind);
    scrollToBottom();
  }
  function removeTypingIndicator() {
    document.getElementById('typingIndicator')?.remove();
  }

  // Défilement, état vide
  function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }
  function hideEmptyState() {
    emptyState && (emptyState.style.display = 'none');
  }
  function showEmptyState() {
    emptyState && (emptyState.style.display = 'flex');
  }

  // Échappement et formatage du texte
  function escapeHtml(u) {
    if (typeof u!=='string') return '';
    return u.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
            .replace(/"/g,"&quot;").replace(/'/g,"&#039;");
  }

  function formatMessage(message) {
    if (typeof message !== 'string') return '';
    // 1) Échappement HTML de base
    let text = escapeHtml(message);

    // 2) Bloc de code triple backticks (```code```)
    text = text.replace(/```([\s\S]+?)```/g, '<pre><code>$1</code></pre>');

    // 3) Block LaTeX $$...$$
    text = text.replace(/\$\$([\s\S]+?)\$\$/g, '<div class="latex-block">$1</div>');

    // 4) Blockquote en début de ligne
    text = text.replace(/^\s*>\s*(.+)$/gm, '<blockquote>$1</blockquote>');

    // 5) Listes non ordonnées et ordonnées multi‑lignes
    text = text
      // UL : lignes commençant par '- '
      .replace(/(?:^|\n)(-\s+.+)(?:\n|$)/g, match => {
        const items = match.trim().split('\n').map(l => l.replace(/^- /, '').trim());
        return '\n<ul>\n' + items.map(i => `<li>${i}</li>`).join('\n') + '\n</ul>\n';
      })
      // OL : lignes '1. ', '2. ', etc.
      .replace(/(?:^|\n)(\d+\.\s+.+)(?:\n|$)/g, match => {
        const items = match.trim().split('\n').map(l => l.replace(/^\d+\.\s+/, '').trim());
        return '\n<ol>\n' + items.map(i => `<li>${i}</li>`).join('\n') + '\n</ol>\n';
      });

    // 6) Inline Markdown links [text](url)
    text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

    // 7) Automatic URL linking (http(s)://...)
    text = text.replace(/(https?:\/\/[^\s<]+)/g,
      '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');

    // 8) Inline code `code`
    text = text.replace(/`([^`\n]+)`/g, '<code>$1</code>');

    // 9) **gras** et *italique* et ~~barré~~
    const inlineStyles = [
      { regex: /\*\*([^*]+)\*\*/g, tag: 'strong' },
      { regex: /\*([^*]+)\*/g,     tag: 'em'     },
      { regex: /~~([^~]+)~~/g,     tag: 'del'    },
    ];
    inlineStyles.forEach(({regex, tag}) => {
      text = text.replace(regex, `<${tag}>$1</${tag}>`);
    });

    // 10) Inline LaTeX $...$
    text = text.replace(/\$(.+?)\$/g, '<span class="latex">$1</span>');

    // 11) Emojis :emoji:
    text = text.replace(/:([a-zA-Z0-9_]+):/g, '<span class="emoji">:$1:</span>');

    // 12) Mentions @user et hashtags #tag
    text = text
      .replace(/@([a-zA-Z0-9_]+)/g, '<span class="mention">@$1</span>')
      .replace(/#([a-zA-Z0-9_]+)/g, '<span class="hashtag">#$1</span>');

    // 13) Dates (YYYY‑MM‑DD) et heures (HH:MM)
    text = text
      .replace(/(\d{4}-\d{2}-\d{2})/g, '<time datetime="$1">$1</time>')
      .replace(/(\b\d{1,2}:\d{2}\b)/g, '<time datetime="$1">$1</time>');

    // 14) Lignes vides en <br> pour conserver les sauts
    text = text.replace(/\n{2,}/g, '<br><br>');
    text = text.replace(/\n/g, '<br>');

    return text.trim();
  }

  // Nettoyage robuste des URL - Version améliorée du second fichier
  function cleanUrl(url) {
    if (!url || url === '') return '#';

    // Remove HTML entities and extra spaces
    let u = url
      .replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>')
      .replace(/&quot;/g,'"').replace(/&#039;/g,"'")
      .replace(/%20+/g,'').replace(/\s+/g,'').trim();

    // If URL doesn't start with protocol, add https://
    if (!u.startsWith('http://') && !u.startsWith('https://')) {
      u = u.replace(/^[\/\\]+/,'');
      u = 'https://' + u;
    }

    // Additional cleaning for malformed URLs
    u = u.replace(/https?:\/\/\s+/,'https://');

    return u;
  }

  function displayUrl(url) {
    try {
      const p = new URL(url);
      return p.hostname + p.pathname.replace(/\/$/, '');
    } catch {
      // For display purposes, show a cleaner version
      return url.replace(/^https?:\/\//, '').replace(/\/$/, '');
    }
  }

  /*** ÉCOUTEUR GLOBAL POUR LIENS WEB - Version améliorée ***/
  document.addEventListener('click', function(e) {
    const a = e.target.closest('.web-result-title a');
    if (!a) return;
    e.preventDefault();

    // Get the cleaned URL
    let url = a.dataset.cleanUrl || a.href;

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
  });

  // Initialisation
  updateSendButton();
  showEmptyState();
});