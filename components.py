# components.py - Enhanced with Rating and Bookmarking Features
import streamlit.components.v1 as components
import html
import uuid

def render_response_box(response_text, response_id):
    """Original render response box for backward compatibility"""
    escaped_text = html.escape(response_text)
    
    components.html(f"""
    <div class="response-box">
      <div class="response-header">
        💬 <strong>Response</strong>
        <button class="copy-button" onclick="copyToClipboard('{response_id}')">📋</button>
      </div>
      <div class="response-content" id="{response_id}">
        {escaped_text}
      </div>
    </div>

    <div id="toast-{response_id}" class="custom-toast">Copied to clipboard ✅</div>

    <script>
    function copyToClipboard(id) {{
        try {{
            const element = document.getElementById(id);
            if (!element) {{
                console.error('Element not found:', id);
                return;
            }}
            
            const text = element.innerText || element.textContent;
            
            if (navigator.clipboard && navigator.clipboard.writeText) {{
                navigator.clipboard.writeText(text).then(() => {{
                    showToast(id);
                }}).catch(err => {{
                    console.error("Clipboard API failed:", err);
                    fallbackCopy(text, id);
                }});
            }} else {{
                fallbackCopy(text, id);
            }}
        }} catch (error) {{
            console.error("Copy failed:", error);
        }}
    }}
    
    function fallbackCopy(text, id) {{
        try {{
            const textArea = document.createElement("textarea");
            textArea.value = text;
            textArea.style.position = "fixed";
            textArea.style.left = "-999999px";
            textArea.style.top = "-999999px";
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            
            const successful = document.execCommand('copy');
            document.body.removeChild(textArea);
            
            if (successful) {{
                showToast(id);
            }} else {{
                console.error("Fallback copy failed");
            }}
        }} catch (error) {{
            console.error("Fallback copy error:", error);
        }}
    }}
    
    function showToast(id) {{
        const toast = document.getElementById("toast-" + id);
        if (toast) {{
            toast.classList.add("show");
            setTimeout(() => {{
                toast.classList.remove("show");
            }}, 3000);
        }}
    }}
    </script>

    <style>
    .response-box {{
        background-color: #f9f9f9;
        border-radius: 12px;
        padding: 16px;
        margin-top: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        font-family: 'Inter', sans-serif;
        border: 1px solid #e0e0e0;
    }}
    
    .response-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 18px;
        margin-bottom: 12px;
        color: #333;
    }}
    
    .copy-button {{
        background: #e0e0e0;
        border: none;
        border-radius: 6px;
        padding: 6px 12px;
        cursor: pointer;
        font-size: 16px;
        transition: background-color 0.2s ease;
    }}
    
    .copy-button:hover {{
        background-color: #d0d0d0;
    }}
    
    .response-content {{
        max-height: 240px;
        overflow-y: auto;
        white-space: pre-wrap;
        background-color: #fff;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #eee;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        color: #333;
        word-break: break-word;
    }}
    
    .custom-toast {{
        visibility: hidden;
        min-width: 220px;
        margin-left: -110px;
        background-color: #4CAF50;
        color: #fff;
        text-align: center;
        border-radius: 8px;
        padding: 12px;
        position: fixed;
        z-index: 9999;
        left: 50%;
        bottom: 50px;
        font-size: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        opacity: 0;
        transform: translateY(30px);
    }}
    
    .custom-toast.show {{
        visibility: visible;
        opacity: 1;
        transform: translateY(0);
    }}
    </style>
    """, height=320)

def render_enhanced_response_box(response_text, message_id, session_id, is_bookmarked=False, rating=None, show_actions=True):
    """Enhanced response box with rating and bookmarking features"""
    escaped_text = html.escape(response_text)
    
    # Rating button states
    thumbs_up_class = "rating-active" if rating == 1 else ""
    thumbs_down_class = "rating-active" if rating == -1 else ""
    
    # Bookmark button state
    bookmark_icon = "🔖" if is_bookmarked else "📑"
    bookmark_class = "bookmark-active" if is_bookmarked else ""
    
    action_buttons = ""
    if show_actions:
        action_buttons = f"""
        <div class="action-buttons">
            <button class="action-btn rating-btn {thumbs_up_class}" 
                    onclick="rateMessage('{session_id}', '{message_id}', 1)" 
                    title="Rate positively">👍</button>
            <button class="action-btn rating-btn {thumbs_down_class}" 
                    onclick="rateMessage('{session_id}', '{message_id}', -1)" 
                    title="Rate negatively">👎</button>
            <button class="action-btn bookmark-btn {bookmark_class}" 
                    onclick="bookmarkMessage('{session_id}', '{message_id}', {str(not is_bookmarked).lower()})" 
                    title="{'Remove bookmark' if is_bookmarked else 'Bookmark response'}">{bookmark_icon}</button>
            <button class="action-btn copy-button" 
                    onclick="copyToClipboard('{message_id}')" 
                    title="Copy to clipboard">📋</button>
        </div>
        """
    
    components.html(f"""
    <div class="enhanced-response-box">
      <div class="response-header">
        💬 <strong>AI Response</strong>
        {action_buttons}
      </div>
      <div class="response-content" id="{message_id}">
        {escaped_text}
      </div>
    </div>

    <div id="toast-{message_id}" class="custom-toast">Action completed ✅</div>

    <script>
    function copyToClipboard(id) {{
        try {{
            const element = document.getElementById(id);
            if (!element) return;
            
            const text = element.innerText || element.textContent;
            
            if (navigator.clipboard && navigator.clipboard.writeText) {{
                navigator.clipboard.writeText(text).then(() => {{
                    showToast(id, 'Copied to clipboard ✅');
                }}).catch(err => {{
                    fallbackCopy(text, id);
                }});
            }} else {{
                fallbackCopy(text, id);
            }}
        }} catch (error) {{
            console.error("Copy failed:", error);
        }}
    }}
    
    function fallbackCopy(text, id) {{
        try {{
            const textArea = document.createElement("textarea");
            textArea.value = text;
            textArea.style.position = "fixed";
            textArea.style.left = "-999999px";
            textArea.style.top = "-999999px";
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            
            const successful = document.execCommand('copy');
            document.body.removeChild(textArea);
            
            if (successful) {{
                showToast(id, 'Copied to clipboard ✅');
            }}
        }} catch (error) {{
            console.error("Fallback copy error:", error);
        }}
    }}
    
    function rateMessage(sessionId, messageId, rating) {{
        // Send rating to Streamlit backend
        const data = {{
            action: 'rate_message',
            session_id: sessionId,
            message_id: messageId,
            rating: rating
        }};
        
        // Use Streamlit's component communication
        window.parent.postMessage({{
            type: 'streamlit:componentValue',
            value: data
        }}, '*');
        
        // Update UI immediately
        const buttons = document.querySelectorAll(`[onclick*="${{messageId}}"]`);
        buttons.forEach(btn => {{
            if (btn.textContent.includes('👍')) {{
                btn.classList.toggle('rating-active', rating === 1);
            }} else if (btn.textContent.includes('👎')) {{
                btn.classList.toggle('rating-active', rating === -1);
            }}
        }});
        
        showToast(messageId, rating === 1 ? 'Rated positively 👍' : 'Rated negatively 👎');
    }}
    
    function bookmarkMessage(sessionId, messageId, isBookmarked) {{
        // Send bookmark action to Streamlit backend
        const data = {{
            action: 'bookmark_message',
            session_id: sessionId,
            message_id: messageId,
            is_bookmarked: isBookmarked
        }};
        
        window.parent.postMessage({{
            type: 'streamlit:componentValue',
            value: data
        }}, '*');
        
        // Update UI immediately
        const bookmarkBtn = document.querySelector(`[onclick*="bookmarkMessage('${{sessionId}}', '${{messageId}}'"]`);
        if (bookmarkBtn) {{
            bookmarkBtn.textContent = isBookmarked ? '🔖' : '📑';
            bookmarkBtn.classList.toggle('bookmark-active', isBookmarked);
            bookmarkBtn.setAttribute('onclick', `bookmarkMessage('${{sessionId}}', '${{messageId}}', ${{!isBookmarked}})`);
            bookmarkBtn.title = isBookmarked ? 'Remove bookmark' : 'Bookmark response';
        }}
        
        showToast(messageId, isBookmarked ? 'Response bookmarked 🔖' : 'Bookmark removed 📑');
    }}
    
    function showToast(id, message) {{
        const toast = document.getElementById("toast-" + id);
        if (toast) {{
            toast.textContent = message;
            toast.classList.add("show");
            setTimeout(() => {{
                toast.classList.remove("show");
            }}, 3000);
        }}
    }}
    </script>

    <style>
    .enhanced-response-box {{
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f2ff 100%);
        border-radius: 16px;
        padding: 20px;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.1);
        border: 1px solid #e8eaff;
        font-family: 'Inter', sans-serif;
        position: relative;
    }}
    
    .response-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        color: #2d3748;
        font-weight: 600;
    }}
    
    .action-buttons {{
        display: flex;
        gap: 8px;
        align-items: center;
    }}
    
    .action-btn {{
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 8px 12px;
        cursor: pointer;
        font-size: 16px;
        transition: all 0.2s ease;
        backdrop-filter: blur(10px);
        min-width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    .action-btn:hover {{
        background: rgba(255, 255, 255, 0.95);
        border-color: #cbd5e0;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }}
    
    .rating-btn.rating-active {{
        background: #667eea;
        color: white;
        border-color: #667eea;
    }}
    
    .bookmark-btn.bookmark-active {{
        background: #f6ad55;
        color: white;
        border-color: #f6ad55;
    }}
    
    .copy-button:hover {{
        background: #e2e8f0;
    }}
    
    .response-content {{
        background: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e8eaff;
        max-height: 300px;
        overflow-y: auto;
        white-space: pre-wrap;
        font-family: 'Segoe UI', system-ui, sans-serif;
        line-height: 1.7;
        color: #2d3748;
        word-break: break-word;
        backdrop-filter: blur(5px);
    }}
    
    .response-content::-webkit-scrollbar {{
        width: 8px;
    }}
    
    .response-content::-webkit-scrollbar-track {{
        background: #f7fafc;
        border-radius: 4px;
    }}
    
    .response-content::-webkit-scrollbar-thumb {{
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 4px;
    }}
    
    .response-content::-webkit-scrollbar-thumb:hover {{
        background: linear-gradient(135deg, #5a6fd8, #6b46a3);
    }}
    
    .custom-toast {{
        visibility: hidden;
        min-width: 250px;
        margin-left: -125px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: #fff;
        text-align: center;
        border-radius: 12px;
        padding: 16px 20px;
        position: fixed;
        z-index: 9999;
        left: 50%;
        bottom: 50px;
        font-size: 16px;
        font-weight: 500;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        opacity: 0;
        transform: translateY(30px) scale(0.9);
        backdrop-filter: blur(10px);
    }}
    
    .custom-toast.show {{
        visibility: visible;
        opacity: 1;
        transform: translateY(0) scale(1);
    }}
    
    /* Responsive design */
    @media (max-width: 768px) {{
        .enhanced-response-box {{
            padding: 16px;
            margin: 12px 0;
        }}
        
        .action-buttons {{
            gap: 6px;
        }}
        
        .action-btn {{
            padding: 6px 8px;
            font-size: 14px;
            min-width: 36px;
            height: 36px;
        }}
        
        .response-content {{
            padding: 16px;
            max-height: 250px;
        }}
    }}
    
    /* Animation for new responses */
    @keyframes slideInUp {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .enhanced-response-box {{
        animation: slideInUp 0.4s ease-out;
    }}
    </style>
    """, height=400)

def render_typing_animation(text, response_id):
    """Render typing animation for the response"""
    escaped_text = html.escape(text)
    
    return f'''
    <div class="enhanced-response-box typing-animation">
      <div class="response-header">
        💬 <strong>AI Response</strong>
        <span class="typing-indicator">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </span>
      </div>
      <div class="response-content">
        {escaped_text}<span class="cursor">▌</span>
      </div>
    </div>
    <style>
    .typing-indicator {{
        display: flex;
        align-items: center;
        gap: 4px;
    }}
    
    .typing-dot {{
        width: 6px;
        height: 6px;
        background: #667eea;
        border-radius: 50%;
        animation: typing-bounce 1.4s infinite ease-in-out;
    }}
    
    .typing-dot:nth-child(1) {{ animation-delay: -0.32s; }}
    .typing-dot:nth-child(2) {{ animation-delay: -0.16s; }}
    .typing-dot:nth-child(3) {{ animation-delay: 0s; }}
    
    @keyframes typing-bounce {{
        0%, 80%, 100% {{
            transform: scale(0.8);
            opacity: 0.5;
        }}
        40% {{
            transform: scale(1);
            opacity: 1;
        }}
    }}
    
    .cursor {{
        animation: blink 1s infinite;
        color: #667eea;
        font-weight: bold;
    }}
    
    @keyframes blink {{
        0%, 50% {{ opacity: 1; }}
        51%, 100% {{ opacity: 0; }}
    }}
    </style>
    '''
