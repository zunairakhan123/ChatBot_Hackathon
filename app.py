# app.py - Complete Enhanced ICodeGuru Chatbot
import os
import json
import uuid
import time
import base64
import datetime
from typing import List, Optional, Dict, Any
import streamlit as st
import streamlit.components.v1 as components
import nest_asyncio
from dataclasses import dataclass, asdict
from pathlib import Path

# LangChain imports (your teammate's backend)
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import JSONLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Enhanced components
from components import render_response_box, render_enhanced_response_box
from user_manager import UserManager, UserProfile
from chat_manager import ChatManager, ChatSession

# Apply asyncio patch for Streamlit compatibility
nest_asyncio.apply()

# ========== Configuration ==========
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY environment variable is not set!")
    st.stop()

GROQ_MODEL = "llama3-8b-8192"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./chroma_db"
DOCS_DIR = "./docs"
USER_DATA_DIR = "./user_data"
CHAT_DATA_DIR = "./chat_data"

# Ensure directories exist
for directory in [USER_DATA_DIR, CHAT_DATA_DIR, DOCS_DIR]:
    Path(directory).mkdir(exist_ok=True)

# ========== Page Configuration ==========
st.set_page_config(
    page_title="ICodeGuru AI Assistant", 
    page_icon="🤖", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load CSS with error handling
try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("style.css file not found. Using default styling.")

# ========== Initialize Managers ==========
@st.cache_resource
def get_user_manager():
    return UserManager(USER_DATA_DIR)

@st.cache_resource
def get_chat_manager():
    return ChatManager(CHAT_DATA_DIR)

user_manager = get_user_manager()
chat_manager = get_chat_manager()

# ========== Logo Function ==========
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return f"data:image/jpeg;base64,{base64.b64encode(img_file.read()).decode()}"
    except FileNotFoundError:
        return "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGNpcmNsZSBjeD0iMzAiIGN5PSIzMCIgcj0iMzAiIGZpbGw9IiM2NjdlZWEiLz4KPHR5cGUgPSJ0ZXh0Ij5JQzwvdGV4dD4KPC9zdmc+"

# ========== User Authentication ==========
def render_user_auth():
    """Render user authentication interface"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    if not st.session_state.user_id:
        st.sidebar.markdown("### 👤 User Profile")
        
        auth_option = st.sidebar.radio("Choose option:", ["Login", "Create New Profile"])
        
        if auth_option == "Create New Profile":
            with st.sidebar.form("create_profile"):
                username = st.text_input("Username", placeholder="Enter username")
                display_name = st.text_input("Display Name", placeholder="Your display name")
                expertise_level = st.selectbox("Programming Experience", 
                    ["Beginner", "Intermediate", "Advanced", "Expert"])
                preferred_languages = st.multiselect("Preferred Languages", 
                    ["Python", "JavaScript", "Java", "C++", "C#", "Go", "Rust", "PHP", "Ruby"])
                learning_goals = st.text_area("Learning Goals", 
                    placeholder="What do you want to learn?")
                
                if st.form_submit_button("Create Profile"):
                    if username and display_name:
                        try:
                            profile = UserProfile(
                                user_id=str(uuid.uuid4()),
                                username=username,
                                display_name=display_name,
                                expertise_level=expertise_level,
                                preferred_languages=preferred_languages,
                                learning_goals=learning_goals
                            )
                            user_manager.create_user(profile)
                            st.session_state.user_id = profile.user_id
                            st.session_state.current_user = profile
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error creating profile: {str(e)}")
                    else:
                        st.error("Username and Display Name are required!")
        
        else:  # Login
            existing_users = user_manager.get_all_usernames()
            if existing_users:
                selected_username = st.sidebar.selectbox("Select Username", existing_users)
                
                if st.sidebar.button("Login"):
                    profile = user_manager.get_user_by_username(selected_username)
                    if profile:
                        st.session_state.user_id = profile.user_id
                        st.session_state.current_user = profile
                        st.rerun()
            else:
                st.sidebar.info("No existing profiles. Create a new one!")
    
    else:
        # User is logged in
        user = st.session_state.get('current_user')
        if user:
            st.sidebar.markdown(f"### 👋 Welcome, {user.display_name}!")
            st.sidebar.markdown(f"**Level:** {user.expertise_level}")
            
            if st.sidebar.button("Logout"):
                st.session_state.user_id = None
                st.session_state.current_user = None
                if 'current_session_id' in st.session_state:
                    del st.session_state.current_session_id
                st.rerun()

# ========== Enhanced LangChain RAG System ==========
class EnhancedLangChainRAGSystem:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.retrieval_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.setup_components()
    
    def setup_components(self):
        """Setup all LangChain components."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0.1,
            max_tokens=1024
        )
        
        self.load_vectorstore()
        self.setup_retrieval_chain()
    
    def load_vectorstore(self):
        """Load existing vectorstore or create empty one."""
        try:
            self.vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=self.embeddings,
                collection_name="icodeguru_knowledge"
            )
        except Exception as e:
            self.vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=self.embeddings,
                collection_name="icodeguru_knowledge"
            )
    
    def setup_retrieval_chain(self):
        """Setup the conversational retrieval chain with personalization."""
        def get_personalized_prompt():
            user = st.session_state.get('current_user')
            if user:
                user_context = f"""
                User Profile Context:
                - Name: {user.display_name}
                - Experience Level: {user.expertise_level}
                - Preferred Languages: {', '.join(user.preferred_languages) if user.preferred_languages else 'None specified'}
                - Learning Goals: {user.learning_goals or 'None specified'}
                
                Please tailor your response to match the user's experience level and preferences.
                """
            else:
                user_context = "User profile not available. Provide general guidance."
            
            return f"""You are an expert assistant for iCodeGuru, a programming education platform. 
            {user_context}
            
            Use the following context to answer the user's question comprehensively and accurately.
            Always provide relevant video links, website links, or resources when available in the context.
            Refer strictly to the provided context. If the answer isn't found in the context, explicitly say: "The provided knowledge base doesn't contain this information."
            
            Context: {{context}}
            Chat History: {{chat_history}}
            Human: {{question}}"""

        PROMPT = PromptTemplate(
        template=get_personalized_prompt(),
        input_variables=["context", "chat_history", "question"]
    )
    
        try:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            self.retrieval_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": PROMPT},
                return_source_documents=True,
                verbose=False
            )
            
        except Exception as e:
            self.retrieval_chain = None
    
    def load_and_process_documents(self) -> List[Document]:
        """Load and process JSON documents from the docs directory."""
        documents = []
        
        if not os.path.exists(DOCS_DIR):
            return documents
        
        json_files = [f for f in os.listdir(DOCS_DIR) if f.endswith('.json')]
        
        if not json_files:
            return documents
        
        for filename in json_files:
            file_path = os.path.join(DOCS_DIR, filename)
            try:
                loader = JSONLoader(
                    file_path=file_path,
                    jq_schema='.[]',
                    text_content=False
                )
                file_docs = loader.load()
                
                for doc in file_docs:
                    doc.metadata['source_file'] = filename
                    doc.metadata['file_path'] = file_path
                
                documents.extend(file_docs)
                
            except Exception as e:
                continue
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        return chunks
    
    def clear_knowledge_base(self):
        """Clear the existing knowledge base."""
        try:
            if self.vectorstore:
                self.vectorstore.delete_collection()
                self.vectorstore = Chroma(
                    persist_directory=CHROMA_PERSIST_DIR,
                    embedding_function=self.embeddings,
                    collection_name="icodeguru_knowledge"
                )
        except Exception as e:
            pass
    
    def ingest_documents(self):
        """Complete document ingestion pipeline."""
        documents = self.load_and_process_documents()
        
        if not documents:
            return False
        
        chunks = self.split_documents(documents)
        
        if not chunks:
            return False
        
        try:
            self.clear_knowledge_base()
            self.vectorstore.add_documents(chunks)
            self.vectorstore.persist()
            self.setup_retrieval_chain()
            return True
            
        except Exception as e:
            return False
                
    def get_answer(self, question: str) -> dict:
        """Get answer for a user question."""
        if not self.retrieval_chain:
            return {
                "answer": "⚠️ Knowledge base is initializing. Please try again in a moment.",
                "source_documents": []
            }
        
        try:
            doc_count = 0
            try:
                doc_count = self.vectorstore._collection.count()
            except:
                try:
                    test_results = self.vectorstore.similarity_search("test", k=1)
                    doc_count = len(test_results) if test_results else 0
                except:
                    doc_count = 0
            
            if doc_count == 0:
                return {
                    "answer": "I'm ready to help! However, I don't have any specific documents loaded in my knowledge base right now. I can still answer general programming questions based on my training. Feel free to ask anything!",
                    "source_documents": []
                }
            
            response = self.retrieval_chain({"question": question})
            return response
            
        except Exception as e:
            return {
                "answer": f"I apologize, but I encountered an issue processing your question. Could you please try rephrasing it?",
                "source_documents": []
            }
    
    def reset_conversation(self):
        """Reset the conversation memory."""
        self.memory.clear()

# Initialize the RAG system
@st.cache_resource
def get_rag_system():
    """Cache the RAG system to avoid reinitialization."""
    return EnhancedLangChainRAGSystem()

# ========== Session Management ==========
def initialize_chat_session():
    """Initialize or load chat session"""
    if 'current_session_id' not in st.session_state:
        user_id = st.session_state.get('user_id')
        if user_id:
            session_id = chat_manager.create_session(user_id)
            st.session_state.current_session_id = session_id
            st.session_state.messages = []
        else:
            st.session_state.messages = []
    else:
        # Load existing session messages
        session = chat_manager.get_session(st.session_state.current_session_id)
        if session:
            st.session_state.messages = []
            for msg in session.messages:
                st.session_state.messages.append({
                    "role": msg.role,
                    "content": msg.content,
                    "message_id": msg.message_id,
                    "rating": msg.rating,
                    "is_bookmarked": msg.is_bookmarked,
                    "source_documents": msg.source_documents
                })

# ========== Chat History Management ==========
def render_chat_history_sidebar():
    """Render chat history in sidebar"""
    if st.session_state.get('user_id'):
        user_sessions = chat_manager.get_user_sessions(st.session_state.user_id)
        
        if user_sessions:
            st.sidebar.markdown("### 💬 Chat History")
            
            for session in user_sessions[:10]:  # Show last 10 sessions
                session_title = session.title[:30] + "..." if len(session.title) > 30 else session.title
                
                col1, col2 = st.sidebar.columns([3, 1])
                
                with col1:
                    if st.button(session_title, key=f"session_{session.session_id}"):
                        st.session_state.current_session_id = session.session_id
                        initialize_chat_session()
                        st.rerun()
                
                with col2:
                    if st.button("🗑️", key=f"delete_{session.session_id}", help="Delete session"):
                        chat_manager.delete_session(session.session_id)
                        if st.session_state.get('current_session_id') == session.session_id:
                            del st.session_state.current_session_id
                        st.rerun()

# ========== Enhanced Sidebar Features ==========
def render_enhanced_sidebar():
    """Render enhanced sidebar with all features"""
    global GROQ_MODEL
    # User Authentication
    render_user_auth()
    
    if st.session_state.get('user_id'):
        # Chat History
        render_chat_history_sidebar()
        
        st.sidebar.markdown("---")
        
        # New Chat Button
        if st.sidebar.button("🆕 New Chat", type="primary"):
            user_id = st.session_state.user_id
            session_id = chat_manager.create_session(user_id)
            st.session_state.current_session_id = session_id
            st.session_state.messages = []
            get_rag_system().reset_conversation()
            st.rerun()
        
        # Model Selection
        st.sidebar.markdown("### 🧠 AI Settings")
        model_options = ["llama3-8b-8192", "llama3-70b-8192"]
        selected_model = st.sidebar.selectbox("Choose LLM Model", model_options, index=0)
        
        if selected_model != GROQ_MODEL:
            GROQ_MODEL = selected_model
            get_rag_system().llm.model_name = selected_model
        
        # Knowledge Base Management
        st.sidebar.markdown("### 📚 Knowledge Base")
        if st.sidebar.button("🔄 Refresh Knowledge Base"):
            with st.spinner("Refreshing knowledge base..."):
                success = get_rag_system().ingest_documents()
                if success:
                    st.sidebar.success("✅ Knowledge base refreshed!")
                else:
                    st.sidebar.warning("⚠️ No documents found to load")
        
        # Export Chat History
        st.sidebar.markdown("### 📤 Export")
        if st.sidebar.button("📄 Export Chat History"):
            if st.session_state.get('current_session_id'):
                export_data = chat_manager.export_chat_history(
                    st.session_state.user_id, 
                    st.session_state.current_session_id
                )
                if export_data:
                    st.sidebar.download_button(
                        label="⬇️ Download JSON",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"chat_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        # User Statistics
        st.sidebar.markdown("### 📊 Your Stats")
        user_stats = user_manager.get_user_stats(st.session_state.user_id)
        chat_stats = chat_manager.get_chat_statistics(st.session_state.user_id)
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Total Chats", chat_stats.get('total_sessions', 0))
        with col2:
            st.metric("Messages", chat_stats.get('total_messages', 0))
        
        st.sidebar.metric("Bookmarks", chat_stats.get('bookmarked_messages', 0))
        
        # Bookmarked Messages
        bookmarked = chat_manager.get_bookmarked_messages(st.session_state.user_id)
        if bookmarked:
            st.sidebar.markdown("### 🔖 Bookmarked Responses")
            for bookmark in bookmarked[:5]:  # Show 5 most recent
                message_preview = bookmark['message']['content'][:50] + "..."
                if st.sidebar.button(message_preview, key=f"bookmark_{bookmark['message']['message_id']}"):
                    # Show full bookmarked message
                    st.sidebar.write(bookmark['message']['content'])

# ========== Message Rating Handler ==========
def handle_component_value():
    """Handle component interactions (ratings, bookmarks)"""
    if 'component_value' in st.session_state and st.session_state.component_value:
        data = st.session_state.component_value
        
        if data.get('action') == 'rate_message':
            chat_manager.rate_message(
                data['session_id'], 
                data['message_id'], 
                data['rating']
            )
        
        elif data.get('action') == 'bookmark_message':
            chat_manager.bookmark_message(
                data['session_id'], 
                data['message_id'], 
                data['is_bookmarked']
            )
        
        # Clear the component value
        st.session_state.component_value = None

# ========== Main App Logic ==========
def main():
    """Main application logic"""
    
    # Handle component interactions
    handle_component_value()
    
    # Display logo and header
    image_data_url = get_base64_image("10001.jpeg")
    st.markdown(f"""
    <div class="custom-header">
        <h1><img src="{image_data_url}" class="chatbot-logo" alt="Bot" /> ICodeGuru AI Assistant</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Render enhanced sidebar
    render_enhanced_sidebar()
    
    # Initialize RAG system
    rag_system = get_rag_system()
    
    # Check if user is logged in
    if not st.session_state.get('user_id'):
        st.info("👈 Please login or create a profile to start chatting!")
        return
    
    # Initialize chat session
    initialize_chat_session()
    
    # Generate response function
    def generate_response(user_query):
        """Generate AI response using LangChain system"""
        if not user_query or not user_query.strip():
            return "Please provide a valid question."
        
        try:
            response = rag_system.get_answer(user_query)
            answer = response.get("answer", "I apologize, but I couldn't generate a response. Please try again.")
            
            source_docs = response.get("source_documents", [])
            if source_docs:
                sources_text = "\n\n📚 **Sources:**\n"
                for i, doc in enumerate(source_docs[:2], 1):
                    source_file = doc.metadata.get('source_file', 'Unknown')
                    content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    sources_text += f"{i}. {source_file}: {content_preview}\n"
                
                answer += sources_text
            
            return answer, [doc.metadata.get('source_file', '') for doc in source_docs]
            
        except Exception as e:
            return "I apologize, but I encountered an issue processing your question. Could you please try again.", []
    
    # Display chat messages
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                message_id = msg.get("message_id", f"msg-{i}")
                session_id = st.session_state.get("current_session_id", "")
                
                render_enhanced_response_box(
                    msg["content"], 
                    message_id, 
                    session_id,
                    is_bookmarked=msg.get("is_bookmarked", False),
                    rating=msg.get("rating"),
                    show_actions=True
                )
            else:
                st.markdown(msg["content"])
    
    # Chat input
    prompt = st.chat_input("Type your message...")
    
    if prompt:
        # Add user message to session
        user_message_id = chat_manager.add_message(
            st.session_state.current_session_id, 
            "user", 
            prompt
        )
        
        # Add to session state
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "message_id": user_message_id
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                full_response, source_docs = generate_response(prompt)
            
            # Add assistant message to session
            assistant_message_id = chat_manager.add_message(
                st.session_state.current_session_id, 
                "assistant", 
                full_response,
                source_docs
            )
            
            # Display response with enhanced box
            render_enhanced_response_box(
                full_response, 
                assistant_message_id, 
                st.session_state.current_session_id,
                is_bookmarked=False,
                rating=None,
                show_actions=True
            )
            
            # Add to session state
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "message_id": assistant_message_id,
                "rating": None,
                "is_bookmarked": False,
                "source_documents": source_docs
            })
        
        # Update user chat count
        user_manager.increment_chat_count(st.session_state.user_id)

if __name__ == "__main__":
    main()
