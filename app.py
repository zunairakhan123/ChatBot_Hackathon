import streamlit as st
import os
import json
from typing import List, Optional
import nest_asyncio

# LangChain imports
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

# Apply asyncio patch for Streamlit compatibility
nest_asyncio.apply()

# --- CONFIGURATION ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY environment variable is not set!")
    st.stop()

GROQ_MODEL = "llama3-8b-8192"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHROMA_PERSIST_DIR = "./chroma_db"
DOCS_DIR = "./docs"

class LangChainRAGSystem:
    def __init__(self):
        """Initialize the LangChain RAG system components."""
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
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize LLM
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL,
            temperature=0.1,
            max_tokens=1024
        )
        
        # Load or create vectorstore
        self.load_vectorstore()
        
        # Setup retrieval chain
        self.setup_retrieval_chain()
    
    def load_vectorstore(self):
        """Load existing vectorstore or create empty one."""
        try:
            self.vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=self.embeddings,
                collection_name="icodeguru_knowledge"
            )
            st.info("✅ Loaded existing knowledge base.")
        except Exception as e:
            st.warning(f"Creating new knowledge base: {e}")
            self.vectorstore = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=self.embeddings,
                collection_name="icodeguru_knowledge"
            )
    
    def setup_retrieval_chain(self):
        """Setup the conversational retrieval chain."""
        # Custom prompt template
        prompt_template = """You are an expert assistant for iCodeGuru, a programming education platform. 
        Use the following context to answer the user's question comprehensively and accurately.
        Always provide relevant video links, website links, or resources when available in the context.
        If you don't know the answer based on the context, say so clearly.
    
        Context: {context}
    
        Chat History: {chat_history}
    
        Human: {question}
    
        Assistant: I'll help you with that based on the iCodeGuru knowledge base.
    
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Always try to create retriever - let it handle empty collections gracefully
        try:
            # Create retriever
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
            )
            
            # Create conversational retrieval chain
            self.retrieval_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": PROMPT},
                return_source_documents=True,
                verbose=True
            )
            st.success("✅ Retrieval chain setup successfully!")
            
        except Exception as e:
            st.warning(f"⚠️ Retrieval chain setup issue: {str(e)}")
            self.retrieval_chain = None
    
    def load_and_process_documents(self) -> List[Document]:
        """Load and process JSON documents from the docs directory."""
        documents = []
        
        if not os.path.exists(DOCS_DIR):
            st.error(f"❌ Documents directory '{DOCS_DIR}' not found!")
            return documents
        
        # Get all JSON files
        json_files = [f for f in os.listdir(DOCS_DIR) if f.endswith('.json')]
        
        if not json_files:
            st.warning(f"⚠️ No JSON files found in '{DOCS_DIR}' directory!")
            return documents
        
        st.info(f"📂 Found {len(json_files)} JSON files to process...")
        
        for filename in json_files:
            file_path = os.path.join(DOCS_DIR, filename)
            try:
                # Use JSONLoader with proper schema
                loader = JSONLoader(
                    file_path=file_path,
                    jq_schema='.[]',
                    text_content=False
                )
                file_docs = loader.load()
                
                # Add source metadata
                for doc in file_docs:
                    doc.metadata['source_file'] = filename
                    doc.metadata['file_path'] = file_path
                
                documents.extend(file_docs)
                st.success(f"✅ Loaded {len(file_docs)} documents from {filename}")
                
            except Exception as e:
                st.error(f"❌ Error loading {filename}: {str(e)}")
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
        st.info(f"📄 Created {len(chunks)} document chunks")
        return chunks
    
    def clear_knowledge_base(self):
        """Clear the existing knowledge base."""
        try:
            if self.vectorstore:
                # Delete the collection
                self.vectorstore.delete_collection()
                st.success("🗑️ Cleared existing knowledge base")
                
                # Recreate empty vectorstore
                self.vectorstore = Chroma(
                    persist_directory=CHROMA_PERSIST_DIR,
                    embedding_function=self.embeddings,
                    collection_name="icodeguru_knowledge"
                )
        except Exception as e:
            st.error(f"❌ Error clearing knowledge base: {str(e)}")
    
    def ingest_documents(self):
        """Complete document ingestion pipeline."""
        with st.spinner("🔄 Loading documents..."):
            # Load documents
            documents = self.load_and_process_documents()
            
            if not documents:
                st.error("❌ No documents loaded. Please check your docs folder.")
                return False
        
        with st.spinner("✂️ Splitting documents into chunks..."):
            # Split documents
            chunks = self.split_documents(documents)
            
            if not chunks:
                st.error("❌ No document chunks created.")
                return False
        
        with st.spinner("🧠 Creating embeddings and storing in vector database..."):
            try:
                # Clear existing data
                self.clear_knowledge_base()
                
                # Add chunks to vectorstore
                self.vectorstore.add_documents(chunks)
                
                # Persist the vectorstore
                self.vectorstore.persist()
                
                st.success(f"✅ Successfully ingested {len(chunks)} document chunks!")
                
                # Force recreate retrieval chain with new data
                self.setup_retrieval_chain()
                
                # Verify the setup worked
                try:
                    doc_count = self.vectorstore._collection.count()
                    st.info(f"📊 Knowledge base now contains {doc_count} documents")
                except:
                    st.info("📊 Knowledge base updated successfully")
                
                return True
                
            except Exception as e:
                st.error(f"❌ Error during ingestion: {str(e)}")
                return False
                
    def get_answer(self, question: str) -> dict:
        """Get answer for a user question."""
        if not self.retrieval_chain:
            return {
                "answer": "⚠️ Knowledge base is empty. Please refresh the knowledge base first.",
                "source_documents": []
            }
        
        try:
            # Check if vectorstore has documents before querying
            doc_count = 0
            try:
                doc_count = self.vectorstore._collection.count()
            except:
                # If count fails, try a simple similarity search to test
                try:
                    test_results = self.vectorstore.similarity_search("test", k=1)
                    doc_count = len(test_results) if test_results else 0
                except:
                    doc_count = 0
            
            if doc_count == 0:
                return {
                    "answer": "⚠️ No documents found in knowledge base. Please refresh the knowledge base first.",
                    "source_documents": []
                }
            
            # Get response from the chain
            response = self.retrieval_chain({"question": question})
            return response
            
        except Exception as e:
            return {
                "answer": f"❌ Error getting answer: {str(e)}",
                "source_documents": []
            }
    
    def reset_conversation(self):
        """Reset the conversation memory and UI chat history."""
        self.memory.clear()
        # Also clear Streamlit session state messages
        if "messages" in st.session_state:
            st.session_state.messages = []
        st.success("🔄 Conversation history cleared!")

# Initialize the RAG system
@st.cache_resource
def get_rag_system():
    """Cache the RAG system to avoid reinitialization."""
    return LangChainRAGSystem()

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="EduBot for iCodeGuru",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🎓 EduBot for @icodeguru0")
    st.markdown("**Powered by LangChain** | Ask anything based on pre-loaded iCodeGuru knowledge.")
    
    # Initialize RAG system
    rag_system = get_rag_system()
    
    # Sidebar for admin functions
    with st.sidebar:
        st.header("⚙️ Admin Panel")
        
        if st.button("🔄 Refresh Knowledge Base", type="primary"):
            success = rag_system.ingest_documents()
            if success:
                st.balloons()
        
        if st.button("🗑️ Clear Conversation"):
            rag_system.reset_conversation()
            st.rerun()  # Force UI refresh
        
        st.markdown("---")
        st.subheader("📊 System Info")
        
        # Show vectorstore stats
        if rag_system.vectorstore:
            try:
                doc_count = rag_system.vectorstore._collection.count()
                st.metric("Documents in KB", doc_count)
            except:
                st.metric("Documents in KB", "N/A")
        
        st.markdown("---")
        st.caption("🧠 **ChromaDB** for vector storage")
        st.caption("⚡ **Groq LLM** for answers")
        st.caption("🔗 **LangChain** for orchestration")
    
    # Main chat interface
    st.markdown("---")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source}")
    
    # User input
    if prompt := st.chat_input("💬 Ask your question about iCodeGuru..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = rag_system.get_answer(prompt)
                answer = response.get("answer", "No answer available.")
                source_docs = response.get("source_documents", [])
                
                st.markdown(answer)
                
                # Show sources if available
                if source_docs:
                    sources = []
                    for doc in source_docs[:3]:  # Show top 3 sources
                        source = doc.metadata.get('source_file', 'Unknown source')
                        content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                        sources.append(f"{source}: {content_preview}")
                    
                    if sources:
                        with st.expander("📚 Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:** {source}")
                        
                        # Add to session state with sources
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
