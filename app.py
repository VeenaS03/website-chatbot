"""
Streamlit web application for the website chatbot.
Provides user interface for indexing websites and asking questions.
"""

import os
import logging
from typing import Optional

import streamlit as st

from crawler import create_crawler
from text_processing import create_text_processor
from embeddings import create_embedding_manager
from qa_chain import create_qa_chain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Page configuration
st.set_page_config(
    page_title="Website Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        padding: 20px 0;
    }
    .status-box {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    </style>
    """, unsafe_allow_html=True)


class ChatbotApp:
    """Main application class for the website chatbot."""
    
    def __init__(self):
        """Initialize the chatbot application."""
        self.crawler = create_crawler()
        self.text_processor = create_text_processor(chunk_size=500, chunk_overlap=100)
        self.embedding_manager = create_embedding_manager()
        self.qa_chain: Optional[object] = None
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'indexed_url' not in st.session_state:
            st.session_state.indexed_url = None
        if 'indexed_page_title' not in st.session_state:
            st.session_state.indexed_page_title = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'qa_chain_initialized' not in st.session_state:
            st.session_state.qa_chain_initialized = False
    
    def index_website(self, url: str) -> bool:
        """
        Index a website for Q&A.
        
        Args:
            url: Website URL to index
            
        Returns:
            True if indexing was successful
        """
        try:
            with st.spinner("Crawling website..."):
                # Step 1: Crawl website
                success, content, title = self.crawler.crawl(url)
                
                if not success:
                    st.error(f"❌ {content}")
                    return False
                
                st.success(f"✓ Website crawled successfully. Found {len(content)} characters.")
            
            with st.spinner("Processing text..."):
                # Step 2: Process text
                chunks = self.text_processor.chunk_text(content)
                
                if not chunks:
                    st.error("❌ No text content could be extracted.")
                    return False
                
                st.success(f"✓ Created {len(chunks)} text chunks.")
            
            with st.spinner("Generating embeddings and indexing..."):
                # Step 3: Create embeddings and index
                self.embedding_manager.index_chunks(chunks, url, title)
                self.embedding_manager.save_to_disk()
                
                st.success("✓ Embeddings generated and indexed successfully.")
            
            # Step 4: Initialize QA chain
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
                return False
            
            self.qa_chain = create_qa_chain(self.embedding_manager, api_key)
            st.session_state.qa_chain_initialized = True
            st.session_state.indexed_url = url
            st.session_state.indexed_page_title = title
            
            # Reset chat history when indexing new website
            st.session_state.chat_history = []
            
            return True
            
        except Exception as e:
            logger.error(f"Error indexing website: {str(e)}")
            st.error(f"❌ An error occurred: {str(e)}")
            return False
    
    def handle_query(self, query: str) -> str:
        """
        Process a user query and generate an answer.
        
        Args:
            query: User question
            
        Returns:
            Generated answer
        """
        if not self.qa_chain or not st.session_state.qa_chain_initialized:
            return "❌ Please index a website first."
        
        try:
            answer, sources = self.qa_chain.answer(query)
            
            # Store in chat history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': query
            })
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': answer,
                'sources': sources
            })
            
            return answer
        except Exception as e:
            logger.error(f"Error handling query: {str(e)}")
            return f"❌ Error: {str(e)}"
    
    def render_sidebar(self):
        """Render the sidebar with controls and information."""
        st.sidebar.markdown("### 🔧 Controls")
        
        # Website indexing section
        st.sidebar.markdown("#### Index Website")
        url_input = st.sidebar.text_input(
            "Enter website URL",
            placeholder="https://example.com",
            help="Enter the URL of the website you want to chat with"
        )
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("📥 Index", use_container_width=True):
                if not url_input.strip():
                    st.sidebar.error("Please enter a URL")
                else:
                    self.index_website(url_input.strip())
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                self.embedding_manager.clear_index()
                st.session_state.indexed_url = None
                st.session_state.indexed_page_title = None
                st.session_state.chat_history = []
                st.session_state.qa_chain_initialized = False
                st.sidebar.success("Index cleared!")
        
        # Status section
        st.sidebar.markdown("#### 📊 Status")
        stats = self.embedding_manager.get_index_stats()
        
        if stats['initialized']:
            st.sidebar.info(
                f"**Indexed Website:** {st.session_state.indexed_page_title}\n\n"
                f"**Vectors:** {stats['vector_count']}\n\n"
                f"**Dimension:** {stats['dimension']}"
            )
        else:
            st.sidebar.warning("No website indexed yet")
        
        # Settings section
        st.sidebar.markdown("#### ⚙️ Settings")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            chunk_size = st.sidebar.slider(
                "Chunk Size",
                min_value=100,
                max_value=2000,
                value=500,
                step=100,
                help="Text chunk size in characters"
            )
        
        with col2:
            chunk_overlap = st.sidebar.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=100,
                step=50,
                help="Overlap between consecutive chunks"
            )
        
        # Update if changed
        if chunk_size != self.text_processor.chunk_size or chunk_overlap != self.text_processor.chunk_overlap:
            self.text_processor.update_chunk_config(chunk_size, chunk_overlap)
            st.sidebar.success("Settings updated!")
        
        # About section
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### ℹ️ About")
        st.sidebar.markdown(
            "**Website Chatbot** powered by:\n"
            "- OpenAI GPT-3.5-turbo\n"
            "- SentenceTransformers\n"
            "- FAISS vector database\n"
            "- Streamlit"
        )
    
    def render_main(self):
        """Render the main chat interface."""
        # Header
        st.markdown("<h1 class='main-header'>🤖 Website Chatbot</h1>", unsafe_allow_html=True)
        st.markdown("Ask questions about any indexed website. Get answers based strictly on its content.")
        
        # Check if website is indexed
        if not st.session_state.qa_chain_initialized:
            st.info("👈 **Get Started:** Enter a website URL in the sidebar and click 'Index' to begin.")
            return
        
        # Display indexed website info
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"✓ Currently chatting about: **{st.session_state.indexed_page_title}**")
        with col2:
            if st.button("🔄 New Website"):
                self.embedding_manager.clear_index()
                st.session_state.indexed_url = None
                st.session_state.indexed_page_title = None
                st.session_state.chat_history = []
                st.session_state.qa_chain_initialized = False
                st.rerun()
        
        st.divider()
        
        # Chat interface
        st.markdown("### 💬 Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.chat_message("user").write(message['content'])
            else:
                with st.chat_message("assistant"):
                    st.write(message['content'])
                    
                    # Show sources if available
                    if 'sources' in message and message['sources']:
                        with st.expander("📚 Sources"):
                            for idx, source in enumerate(message['sources'], 1):
                                st.markdown(
                                    f"**Source {idx}** (Similarity: {source.get('similarity_score', 0):.2f})\n\n"
                                    f"_{source.get('page_title', 'Unknown')}_\n\n"
                                    f"{source['content'][:200]}..."
                                )
        
        # User input
        st.divider()
        user_input = st.chat_input(
            "Ask a question about the website...",
            key="user_input"
        )
        
        if user_input:
            # Display user message
            st.chat_message("user").write(user_input)
            
            # Get answer
            with st.spinner("🤔 Thinking..."):
                answer = self.handle_query(user_input)
            
            # Display assistant response
            st.chat_message("assistant").write(answer)
            
            # Show sources
            if st.session_state.chat_history:
                last_message = st.session_state.chat_history[-1]
                if 'sources' in last_message and last_message['sources']:
                    with st.expander("📚 Sources"):
                        for idx, source in enumerate(last_message['sources'], 1):
                            st.markdown(
                                f"**Source {idx}** (Similarity: {source.get('similarity_score', 0):.2f})\n\n"
                                f"_{source.get('page_title', 'Unknown')}_\n\n"
                                f"{source['content'][:200]}..."
                            )


def main():
    """Main entry point for the Streamlit app."""
    app = ChatbotApp()
    
    # Render sidebar
    app.render_sidebar()
    
    # Render main content
    app.render_main()


if __name__ == "__main__":
    main()