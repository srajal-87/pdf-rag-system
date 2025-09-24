"""Streamlit frontend for PDF RAG System."""

import json
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests
import streamlit as st
from streamlit_extras.colored_header import colored_header

from config.settings import settings


# Configuration
API_BASE_URL = "http://localhost:8000"
POLLING_INTERVAL = settings.polling_interval


# Helper functions
def api_call(
    method: str,
    endpoint: str,
    data: Optional[Dict] = None,
    files: Optional[Dict] = None
) -> Dict:
    """Make API call to backend."""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, data=data)
            else:
                response = requests.post(url, json=data)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Please ensure the API is running.")
        return {}
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API Error: {e}")
        return {}
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return {}


def init_session_state():
    """Initialize session state variables."""
    if "session_id" not in st.session_state:
        # Create new session
        response = api_call("POST", "/sessions/create")
        if response:
            st.session_state.session_id = response["session_id"]
        else:
            st.session_state.session_id = "default"
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "polling_docs" not in st.session_state:
        st.session_state.polling_docs = set()
    
    if "last_poll_time" not in st.session_state:
        st.session_state.last_poll_time = 0
    
    if "documents" not in st.session_state:
        st.session_state.documents = []
    
    if "show_context" not in st.session_state:
        st.session_state.show_context = {}


def poll_document_status():
    current_time = time.time()
    print(f"[DEBUG] Polling check at {current_time}, last poll: {st.session_state.last_poll_time}")
    
    if (current_time - st.session_state.last_poll_time) >= POLLING_INTERVAL:
        print(f"[DEBUG] Polling {len(st.session_state.polling_docs)} documents")
        st.session_state.last_poll_time = current_time
        
        docs_to_remove = []
        for doc_id in st.session_state.polling_docs:
            print(f"[DEBUG] Checking status for doc_id: {doc_id}")
            status = api_call("GET", f"/status/{doc_id}")
            print(f"[DEBUG] Status response: {status}")
            
            if status and status["status"] in ["indexed", "failed"]:
                print(f"[DEBUG] Document {doc_id} finished processing: {status['status']}")
                docs_to_remove.append(doc_id)
                
                # Show notification
                if status["status"] == "indexed":
                    st.success(
                        f"✅ Document '{status['filename']}' indexed successfully! "
                        f"({status['total_pages']} pages, {status['total_chunks']} chunks)"
                    )
                else:
                    st.error(
                        f"❌ Failed to process '{status['filename']}': "
                        f"{status.get('error_message', 'Unknown error')}"
                    )
        
        # Remove completed documents from polling set
        for doc_id in docs_to_remove:
            st.session_state.polling_docs.discard(doc_id)
        
        # Rerun if still polling
        if st.session_state.polling_docs:
            time.sleep(0.1)  # Small delay to prevent too rapid reruns
            st.rerun()


def fetch_documents():
    """Fetch all documents from backend."""
    response = api_call("GET", "/documents")
    if response:
        st.session_state.documents = response.get("documents", [])


def render_sidebar():
    """Render sidebar with files panel and settings."""
    with st.sidebar:
        # Logo/Title
        st.title("📚 PDF RAG System")
        st.divider()
        
        # Upload Section
        st.subheader("📤 Upload PDF")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            key="pdf_uploader"
        )
        
        if uploaded_file is not None:
            if st.button("📥 Upload / Index", type="primary", use_container_width=True):
                with st.spinner("Uploading..."):
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    response = api_call("POST", "/upload", files=files)
                    
                    if response and "doc_id" in response:
                        st.success(f"✅ Upload successful! Processing...")
                        st.session_state.polling_docs.add(response["doc_id"])
                        time.sleep(1)
                        st.rerun()
        
        st.divider()
        
        # Files Panel
        st.subheader("📁 Uploaded Documents")
        
        if st.button("🔄 Refresh", use_container_width=True):
            fetch_documents()
            st.rerun()
        
        if st.session_state.documents:
            for doc in st.session_state.documents:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Document info
                        st.markdown(f"**📄 {doc['filename']}**")
                        
                        # Status badge
                        status = doc['status']
                        if status == "processing":
                            st.markdown("🔄 **Status:** Processing...")
                        elif status == "indexed":
                            st.markdown(
                                f"✅ **Status:** Indexed "
                                f"({doc.get('total_pages', 0)} pages, "
                                f"{doc.get('total_chunks', 0)} chunks)"
                            )
                        elif status == "failed":
                            st.markdown("❌ **Status:** Failed")
                    
                    with col2:
                        if st.button("🗑️", key=f"del_{doc['doc_id']}"):
                            response = api_call(
                                "DELETE",
                                f"/documents/{doc['doc_id']}"
                            )
                            if response:
                                st.success("Deleted!")
                                fetch_documents()
                                st.rerun()
                
                st.divider()
        else:
            st.info("No documents uploaded yet.")
        
        # Settings Drawer
        with st.expander("⚙️ Settings", expanded=False):
            st.subheader("Query Settings")
            
            # Top-k setting
            st.session_state.top_k = st.slider(
                "Number of chunks to retrieve (top_k)",
                min_value=1,
                max_value=10,
                value=settings.top_k_default,
                help="Number of most relevant chunks to retrieve"
            )
            
            # Document filter
            doc_options = ["All Documents"] + [
                f"{doc['filename']}" for doc in st.session_state.documents
                if doc['status'] == 'indexed'
            ]
            
            selected_doc = st.selectbox(
                "Document Filter",
                options=doc_options,
                help="Filter search to specific document"
            )
            
            st.session_state.doc_filter = None
            if selected_doc != "All Documents":
                # Find the doc_id for the selected document
                for doc in st.session_state.documents:
                    if doc['filename'] == selected_doc:
                        st.session_state.doc_filter = doc['doc_id']
                        break
            
            # Only answer if sources found toggle
            st.session_state.only_use_sources = st.checkbox(
                "Only answer if sources found",
                value=True,
                help="If checked, the system will only answer when relevant sources are found"
            )
            
            # Model selector (placeholder for future)
            st.selectbox(
                "LLM Model",
                options=["Gemini Pro"],
                disabled=True,
                help="Currently only Gemini Pro is supported"
            )
            
            st.divider()
            
            # Session management
            st.subheader("Session")
            st.text(f"ID: {st.session_state.session_id[:8]}...")
            
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.show_context = {}
                st.rerun()


def render_chat_interface():
    """Render main chat interface."""
    # Header
    colored_header(
        label="💬 Chat with your PDFs",
        description="Ask questions about your uploaded documents",
        color_name="blue-70"
    )
    
    # Check if any documents are indexed
    indexed_docs = [
        doc for doc in st.session_state.documents
        if doc['status'] == 'indexed'
    ]
    
    if not indexed_docs:
        st.info(
            "📤 Please upload and wait for at least one PDF to be indexed "
            "before asking questions."
        )
    
    # Chat messages container
    messages_container = st.container()
    
    with messages_container:
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show sources if available
                if "sources" in message and message["sources"]:
                    st.markdown("**📚 Sources:**")
                    for j, source in enumerate(message["sources"][:3]):
                        st.markdown(
                            f"- {source['filename']} - "
                            f"Page {source['page_number']}, "
                            f"Section {source['chunk_index'] + 1}"
                        )
                
                # Show context toggle
                if "sources" in message and message["sources"]:
                    context_key = f"context_{i}"
                    
                    if st.button(
                        f"{'Hide' if st.session_state.show_context.get(context_key, False) else 'Show'} Context",
                        key=f"toggle_{context_key}"
                    ):
                        st.session_state.show_context[context_key] = not st.session_state.show_context.get(
                            context_key,
                            False
                        )
                        st.rerun()
                    
                    # Show context if toggled
                    if st.session_state.show_context.get(context_key, False):
                        with st.expander("Retrieved Context", expanded=True):
                            for j, source in enumerate(message["sources"]):
                                st.markdown(
                                    f"**[{j+1}] {source['filename']} - "
                                    f"Page {source['page_number']}:**"
                                )
                                st.text(source['text'][:500] + "...")
                                if "similarity_scores" in message:
                                    st.caption(
                                        f"Similarity: {message['similarity_scores'][j]:.3f}"
                                    )
                                st.divider()
                
                # Show telemetry if available
                if "telemetry" in message:
                    with st.expander("📊 Performance Metrics"):
                        tel = message["telemetry"]
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Embedding Time",
                                f"{tel['embedding_time']:.3f}s"
                            )
                        with col2:
                            st.metric(
                                "Retrieval Time",
                                f"{tel['retrieval_time']:.3f}s"
                            )
                        with col3:
                            st.metric(
                                "LLM Time",
                                f"{tel['llm_time']:.3f}s"
                            )
                        with col4:
                            st.metric(
                                "Total Time",
                                f"{tel['total_time']:.3f}s"
                            )
    
    # Chat input
    if prompt := st.chat_input(
        "Ask a question about your documents...",
        disabled=not indexed_docs
    ):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from API
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                response = api_call(
                    "POST",
                    "/query",
                    data={
                        "question": prompt,
                        "session_id": st.session_state.session_id,
                        "top_k": st.session_state.get("top_k", settings.top_k_default),
                        "doc_filter": st.session_state.get("doc_filter"),
                        "only_use_sources": st.session_state.get("only_use_sources", True)
                    }
                )
                
                if response:
                    answer = response.get("answer", "Failed to generate answer.")
                    st.markdown(answer)
                    
                    # Add assistant message with metadata
                    assistant_message = {
                        "role": "assistant",
                        "content": answer,
                        "sources": response.get("sources", []),
                        "similarity_scores": response.get("similarity_scores", []),
                        "telemetry": response.get("telemetry", {})
                    }
                    st.session_state.messages.append(assistant_message)
                    
                    # Show sources
                    if response.get("sources"):
                        st.markdown("**📚 Sources:**")
                        for i, source in enumerate(response["sources"][:3]):
                            st.markdown(
                                f"- {source['filename']} - "
                                f"Page {source['page_number']}, "
                                f"Section {source['chunk_index'] + 1}"
                            )
                else:
                    error_msg = "Failed to get response from the backend."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
        
        st.rerun()


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="PDF RAG System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .stButton > button {
            width: 100%;
        }
        .stExpander {
            background-color: #f0f2f6;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Fetch initial documents
    if not st.session_state.documents:
        fetch_documents()
    
    # Poll for processing documents
    if st.session_state.polling_docs:
        poll_document_status()
    
    # Render sidebar
    render_sidebar()
    
    # Render chat interface
    render_chat_interface()


if __name__ == "__main__":
    main()