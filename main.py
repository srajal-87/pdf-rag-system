"""FastAPI backend application."""

import asyncio
import io
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from models.database import Document, ChatSession, QueryRun, get_db, init_db, SessionLocal

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config.settings import settings
from models.database import Document, get_db, init_db
from services import (
    EmbeddingService,
    LLMService,
    PDFProcessor,
    TelemetryService,
    VectorStore,
)
from utils.helpers import generate_doc_id


# Initialize FastAPI app
app = FastAPI(title="PDF RAG System API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
pdf_processor = PDFProcessor()
embedding_service = EmbeddingService()
vector_store = VectorStore()
llm_service = LLMService()


# Request/Response models
class QueryRequest(BaseModel):
    """Query request model."""
    
    question: str
    session_id: str
    top_k: int = settings.top_k_default
    doc_filter: Optional[str] = None
    only_use_sources: bool = True


class QueryResponse(BaseModel):
    """Query response model."""
    
    answer: str
    sources: List[Dict]
    similarity_scores: List[float]
    telemetry: Dict


class DocumentStatus(BaseModel):
    """Document status model."""
    
    doc_id: str
    filename: str
    status: str
    error_message: Optional[str]
    total_pages: Optional[int]
    total_chunks: Optional[int]

async def process_pdf_background(
    doc_id: str,
    filename: str,
    content: bytes
):
    """Process PDF in background with its own database session."""
    print(f"[BACKEND] Starting background processing for {doc_id}")
    
    # Create a new database session for this background task
    db = SessionLocal()
    
    try:
        telemetry = TelemetryService(db)
        
        print(f"[BACKEND] Updating status to processing for {doc_id}")
        telemetry.update_document_status(doc_id, "processing")
        
        print(f"[BACKEND] Processing PDF for {doc_id}")
        chunks, total_pages, error = pdf_processor.process_pdf(
            content, doc_id, filename
        )
        print(f"[BACKEND] PDF processed: {len(chunks) if chunks else 0} chunks, error: {error}")
        
        if error and not chunks:
            print(f"[BACKEND] Setting status to failed for {doc_id}: {error}")
            telemetry.update_document_status(
                doc_id, "failed", error_message=error
            )
            return
        
        # Continue with embedding and storage
        print(f"[BACKEND] Creating embeddings for {len(chunks)} chunks")
        
        # Extract text from chunks for embedding
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        # Create embeddings in batch
        embeddings, embedding_time = embedding_service.create_embeddings_batch(chunk_texts)
        print(f"[BACKEND] Embeddings created in {embedding_time:.2f}s")
        
        # Store chunks and embeddings in vector store
        print(f"[BACKEND] Storing {len(chunks)} chunks in vector store")
        success = vector_store.add_documents(chunks, embeddings)
        
        if success:
            print(f"[BACKEND] Successfully stored chunks for {doc_id}")
            # Update status to indexed with full details
            telemetry.update_document_status(
                doc_id, 
                "indexed",
                total_pages=total_pages,
                total_chunks=len(chunks)
            )
            print(f"[BACKEND] Status updated to 'indexed' for {doc_id}")
        else:
            print(f"[BACKEND] Failed to store chunks in vector store for {doc_id}")
            telemetry.update_document_status(
                doc_id, 
                "failed", 
                error_message="Failed to store chunks in vector store"
            )
        
    except Exception as e:
        print(f"[BACKEND] Exception in background task for {doc_id}: {str(e)}")
        # Make sure we have a telemetry instance even if there was an earlier error
        try:
            if 'telemetry' not in locals():
                telemetry = TelemetryService(db)
            telemetry.update_document_status(
                doc_id, "failed", error_message=str(e)
            )
        except Exception as nested_e:
            print(f"[BACKEND] Failed to update status after error: {str(nested_e)}")
    
    finally:
        # Always close the database session
        db.close()
        print(f"[BACKEND] Database session closed for {doc_id}")

async def log_telemetry_background(
    session_id: str,
    question: str,
    answer: str,
    retrieved_chunks: List[Dict],
    similarity_scores: List[float],
    embedding_time: float,
    retrieval_time: float,
    llm_time: float,
    sources_found: bool
):
    """Log telemetry in background with its own DB session."""
    db = SessionLocal()
    try:
        telemetry = TelemetryService(db)
        telemetry.log_query_run(
            session_id,
            question,
            answer,
            retrieved_chunks,
            similarity_scores,
            embedding_time,
            retrieval_time,
            llm_time,
            sources_found
        )
    except Exception as e:
        print(f"[BACKEND] Failed to log telemetry: {str(e)}")
    finally:
        db.close()

@app.on_event("startup")
async def startup_event():
    """Initialize database and vector store on startup."""
    # Initialize database tables
    init_db()
    
    # Initialize vector store collection
    vector_dimension = embedding_service.get_embedding_dimension()
    vector_store.initialize_collection(vector_dimension)
    
    print(f"✓ Database initialized")
    print(f"✓ Vector store initialized (dimension: {vector_dimension})")
    print(f"✓ Using embedding model: {settings.embedding_model}")
    
    # Validate Gemini API key
    if not llm_service.validate_api_key():
        print("⚠ Warning: Gemini API key validation failed")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "PDF RAG System API",
        "version": "1.0.0",
        "status": "running"
    }


@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload PDF document for processing.
    Returns doc_id immediately and processes in background.
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Read file content
    content = await file.read()
    
    # Generate document ID
    doc_id = generate_doc_id(file.filename)
    
    # Create document record
    document = Document(
        doc_id=doc_id,
        filename=file.filename,
        status="processing"
    )
    db.add(document)
    db.commit()
    
    # Add background task for processing (removed db parameter)
    background_tasks.add_task(
        process_pdf_background,
        doc_id,
        file.filename,
        content
    )
    
    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "status": "processing",
        "message": "Document uploaded successfully. Processing in background."
    }


@app.get("/documents")
async def get_documents(db: Session = Depends(get_db)):
    """Get all documents with their current status."""
    telemetry = TelemetryService(db)
    documents = telemetry.get_all_documents()
    
    return {"documents": documents}


@app.get("/status/{doc_id}")
async def get_document_status(
    doc_id: str,
    db: Session = Depends(get_db)
):
    """Get status of a specific document."""
    telemetry = TelemetryService(db)
    status = telemetry.get_document_status(doc_id)
    
    if not status:
        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )
    
    return status


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Query documents using RAG pipeline with comprehensive debug logging.
    """
    print(f"\n[DEBUG] ========== QUERY DEBUG SESSION ==========")
    print(f"[DEBUG] Question: '{request.question}'")
    print(f"[DEBUG] Session ID: {request.session_id}")
    print(f"[DEBUG] Top K: {request.top_k}")
    print(f"[DEBUG] Doc filter: {request.doc_filter}")
    print(f"[DEBUG] Only use sources: {request.only_use_sources}")
    print(f"[DEBUG] Similarity threshold: {settings.similarity_threshold}")
    
    try:
        # STEP 1: Check vector store collection status
        print(f"\n[DEBUG] Step 1: Checking vector store collection...")
        try:
            collection_info = vector_store.client.get_collection(vector_store.collection_name)
            points_count = collection_info.points_count
            # Fix: Use correct attribute path for vector size
            vector_size = collection_info.config.params.vectors.size if hasattr(collection_info.config.params, 'vectors') else collection_info.config.params.vector.size
            print(f"[DEBUG] ✓ Collection exists: {vector_store.collection_name}")
            print(f"[DEBUG] ✓ Total points in collection: {points_count}")
            print(f"[DEBUG] ✓ Vector dimension: {vector_size}")
            
            if points_count == 0:
                print(f"[DEBUG] ⚠️  WARNING: Vector store is empty! No documents stored.")
                return QueryResponse(
                    answer="I don't know. No documents have been uploaded and indexed yet.",
                    sources=[],
                    similarity_scores=[],
                    telemetry={
                        "embedding_time": 0,
                        "retrieval_time": 0,
                        "llm_time": 0,
                        "total_time": 0,
                        "sources_found": False,
                        "chunks_retrieved": 0,
                        "debug_info": "Vector store is empty"
                    }
                )
        except Exception as e:
            print(f"[DEBUG] ❌ Error accessing collection: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Vector store error: {str(e)}")

        # STEP 2: Create embedding for question
        print(f"\n[DEBUG] Step 2: Creating query embedding...")
        query_embedding, embedding_time = embedding_service.create_embedding(request.question)
        print(f"[DEBUG] ✓ Query embedding created")
        print(f"[DEBUG] ✓ Embedding shape: {query_embedding.shape}")
        print(f"[DEBUG] ✓ Embedding time: {embedding_time:.4f}s")
        print(f"[DEBUG] ✓ Embedding preview: [{query_embedding[0]:.4f}, {query_embedding[1]:.4f}, ...]")
        
        # Validate embedding dimension
        expected_dim = embedding_service.get_embedding_dimension()
        if len(query_embedding) != expected_dim:
            print(f"[DEBUG] ❌ Embedding dimension mismatch! Got {len(query_embedding)}, expected {expected_dim}")
            raise HTTPException(status_code=500, detail="Embedding dimension mismatch")

        # STEP 3: Search vector store with debug info
        print(f"\n[DEBUG] Step 3: Searching vector store...")
        
        # First, let's do a raw search without filtering to see what we get
        print(f"[DEBUG] Performing raw search (no threshold filtering)...")
        try:
            # Build filter if document specified
            query_filter = None
            if request.doc_filter:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=request.doc_filter)
                        )
                    ]
                )
                print(f"[DEBUG] ✓ Document filter applied: {request.doc_filter}")
            
            # Raw search without threshold
            raw_results = vector_store.client.search(
                collection_name=vector_store.collection_name,
                query_vector=query_embedding.tolist(),
                limit=request.top_k,
                query_filter=query_filter,
                with_payload=True
            )
            
            print(f"[DEBUG] ✓ Raw search completed")
            print(f"[DEBUG] ✓ Raw results count: {len(raw_results)}")
            
            # Log all raw results with scores
            for i, result in enumerate(raw_results):
                print(f"[DEBUG] Result {i+1}:")
                print(f"[DEBUG]   - Score: {result.score:.6f}")
                print(f"[DEBUG]   - Doc ID: {result.payload.get('doc_id', 'N/A')}")
                print(f"[DEBUG]   - Filename: {result.payload.get('filename', 'N/A')}")
                print(f"[DEBUG]   - Page: {result.payload.get('page_number', 'N/A')}")
                print(f"[DEBUG]   - Text preview: {result.payload.get('text', '')[:100]}...")
            
            # Now apply threshold filtering
            print(f"\n[DEBUG] Applying similarity threshold: {settings.similarity_threshold}")
            filtered_results = []
            filtered_scores = []
            
            for result in raw_results:
                print(f"[DEBUG] Checking score {result.score:.6f} >= {settings.similarity_threshold}")
                if result.score >= settings.similarity_threshold:
                    filtered_results.append(result.payload)
                    filtered_scores.append(result.score)
                    print(f"[DEBUG]   ✓ PASSED threshold")
                else:
                    print(f"[DEBUG]   ❌ FAILED threshold")
            
            print(f"[DEBUG] ✓ Filtered results count: {len(filtered_results)}")
            
            # If no results pass threshold, let's use a relaxed threshold
            if not filtered_results and raw_results:
                relaxed_threshold = 0.1  # Very permissive
                print(f"[DEBUG] ⚠️  No results passed strict threshold. Trying relaxed threshold: {relaxed_threshold}")
                
                for result in raw_results:
                    if result.score >= relaxed_threshold:
                        filtered_results.append(result.payload)
                        filtered_scores.append(result.score)
                
                print(f"[DEBUG] ✓ Relaxed results count: {len(filtered_results)}")
            
        except Exception as e:
            print(f"[DEBUG] ❌ Search error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")

        # STEP 4: Generate answer with LLM
        print(f"\n[DEBUG] Step 4: Generating answer...")
        print(f"[DEBUG] Retrieved chunks for LLM: {len(filtered_results)}")
        
        if not filtered_results and request.only_use_sources:
            answer = (
                "I don't know. No relevant sources were found in the "
                "uploaded documents to answer your question."
            )
            llm_time = 0.0
            print(f"[DEBUG] ✓ No sources found, using default answer")
        else:
            print(f"[DEBUG] Generating answer with {len(filtered_results)} chunks...")
            answer, llm_time = llm_service.generate_answer(
                request.question,
                filtered_results,
                request.only_use_sources
            )
            print(f"[DEBUG] ✓ Answer generated in {llm_time:.4f}s")
            print(f"[DEBUG] ✓ Answer preview: {answer[:200]}...")

        # STEP 5: Log comprehensive telemetry
        sources_found = len(filtered_results) > 0
        retrieval_time = embedding_time  # Approximate, since we don't time just retrieval
        
        print(f"\n[DEBUG] Step 5: Logging telemetry...")
        print(f"[DEBUG] ✓ Sources found: {sources_found}")
        print(f"[DEBUG] ✓ Chunks retrieved: {len(filtered_results)}")
        print(f"[DEBUG] ✓ Similarity scores: {filtered_scores}")
        
        # Add background task for telemetry logging
        background_tasks.add_task(
            log_telemetry_background,
            request.session_id,
            request.question,
            answer,
            filtered_results,
            filtered_scores,
            embedding_time,
            retrieval_time,
            llm_time,
            sources_found
        )
        
        print(f"[DEBUG] ========== QUERY DEBUG COMPLETE ==========\n")
        
        return QueryResponse(
            answer=answer,
            sources=filtered_results,
            similarity_scores=filtered_scores,
            telemetry={
                "embedding_time": embedding_time,
                "retrieval_time": retrieval_time,
                "llm_time": llm_time,
                "total_time": embedding_time + retrieval_time + llm_time,
                "sources_found": sources_found,
                "chunks_retrieved": len(filtered_results),
                "debug_info": {
                    "total_vectors_in_store": points_count,
                    "raw_results_count": len(raw_results),
                    "filtered_results_count": len(filtered_results),
                    "similarity_threshold": settings.similarity_threshold,
                    "best_score": raw_results[0].score if raw_results else None
                }
            }
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"[DEBUG] ❌ Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )

@app.get("/sessions")
async def get_sessions(db: Session = Depends(get_db)):
    """Get all chat sessions."""
    sessions = db.query(ChatSession).order_by(
        ChatSession.created_at.desc()
    ).all()
    
    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "created_at": s.created_at.isoformat()
            }
            for s in sessions
        ]
    }


@app.post("/sessions/create")
async def create_session(db: Session = Depends(get_db)):
    """Create a new chat session."""
    telemetry = TelemetryService(db)
    session_id = telemetry.create_session()
    
    return {"session_id": session_id}


@app.get("/sessions/{session_id}/history")
async def get_session_history(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get query history for a session."""
    telemetry = TelemetryService(db)
    history = telemetry.get_session_history(session_id)
    
    return {"history": history}


@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """Delete a chat session and its history."""
    # Delete query runs
    db.query(QueryRun).filter_by(session_id=session_id).delete()
    
    # Delete session
    db.query(ChatSession).filter_by(session_id=session_id).delete()
    
    db.commit()
    
    return {"message": "Session deleted successfully"}


@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    db: Session = Depends(get_db)
):
    """Delete a document and its vectors."""
    # Delete from vector store
    success = vector_store.delete_document(doc_id)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to delete from vector store"
        )
    
    # Delete from database
    db.query(Document).filter_by(doc_id=doc_id).delete()
    db.commit()
    
    return {"message": "Document deleted successfully"}

@app.get("/debug/vector-store")
async def debug_vector_store():
    """Debug endpoint to inspect vector store contents."""
    try:
        collection_info = vector_store.client.get_collection(vector_store.collection_name)
        
        # Get some sample points
        scroll_result = vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            limit=10,
            with_payload=True,
            with_vectors=False
        )
        
        sample_points = []
        for point in scroll_result[0]:
            sample_points.append({
                "id": str(point.id),
                "payload": {
                    "doc_id": point.payload.get("doc_id"),
                    "filename": point.payload.get("filename"),
                    "page_number": point.payload.get("page_number"),
                    "text_preview": point.payload.get("text", "")[:100]
                }
            })
        
        return {
            "collection_name": vector_store.collection_name,
            "total_points": collection_info.points_count,
            "vector_dimension": collection_info.config.params.vector.size,
            "distance_metric": collection_info.config.params.vector.distance,
            "sample_points": sample_points
        }
        
    except Exception as e:
        return {"error": f"Failed to inspect vector store: {str(e)}"}

@app.get("/debug/test-similarity")
async def test_similarity(text1: str, text2: str):
    """Test similarity between two texts."""
    try:
        embedding1, _ = embedding_service.create_embedding(text1)
        embedding2, _ = embedding_service.create_embedding(text2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        return {
            "text1": text1,
            "text2": text2,
            "cosine_similarity": float(similarity),
            "embedding_dimension": len(embedding1)
        }
        
    except Exception as e:
        return {"error": f"Failed to test similarity: {str(e)}"}
