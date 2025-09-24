"""Telemetry service for logging and metrics."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from models.database import QueryRun, ChatSession, Document
from utils.helpers import generate_session_id


class TelemetryService:
    """Service for telemetry and logging."""

    def __init__(self, db: Session):
        self.db = db

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create new chat session."""
        if not session_id:
            session_id = generate_session_id()
        
        session = ChatSession(session_id=session_id)
        self.db.add(session)
        self.db.commit()
        
        return session_id

    def log_query_run(
        self,
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
        """Log query run with telemetry."""
        total_time = embedding_time + retrieval_time + llm_time
        
        query_run = QueryRun(
            session_id=session_id,
            question=question,
            answer=answer,
            retrieved_chunks=json.dumps(retrieved_chunks),
            similarity_scores=json.dumps(similarity_scores),
            embedding_time=embedding_time,
            retrieval_time=retrieval_time,
            llm_time=llm_time,
            total_time=total_time,
            sources_found=sources_found
        )
        
        self.db.add(query_run)
        self.db.commit()

    def update_document_status(
        self,
        doc_id: str,
        status: str,
        error_message: Optional[str] = None,
        total_pages: Optional[int] = None,
        total_chunks: Optional[int] = None
    ):
        """Update document processing status."""
        document = self.db.query(Document).filter_by(doc_id=doc_id).first()
        
        if document:
            document.status = status
            document.updated_at = datetime.utcnow()
            
            if error_message:
                document.error_message = error_message
            if total_pages:
                document.total_pages = total_pages
            if total_chunks:
                document.total_chunks = total_chunks
            
            self.db.commit()

    def get_document_status(self, doc_id: str) -> Optional[Dict]:
        """Get document status."""
        document = self.db.query(Document).filter_by(doc_id=doc_id).first()
        
        if document:
            return {
                "doc_id": document.doc_id,
                "filename": document.filename,
                "status": document.status,
                "error_message": document.error_message,
                "total_pages": document.total_pages,
                "total_chunks": document.total_chunks,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat()
            }
        
        return None

    def get_all_documents(self) -> List[Dict]:
        """Get all documents with status."""
        documents = self.db.query(Document).all()
        
        return [
            {
                "doc_id": doc.doc_id,
                "filename": doc.filename,
                "status": doc.status,
                "total_pages": doc.total_pages,
                "total_chunks": doc.total_chunks,
                "created_at": doc.created_at.isoformat()
            }
            for doc in documents
        ]

    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get query history for session."""
        runs = (
            self.db.query(QueryRun)
            .filter_by(session_id=session_id)
            .order_by(QueryRun.created_at)
            .all()
        )
        
        return [
            {
                "question": run.question,
                "answer": run.answer,
                "sources_found": run.sources_found,
                "created_at": run.created_at.isoformat()
            }
            for run in runs
        ]