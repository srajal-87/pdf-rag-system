"""SQLAlchemy database models."""

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    Boolean,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from config.settings import settings

# Database setup
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Document(Base):
    """Document model for tracking uploaded PDFs."""

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    doc_id = Column(String(255), unique=True, nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    status = Column(String(50), default="processing", index=True)
    error_message = Column(Text)
    total_pages = Column(Integer)
    total_chunks = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChatSession(Base):
    """Chat session model."""

    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class QueryRun(Base):
    """Query run model for telemetry."""

    __tablename__ = "query_runs"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text)
    retrieved_chunks = Column(JSONB)
    similarity_scores = Column(JSONB)
    embedding_time = Column(Float)
    retrieval_time = Column(Float)
    llm_time = Column(Float)
    total_time = Column(Float)
    sources_found = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)


def get_db():
    """Database dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)