"""Utility functions."""

import hashlib
import time
import uuid
from typing import Any, Dict, List, Optional


def generate_doc_id(filename: str) -> str:
    """Generate unique document ID."""
    timestamp = str(time.time())
    unique_string = f"{filename}_{timestamp}_{uuid.uuid4()}"
    return hashlib.md5(unique_string.encode()).hexdigest()


def generate_session_id() -> str:
    """Generate unique session ID."""
    return str(uuid.uuid4())


def chunk_text(
    text: str,
    chunk_size: int = 400,
    chunk_overlap: int = 50
) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks


def format_citation(
    filename: str,
    page_number: int,
    chunk_index: int
) -> str:
    """Format citation for display."""
    return f"{filename} - Page {page_number}, Section {chunk_index + 1}"


def clean_text(text: str) -> str:
    """Clean extracted text."""
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove non-printable characters
    text = "".join(char for char in text if char.isprintable() or char.isspace())
    return text.strip()


class Timer:
    """Context manager for timing operations."""

    def __init__(self):
        self.start_time = None
        self.elapsed = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time