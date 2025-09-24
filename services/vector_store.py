"""Vector store service for Qdrant operations."""

from typing import Dict, List, Optional, Tuple
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from config.settings import settings
from utils.helpers import Timer


class VectorStore:
    """Service for vector database operations."""

    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
        self.collection_name = settings.qdrant_collection

    def initialize_collection(self, vector_size: int):
        """Create or recreate collection."""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if not exists:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize collection: {str(e)}")


    def add_documents(
        self,
        chunks: List[Dict],
        embeddings: List[np.ndarray]
    ) -> bool:
        """
        Add document chunks to vector store with unique IDs.
        
        Returns:
            Success status
        """
        try:
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Generate unique ID instead of using sequential numbers
                unique_id = str(uuid.uuid4())
                
                point = PointStruct(
                    id=unique_id,  # Use UUID instead of sequential ID
                    vector=embedding,
                    payload={
                        "doc_id": chunk["doc_id"],
                        "filename": chunk["filename"],
                        "page_number": chunk["page_number"],
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"]
                    }
                )
                points.append(point)
            
            print(f"[VECTOR_STORE] Uploading {len(points)} points with unique IDs")
            
            # Batch upload
            self.client.upload_points(
                collection_name=self.collection_name,
                points=points
            )
            
            # Verify upload
            collection_info = self.client.get_collection(self.collection_name)
            print(f"[VECTOR_STORE] Total points after upload: {collection_info.points_count}")
            
            return True
            
        except Exception as e:
            print(f"Failed to add documents to vector store: {str(e)}")
            return False
        
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        doc_filter: Optional[str] = None,
        threshold: float = 0.7
    ) -> Tuple[List[Dict], List[float], float]:
        """
        Search for similar chunks.
        
        Returns:
            Tuple of (results, scores, time_taken)
        """
        with Timer() as timer:
            # Build filter if document specified
            query_filter = None
            if doc_filter:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_filter)
                        )
                    ]
                )
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k,
                query_filter=query_filter,
                with_payload=True
            )
            
            # Filter by similarity threshold
            results = []
            scores = []
            
            for result in search_results:
                if result.score >= threshold:
                    results.append(result.payload)
                    scores.append(result.score)
        
        return results, scores, timer.elapsed

    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a document."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="doc_id",
                            match=MatchValue(value=doc_id)
                        )
                    ]
                )
            )
            return True
        except Exception as e:
            print(f"Failed to delete document: {str(e)}")
            return False