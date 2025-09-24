import io
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF

from config.settings import settings
from utils.helpers import chunk_text, clean_text


class PDFProcessor:
    """Service for PDF parsing and chunking."""

    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

    def extract_text_from_pdf(
        self,
        pdf_content: bytes
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        Extract text from PDF page by page.
        
        Returns:
            Tuple of (pages_data, error_message)
        """
        pages_data = []
        error_message = None
        
        try:
            pdf_stream = io.BytesIO(pdf_content)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    text = page.get_text()
                    cleaned_text = clean_text(text)
                    
                    if cleaned_text:
                        pages_data.append({
                            "page_number": page_num + 1,
                            "text": cleaned_text
                        })
                except Exception as e:
                    # Continue processing other pages even if one fails
                    error_message = f"Error on page {page_num + 1}: {str(e)}"
                    continue
            
            doc.close()
            
            if not pages_data:
                error_message = "No text could be extracted from PDF"
                
        except Exception as e:
            error_message = f"Failed to process PDF: {str(e)}"
        
        return pages_data, error_message

    def create_chunks(
        self,
        pages_data: List[Dict],
        doc_id: str,
        filename: str
    ) -> List[Dict]:
        """
        Create overlapping chunks from pages.
        
        Returns:
            List of chunk dictionaries with metadata.
        """
        chunks = []
        chunk_index = 0
        
        for page_data in pages_data:
            page_number = page_data["page_number"]
            page_text = page_data["text"]
            
            # Split page text into chunks
            page_chunks = chunk_text(
                page_text,
                self.chunk_size,
                self.chunk_overlap
            )
            
            for chunk_text_content in page_chunks:
                chunks.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                    "text": chunk_text_content,
                    "metadata": {
                        "doc_id": doc_id,
                        "filename": filename,
                        "page_number": page_number,
                        "chunk_index": chunk_index
                    }
                })
                chunk_index += 1
        
        return chunks

    def process_pdf(
        self,
        pdf_content: bytes,
        doc_id: str,
        filename: str
    ) -> Tuple[List[Dict], int, Optional[str]]:
        """
        Full PDF processing pipeline.
        
        Returns:
            Tuple of (chunks, total_pages, error_message)
        """
        # Extract text from PDF
        pages_data, error_message = self.extract_text_from_pdf(pdf_content)
        
        if not pages_data:
            return [], 0, error_message
        
        # Create chunks
        chunks = self.create_chunks(pages_data, doc_id, filename)
        
        return chunks, len(pages_data), error_message