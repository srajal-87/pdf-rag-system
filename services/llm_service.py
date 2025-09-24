"""LLM service for Gemini API integration."""

import json
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai

from config.settings import settings
from utils.helpers import Timer, format_citation


class LLMService:
    """Service for LLM operations."""

    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[Dict],
        only_use_sources: bool = True
    ) -> Tuple[str, float]:
        """
        Generate answer using LLM.
        
        Returns:
            Tuple of (answer, time_taken)
        """
        with Timer() as timer:
            # Build context from retrieved chunks
            context_parts = []
            citations = []
            
            for i, chunk in enumerate(retrieved_chunks):
                context_parts.append(
                    f"[Source {i+1}] "
                    f"{chunk['filename']} - Page {chunk['page_number']}:\n"
                    f"{chunk['text']}\n"
                )
                citations.append(format_citation(
                    chunk['filename'],
                    chunk['page_number'],
                    chunk['chunk_index']
                ))
            
            context = "\n".join(context_parts)
            
            # Build prompt with strict grounding instructions
            system_prompt = (
                "You are a helpful assistant that answers questions based ONLY on "
                "the provided source documents. You must:\n"
                "1. Only use information from the provided sources\n"
                "2. Never make up or hallucinate information\n"
                "3. If the sources don't contain the answer, say 'I don't know' or "
                "'The provided documents don't contain this information'\n"
                "4. Reference specific sources when providing information\n"
                "5. Be accurate and precise in your answers\n"
            )
            
            if not retrieved_chunks and only_use_sources:
                answer = (
                    "I don't know. No relevant sources were found in the "
                    "uploaded documents to answer your question."
                )
            else:
                prompt = (
                    f"{system_prompt}\n\n"
                    f"Context from documents:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer (using ONLY the above sources):"
                )
                
                try:
                    response = self.model.generate_content(prompt)
                    answer = response.text
                    
                    # Add citations
                    if citations:
                        answer += "\n\nSources:\n" + "\n".join(
                            f"- {citation}" for citation in citations[:3]
                        )
                except Exception as e:
                    answer = f"Error generating answer: {str(e)}"
        
        return answer, timer.elapsed

    def validate_api_key(self) -> bool:
        """Validate Gemini API key."""
        try:
            # Try a simple generation
            response = self.model.generate_content("Hello")
            return True
        except Exception:
            return False