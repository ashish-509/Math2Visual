# Loads docs, chunks them, builds index, retrieves relevant context

import os
import logging
from typing import Optional, List, Dict
from .chunker import DocumentChunker
from .retriever import SimpleRetriever
from .context_manager import ContextManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, 
                 chunk_size=800,
                 chunk_overlap=150, 
                 max_context_tokens=2000,
                 top_k_chunks=3):
        """
        Initialize RAG pipeline with configuration
        
        chunk_size: max characters per document chunk
        chunk_overlap: overlap between chunks to preserve context
        max_context_tokens: max tokens to use for retrieved context
        top_k_chunks: number of chunks to retrieve for each query
        """
        self.chunker = DocumentChunker(chunk_size, chunk_overlap)
        self.retriever = SimpleRetriever()
        self.context_mgr = ContextManager(max_context_tokens)
        self.top_k = top_k_chunks
        self.is_indexed = False
        
        logger.info(f"RAG pipeline initialized: chunk_size={chunk_size}, top_k={top_k_chunks}")
    
    def load_and_index(self, doc_path):
        """
        Load documentation file and build search index
        doc_path: path to markdown file (like crawler/markdown_output.md)
        """
        if not os.path.exists(doc_path):
            logger.error(f"Doc file not found: {doc_path}")
            return False
        
        logger.info(f"Loading documentation from {doc_path}")
        
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                doc_text = f.read()
            
            logger.info(f"Loaded {len(doc_text)} characters")
            
            # chunk the document
            chunks = self.chunker.chunk_by_section(doc_text)
            stats = self.chunker.get_stats(chunks)
            
            logger.info(f"Created {stats['num_chunks']} chunks")
            logger.info(f"Avg chunk size: {stats['avg_chunk_size']} chars")
            
            # build retrieval index
            self.retriever.index_chunks(chunks)
            self.is_indexed = True
            
            logger.info("Indexing complete - RAG ready")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load/index docs: {e}")
            return False
    
    def retrieve_context(self, query):
        """
        Retrieve relevant chunks for a query
        Returns formatted context string ready for LLM
        """
        if not self.is_indexed:
            logger.warning("RAG not indexed yet - no context available")
            return ""
        
        # get relevant chunks
        results = self.retriever.retrieve(query, top_k=self.top_k)
        
        if not results:
            logger.warning(f"No relevant docs found for query: {query[:50]}...")
            return ""
        
        # log what we found
        logger.info(f"Retrieved {len(results)} chunks (scores: {[r['score'] for r in results]})")
        
        # build context string
        context = self.context_mgr.build_context(results)
        token_est = self.context_mgr.estimate_tokens(context)
        
        logger.info(f"Built context: ~{token_est} tokens")
        
        return context
    
    def augment_prompt(self, user_query, base_prompt="", max_total_tokens=4000):
        """
        Complete RAG workflow - retrieve docs and build augmented prompt
        user_query: what the user wants to do
        base_prompt: any existing prompt text to prepend
        max_total_tokens: maximum total tokens for the final prompt
        """
        context = self.retrieve_context(user_query)
        augmented = self.context_mgr.build_rag_prompt(user_query, context, base_prompt, max_total_tokens)
        
        return augmented
    
    def get_stats(self):
        """Get pipeline statistics"""
        if not self.is_indexed:
            return {"status": "not indexed"}
        
        retriever_stats = self.retriever.get_stats()
        return {
            "status": "ready",
            "num_chunks": retriever_stats['num_chunks'],
            "vocab_size": retriever_stats['vocab_size'],
            "top_k": self.top_k,
            "max_context_tokens": self.context_mgr.max_tokens
        }


# convenience function to create a ready-to-use RAG pipeline
def create_rag_pipeline(doc_path=None):
    """
    Create and initialize RAG pipeline with default settings
    doc_path: optional path to docs. If None, uses default location
    """
    if doc_path is None:
        # default to crawler output
        doc_path = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', 
            'crawler', 
            'markdown_output.md'
        )
    
    rag = RAGPipeline()
    
    # try to load and index
    if os.path.exists(doc_path):
        rag.load_and_index(doc_path)
    else:
        logger.warning(f"Doc file not found at {doc_path} - RAG will work without context")
    
    return rag
