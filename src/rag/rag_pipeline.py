"""
RAG pipeline for Manim documentation. Loads docs, chunks them, builds a
vector index, and retrieves relevant context for queries.
"""

import os
import logging
from .chunker import SemanticChunker, DocumentChunker
from .retriever import SemanticRetriever, create_semantic_retriever, HAS_DEPS as SEMANTIC_AVAILABLE
from .context_manager import ContextManager

logger = logging.getLogger(__name__)


class SemanticRAGPipeline:
    """RAG pipeline with semantic search capabilities."""
    
    def __init__(self, chunk_size=600, chunk_overlap=100, max_context_tokens=2000,
                 top_k_chunks=5, model_name=None, use_reranker=True, min_score=0.3):
        
        if not SEMANTIC_AVAILABLE:
            raise ImportError("Install: pip install sentence-transformers faiss-cpu")
        
        self.chunker = SemanticChunker(chunk_size, chunk_overlap)
        self.retriever = None
        self.context_mgr = ContextManager(max_context_tokens)
        
        self.top_k = top_k_chunks
        self.min_score = min_score
        self.model_name = model_name
        self.use_reranker = use_reranker
        self.is_indexed = False
        
        logger.info(f"RAG pipeline initialized (chunk_size={chunk_size}, top_k={top_k_chunks})")
    
    def load_and_index(self, doc_path, force_reindex=False):
        """Load documentation and build the search index."""
        if not os.path.exists(doc_path):
            logger.error(f"File not found: {doc_path}")
            return False
        
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            logger.info(f"Loaded {len(text):,} chars from {os.path.basename(doc_path)}")
            
            # Chunk the document
            chunks = self.chunker.chunk_document(text, {'source': os.path.basename(doc_path)})
            stats = self.chunker.get_stats(chunks)
            logger.info(f"Created {stats['count']} chunks (avg {stats['avg_size']} chars)")
            
            # Build index
            self.retriever = create_semantic_retriever(
                model_name=self.model_name, use_reranker=self.use_reranker
            )
            
            if self.retriever.index_documents(chunks, force=force_reindex):
                self.is_indexed = True
                logger.info("Index ready")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            return False
    
    def retrieve_context(self, query, top_k=None, min_score=None):
        """Get relevant context for a query as formatted string."""
        if not self.is_indexed or not self.retriever:
            return ""
        
        results = self.retriever.retrieve(query, top_k or self.top_k, min_score or self.min_score)
        
        if not results:
            return ""
        
        # Format for context manager
        formatted = [
            {'chunk': {'text': r.get('text', ''), 'header': r.get('header', ''), 'size': len(r.get('text', ''))},
             'score': r.get('score', 0), 'rank': i + 1}
            for i, r in enumerate(results)
        ]
        
        return self.context_mgr.build_context(formatted)
    
    def retrieve_raw(self, query, top_k=None, min_score=None):
        """Get raw results (dicts with text, score, etc)."""
        if not self.is_indexed or not self.retriever:
            return []
        return self.retriever.retrieve(query, top_k or self.top_k, min_score or self.min_score)
    
    def augment_prompt(self, query, base_prompt="", max_tokens=4000):
        """Add retrieved context to a prompt."""
        context = self.retrieve_context(query)
        return self.context_mgr.build_rag_prompt(query, context, base_prompt, max_tokens)
    
    def get_stats(self):
        if not self.is_indexed:
            return {"status": "not indexed"}
        
        s = self.retriever.get_status()
        return {
            "status": "ready",
            "model": s.get('model'),
            "docs": s.get('docs', 0),
            "reranker": s.get('reranker', False),
            "top_k": self.top_k
        }


# Backward compatible alias
class RAGPipeline(SemanticRAGPipeline):
    def __init__(self, chunk_size=800, chunk_overlap=150, max_context_tokens=2000, top_k_chunks=3):
        if SEMANTIC_AVAILABLE:
            super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                           max_context_tokens=max_context_tokens, top_k_chunks=top_k_chunks)
        else:
            # Basic fallback
            from .retriever import SimpleRetriever
            self.chunker = DocumentChunker(chunk_size, chunk_overlap)
            self.retriever = SimpleRetriever()
            self.context_mgr = ContextManager(max_context_tokens)
            self.top_k = top_k_chunks
            self.is_indexed = False


def create_rag_pipeline(doc_path=None, auto_sync=False, **kwargs):
    """Factory function to create and optionally initialize a RAG pipeline."""
    
    # Try auto-sync if requested
    if auto_sync:
        try:
            from crawler.docs_sync import get_docs_sync
            sync = get_docs_sync()
            if sync.check_for_updates():
                logger.info("Syncing documentation...")
                success, msg = sync.sync_docs()
                logger.info(f"Sync {'done' if success else 'failed'}: {msg}")
        except Exception as e:
            logger.warning(f"Auto-sync skipped: {e}")
    
    # Find doc path
    if not doc_path:
        base = os.path.dirname(__file__)
        synced = os.path.join(base, '..', '..', 'crawler', 'docs_output', 'manim_docs_combined.md')
        legacy = os.path.join(base, '..', '..', 'crawler', 'markdown_output.md')
        
        doc_path = synced if os.path.exists(synced) else legacy
    
    # Create pipeline
    rag = SemanticRAGPipeline(**kwargs)
    
    if os.path.exists(doc_path):
        rag.load_and_index(doc_path)
    else:
        logger.warning(f"No docs at {doc_path}")
    
    return rag
