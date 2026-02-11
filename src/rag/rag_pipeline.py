"""
Enhanced RAG pipeline with semantic retrieval using embeddings.
Loads docs, chunks them intelligently, builds vector index, retrieves relevant context.
"""

import os
import logging
from typing import Optional, List, Dict
from .chunker import SemanticChunker, DocumentChunker
from .retriever import SemanticRetriever, create_semantic_retriever, SEMANTIC_AVAILABLE
from .context_manager import ContextManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticRAGPipeline:
    
    def __init__(
        self, 
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        max_context_tokens: int = 2000,
        top_k_chunks: int = 5,
        model_name: Optional[str] = None,
        use_reranker: bool = True,
        min_score: float = 0.3
    ):
        
        if not SEMANTIC_AVAILABLE:
            raise ImportError(
                "Semantic search not available. "
                "Install: pip install sentence-transformers faiss-cpu"
            )
        
        self.chunker = SemanticChunker(chunk_size, chunk_overlap)
        self.retriever = None  # Initialized when documents loaded
        self.context_mgr = ContextManager(max_context_tokens)
        
        self.top_k = top_k_chunks
        self.min_score = min_score
        self.model_name = model_name
        self.use_reranker = use_reranker
        self.is_indexed = False
        
        logger.info(f" Semantic RAG pipeline initialized")
        logger.info(f"  Chunk size: {chunk_size}, Top-K: {top_k_chunks}")
        logger.info(f"  Model: {model_name or 'auto-select'}")
    

    def load_and_index(self, doc_path: str, force_reindex: bool = False) -> bool:
        
        if not os.path.exists(doc_path):
            logger.error(f"Documentation file not found: {doc_path}")
            return False
        
        logger.info(f"Loading documentation from {doc_path}")
        
        try:
            # Load document
            with open(doc_path, 'r', encoding='utf-8') as f:
                doc_text = f.read()
            
            logger.info(f"Loaded {len(doc_text):,} characters")
            
            # Chunk the document
            logger.info("Chunking document...")
            chunks = self.chunker.chunk_document(
                doc_text,
                metadata={'source': os.path.basename(doc_path)}
            )
            
            stats = self.chunker.get_stats(chunks)
            logger.info(f" Created {stats['num_chunks']} chunks")
            logger.info(f"  Avg size: {stats['avg_chunk_size']} chars")
            logger.info(f"  Range: {stats['min_size']}-{stats['max_size']} chars")
            
            # Initialize semantic retriever
            logger.info("Initializing semantic retriever...")
            self.retriever = create_semantic_retriever(
                model_name=self.model_name,
                use_reranker=self.use_reranker
            )
            
            # Index documents
            logger.info("Building semantic index (this may take a moment)...")
            success = self.retriever.index_documents(
                chunks,
                force_reindex=force_reindex,
                show_progress=True
            )
            
            if success:
                self.is_indexed = True
                logger.info(" RAG pipeline ready!")
                return True
            else:
                logger.error("Failed to index documents")
                return False
            
        except Exception as e:
            logger.error(f"Failed to load/index docs: {e}", exc_info=True)
            return False

    
    def retrieve_context(
        self, 
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> str:
        
        
        if not self.is_indexed or not self.retriever:
            logger.warning("RAG not indexed - no context available")
            return ""
        
        top_k = top_k or self.top_k
        min_score = min_score or self.min_score
        
        # Retrieve with semantic search
        results = self.retriever.retrieve(query, top_k, min_score)
        
        if not results:
            logger.warning(f"No relevant docs found for: '{query[:50]}...'")
            return ""
        
        # Log retrieval info
        scores = [r.get('score', 0) for r in results]
        logger.info(f" Retrieved {len(results)} chunks")
        logger.info(f"  Scores: {['%.3f' % s for s in scores]}")
        
        # Format results for old context manager compatibility
        formatted_results = []
        for i, doc in enumerate(results):
            formatted_results.append({
                'chunk': {
                    'text': doc.get('text', doc.get('content', '')),
                    'header': doc.get('header', 'Unknown'),
                    'size': doc.get('size', 0)
                },
                'score': doc.get('score', 0),
                'rank': i + 1
            })
        
        # Build context string
        context = self.context_mgr.build_context(formatted_results)
        token_est = self.context_mgr.estimate_tokens(context)
        
        logger.info(f"  Context: ~{token_est} tokens")
        
        return context
    

    def retrieve_raw(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[Dict]:
        
        
        if not self.is_indexed or not self.retriever:
            return []
        
        top_k = top_k or self.top_k
        min_score = min_score or self.min_score
        
        return self.retriever.retrieve(query, top_k, min_score)
    

    def augment_prompt(
        self, 
        user_query: str, 
        base_prompt: str = "", 
        max_total_tokens: int = 4000
    ) -> str:
        
        context = self.retrieve_context(user_query)
        augmented = self.context_mgr.build_rag_prompt(
            user_query, 
            context, 
            base_prompt, 
            max_total_tokens
        )
        
        return augmented

    
    def get_stats(self) -> Dict:
        if not self.is_indexed or not self.retriever:
            return {"status": "not indexed"}
        
        retriever_status = self.retriever.get_status()
        
        return {
            "status": "ready",
            "model": retriever_status.get('model', 'unknown'),
            "embedding_dim": retriever_status.get('embedding_dim', 0),
            "num_documents": retriever_status.get('num_documents', 0),
            "reranker_enabled": retriever_status.get('reranker', False),
            "top_k": self.top_k,
            "min_score": self.min_score,
            "max_context_tokens": self.context_mgr.max_tokens
        }
    

    def search_similar_docs(self, doc_index: int, top_k: int = 5) -> List[Dict]:
        if not self.is_indexed or not self.retriever:
            return []
        
        return self.retriever.search_similar(doc_index, top_k)


# Backward compatible class
class RAGPipeline(SemanticRAGPipeline):
    
    def __init__(
        self,
        chunk_size=800,
        chunk_overlap=150,
        max_context_tokens=2000,
        top_k_chunks=3
    ):
        # Check if semantic search is available
        if SEMANTIC_AVAILABLE:
            logger.info("Using enhanced semantic RAG")
            super().__init__(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_context_tokens=max_context_tokens,
                top_k_chunks=top_k_chunks
            )
        else:
            # Fallback to old implementation
            logger.warning("Semantic search not available, using basic TF-IDF")
            from .retriever import SimpleRetriever
            
            self.chunker = DocumentChunker(chunk_size, chunk_overlap)
            self.retriever = SimpleRetriever()
            self.context_mgr = ContextManager(max_context_tokens)
            self.top_k = top_k_chunks
            self.is_indexed = False


def create_rag_pipeline(
    doc_path: Optional[str] = None,
    **kwargs
) -> SemanticRAGPipeline:
    
    # Auto-detect doc path if not provided
    if doc_path is None:
        doc_path = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', 
            'crawler', 
            'markdown_output.md'
        )
    
    # Create pipeline
    rag = SemanticRAGPipeline(**kwargs)
    
    # Try to load and index
    if os.path.exists(doc_path):
        logger.info(f"Auto-loading documentation: {doc_path}")
        rag.load_and_index(doc_path)
    else:
        logger.warning(f"Doc file not found: {doc_path}")
        logger.warning("RAG will work without documentation context")
    
    return rag
