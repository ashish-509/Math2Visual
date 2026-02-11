"""
Enhanced semantic retrieval using embeddings and vector search.

Models used:
- Embeddings: BAAI/bge-large-en-v1.5 (SOTA open-source)
- Fallback: sentence-transformers/all-MiniLM-L6-v2 (lightweight)
- Reranker: BAAI/bge-reranker-base (optional)
"""

import os
import logging
import pickle
import json
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import faiss
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    logging.warning("Semantic search dependencies not installed. Install: pip install sentence-transformers faiss-cpu")

logger = logging.getLogger(__name__)


class SemanticRetriever:
 
    DEFAULT_MODELS = [
        "BAAI/bge-large-en-v1.5",       # Best quality (33.5M params, 1024 dim)
        "BAAI/bge-base-en-v1.5",        # Good balance (109M params, 768 dim)
        "sentence-transformers/all-mpnet-base-v2",  # Excellent general (110M params, 768 dim)
        "sentence-transformers/all-MiniLM-L6-v2",   # Fast & lightweight (22M params, 384 dim)
    ]
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        use_reranker: bool = True,
        cache_dir: Optional[str] = None,
        device: str = "cpu"
    ):
        
        if not SEMANTIC_AVAILABLE:
            raise ImportError("Please install: pip install sentence-transformers faiss-cpu")
        
        self.device = device
        self.use_reranker = use_reranker
        
        # Setup cache directory
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), 
            "../../.cache/embeddings"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load embedding model (try best available)
        self.model = self._load_best_model(model_name)
        self.model_name = self.model._model_card_vars.get('model_name', 'unknown')
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Load reranker if requested
        self.reranker = None
        if use_reranker:
            self.reranker = self._load_reranker()
        
        # Vector index and documents
        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        
        logger.info(f" Semantic retriever initialized")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  Reranker: {'' if self.reranker else '✗'}")


    def _load_best_model(self, preferred_model: Optional[str] = None) -> SentenceTransformer:
        
        models_to_try = [preferred_model] if preferred_model else self.DEFAULT_MODELS
        
        for model_name in models_to_try:
            if model_name is None:
                continue
            try:
                logger.info(f"Loading embedding model: {model_name}")
                model = SentenceTransformer(model_name, device=self.device)
                logger.info(f" Successfully loaded {model_name}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        raise RuntimeError("Could not load any embedding model. Check internet connection.")
    

    def _load_reranker(self) -> Optional[CrossEncoder]:
        
        reranker_models = [
            "BAAI/bge-reranker-base",
            "cross-encoder/ms-marco-MiniLM-L-12-v2",
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ]
        
        for model_name in reranker_models:
            try:
                logger.info(f"Loading reranker: {model_name}")
                reranker = CrossEncoder(model_name, device=self.device)
                logger.info(f" Reranker loaded: {model_name}")
                return reranker
            except Exception as e:
                logger.warning(f"Failed to load reranker {model_name}: {e}")
                continue
        
        logger.warning("No reranker available, using semantic search only")
        return None

    
    def index_documents(
        self, 
        documents: List[Dict],
        force_reindex: bool = False,
        show_progress: bool = True
    ) -> bool:
        
        if not documents:
            logger.warning("No documents to index")
            return False
        
        cache_path = os.path.join(self.cache_dir, f"index_{self.model_name.replace('/', '_')}.pkl")
        
        # Try loading from cache
        if not force_reindex and os.path.exists(cache_path):
            try:
                logger.info("Loading cached embeddings...")
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                    
                # Verify cache is compatible
                if len(cached['documents']) == len(documents):
                    self.documents = cached['documents']
                    self.embeddings = cached['embeddings']
                    self.index = cached['index']
                    logger.info(f" Loaded {len(self.documents)} documents from cache")
                    return True
                else:
                    logger.info("Cache size mismatch, reindexing...")
            except Exception as e:
                logger.warning(f"Cache load failed: {e}, reindexing...")
        
        # Extract text from documents
        texts = []
        for doc in documents:
            text = doc.get('text') or doc.get('content', '')
            if text:
                texts.append(text)
        
        if not texts:
            logger.error("No valid text found in documents")
            return False
        
        try:
            # Compute embeddings with optimized settings
            logger.info(f"Computing embeddings for {len(texts)} documents...")
            
            # Use batch encoding for efficiency
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalization for cosine similarity
                device=self.device
            )
            
            # Build FAISS index (using Inner Product for normalized vectors = cosine similarity)
            logger.info("Building FAISS index...")
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(embeddings.astype('float32'))
            
            self.documents = documents
            self.embeddings = embeddings
            
            # Cache the index
            logger.info("Caching index...")
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'embeddings': self.embeddings,
                    'index': self.index,
                    'model_name': self.model_name
                }, f)
            
            logger.info(f" Successfully indexed {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            return False

    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict]:
        
        if not self.index or not self.documents:
            logger.error("Index not initialized. Call index_documents() first.")
            return []
        
        if not query.strip():
            logger.warning("Empty query")
            return []
        
        try:
            # Encode query with same normalization
            query_embedding = self.model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device
            )
            
            # Search FAISS index (get more candidates if using reranker)
            search_k = top_k * 3 if self.reranker else top_k
            scores, indices = self.index.search(
                query_embedding.astype('float32'), 
                min(search_k, len(self.documents))
            )
            
            # Format initial results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score >= min_score:  # FAISS returns -1 for invalid
                    doc = self.documents[idx].copy()
                    doc['score'] = float(score)
                    doc['index'] = int(idx)
                    results.append(doc)
            
            # Apply reranking if available
            if self.reranker and results:
                results = self._rerank(query, results, top_k)
            else:
                results = results[:top_k]
            
            scores_str = [f"{r['score']:.3f}" for r in results]
            logger.info(f"Retrieved {len(results)} documents (scores: {scores_str})")
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []


    # Rerank results using cross-encoder for better precision.
    def _rerank(self, query: str, results: List[Dict], top_k: int) -> List[Dict]:
        
        try:
            # Prepare query-document pairs
            pairs = []
            for doc in results:
                text = doc.get('text') or doc.get('content', '')
                pairs.append([query, text])
            
            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Update scores
            for doc, score in zip(results, rerank_scores):
                doc['rerank_score'] = float(score)
                doc['original_score'] = doc['score']
                doc['score'] = float(score)  # Use rerank score as primary
            
            # Sort by rerank score
            results.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f" Reranked {len(results)} results")
            return results[:top_k]
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, using original scores")
            return results[:top_k]

    
    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        include_metadata: bool = True
    ) -> str:
       
        results = self.retrieve(query, top_k, min_score)
        
        if not results:
            return "No relevant documentation found."
        
        context_parts = []
        for i, doc in enumerate(results, 1):
            score = doc.get('score', 0)
            text = doc.get('text') or doc.get('content', '')
            
            if include_metadata:
                header = doc.get('header', 'Unknown')
                context_parts.append(
                    f"[Chunk {i}] Score: {score:.3f} | Section: {header}\n{text}\n"
                )
            else:
                context_parts.append(text)
        
        return "\n---\n".join(context_parts)

    
    def search_similar(self, doc_index: int, top_k: int = 5) -> List[Dict]:
        
        if not self.embeddings or doc_index >= len(self.embeddings):
            return []
        
        query_emb = self.embeddings[doc_index:doc_index+1]
        scores, indices = self.index.search(query_emb.astype('float32'), top_k + 1)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != doc_index and idx >= 0:  # Skip self
                doc = self.documents[idx].copy()
                doc['score'] = float(score)
                results.append(doc)
        
        return results[:top_k]
    

    def get_status(self) -> Dict:
        
        return {
            "model": self.model_name,
            "embedding_dim": self.embedding_dim,
            "num_documents": len(self.documents),
            "index_ready": self.index is not None,
            "reranker": self.reranker is not None,
            "device": self.device,
            "cache_dir": self.cache_dir
        }
    
    def clear_cache(self):
       
        cache_files = Path(self.cache_dir).glob("index_*.pkl")
        for f in cache_files:
            try:
                f.unlink()
                logger.info(f"Deleted cache: {f}")
            except Exception as e:
                logger.warning(f"Failed to delete {f}: {e}")


class HybridRetriever:

    # Hybrid retrieval combining semantic search with keyword filtering. Provides best of both worlds: semantic understanding + exact matching.
    
    def __init__(self, semantic_retriever: SemanticRetriever):
        self.semantic = semantic_retriever

    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        required_keywords: Optional[List[str]] = None,
        boost_keywords: Optional[List[str]] = None
    ) -> List[Dict]:
        
        # Get semantic results (fetch more for filtering)
        initial_k = top_k * 3 if required_keywords else top_k
        results = self.semantic.retrieve(query, initial_k, min_score)
        
        # Apply required keyword filter
        if required_keywords:
            filtered = []
            for doc in results:
                text = (doc.get('text') or doc.get('content', '')).lower()
                if any(kw.lower() in text for kw in required_keywords):
                    filtered.append(doc)
            results = filtered
        
        # Boost scores for documents containing boost keywords
        if boost_keywords:
            for doc in results:
                text = (doc.get('text') or doc.get('content', '')).lower()
                boost_count = sum(1 for kw in boost_keywords if kw.lower() in text)
                if boost_count > 0:
                    doc['score'] = doc['score'] * (1 + 0.1 * boost_count)
                    doc['boosted'] = True
            
            # Re-sort after boosting
            results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_k]


# Factory function for backward compatibility with old code
def create_semantic_retriever(
    documents: Optional[List[Dict]] = None,
    model_name: Optional[str] = None,
    use_reranker: bool = True,
    **kwargs
) -> SemanticRetriever:
    
    retriever = SemanticRetriever(
        model_name=model_name,
        use_reranker=use_reranker,
        **kwargs
    )
    
    if documents:
        retriever.index_documents(documents)
    
    return retriever


# Simple wrapper maintaining old API (for backward compatibility)
class SimpleRetriever:
    
    def __init__(self):
        logger.warning("SimpleRetriever is deprecated. Use SemanticRetriever instead.")
        self.semantic_retriever = None
        self.chunks = []


    def index_chunks(self, chunks):
        self.chunks = chunks
        if SEMANTIC_AVAILABLE:
            self.semantic_retriever = create_semantic_retriever(chunks)
        else:
            logger.error("Semantic search not available. Install: pip install sentence-transformers faiss-cpu")
    

    def retrieve(self, query, top_k=3):
        if self.semantic_retriever:
            results = self.semantic_retriever.retrieve(query, top_k)
            # Convert to old format
            return [{'chunk': r, 'score': r.get('score', 0), 'rank': i+1} 
                    for i, r in enumerate(results)]
        return []

    
    def get_stats(self):
        if self.semantic_retriever:
            status = self.semantic_retriever.get_status()
            return {
                'num_chunks': status['num_documents'],
                'vocab_size': status['embedding_dim'],
                'avg_chunk_size': 0
            }
        return {'num_chunks': 0, 'vocab_size': 0, 'avg_chunk_size': 0}
