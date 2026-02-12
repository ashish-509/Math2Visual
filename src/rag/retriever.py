"""
Semantic retrieval using embeddings and FAISS vector search.
Uses BGE models for embeddings and optional cross-encoder reranking.
"""

import os
import logging
import pickle
from pathlib import Path
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import faiss
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

logger = logging.getLogger(__name__)

# Models ordered by quality (best first)
EMBEDDING_MODELS = [
    "BAAI/bge-large-en-v1.5",
    "BAAI/bge-base-en-v1.5", 
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]

RERANKER_MODELS = [
    "BAAI/bge-reranker-base",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
]


class SemanticRetriever:
    """Vector-based document retrieval with optional reranking."""
    
    def __init__(self, model_name=None, use_reranker=True, cache_dir=None, device="cpu"):
        if not HAS_DEPS:
            raise ImportError("Install: pip install sentence-transformers faiss-cpu")
        
        self.device = device
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), "../../.cache/embeddings")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load models
        self.model = self._load_model(model_name)
        self.model_name = getattr(self.model, 'model_card_data', {}).get('model_name', model_name or 'unknown')
        self.dim = self.model.get_sentence_embedding_dimension()
        
        self.reranker = self._load_reranker() if use_reranker else None
        
        # Index state
        self.index = None
        self.docs = []
        self.embeddings = None
        
        logger.info(f"Retriever ready: {self.model_name}, dim={self.dim}, reranker={bool(self.reranker)}")
    
    def _load_model(self, preferred=None):
        """Try to load the best available model."""
        models = [preferred] if preferred else EMBEDDING_MODELS
        for name in models:
            if not name:
                continue
            try:
                return SentenceTransformer(name, device=self.device)
            except Exception as e:
                logger.warning(f"Couldn't load {name}: {e}")
        raise RuntimeError("No embedding model available")
    
    def _load_reranker(self):
        """Load a reranking model if available."""
        for name in RERANKER_MODELS:
            try:
                return CrossEncoder(name, device=self.device)
            except:
                continue
        return None
    
    def index_documents(self, documents, force=False, show_progress=True):
        """Build vector index from documents."""
        if not documents:
            return False
        
        cache_file = os.path.join(self.cache_dir, f"idx_{self.dim}.pkl")
        
        # Try cache first
        if not force and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                if len(data['docs']) == len(documents):
                    self.docs, self.embeddings, self.index = data['docs'], data['emb'], data['idx']
                    logger.info(f"Loaded {len(self.docs)} docs from cache")
                    return True
            except:
                pass
        
        # Get text from docs
        texts = [d.get('text') or d.get('content', '') for d in documents]
        texts = [t for t in texts if t]
        
        if not texts:
            logger.error("No text in documents")
            return False
        
        try:
            logger.info(f"Encoding {len(texts)} documents...")
            self.embeddings = self.model.encode(
                texts, batch_size=32, show_progress_bar=show_progress,
                convert_to_numpy=True, normalize_embeddings=True
            )
            
            # Build FAISS index (inner product = cosine for normalized vectors)
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(self.embeddings.astype('float32'))
            self.docs = documents
            
            # Save cache
            with open(cache_file, 'wb') as f:
                pickle.dump({'docs': self.docs, 'emb': self.embeddings, 'idx': self.index}, f)
            
            logger.info(f"Indexed {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            return False
    
    def retrieve(self, query, top_k=5, min_score=0.0):
        """Find most relevant documents for a query."""
        if not self.index or not query.strip():
            return []
        
        try:
            q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            
            # Search (get extra candidates if we'll rerank)
            k = top_k * 3 if self.reranker else top_k
            scores, indices = self.index.search(q_emb.astype('float32'), min(k, len(self.docs)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score >= min_score:
                    doc = self.docs[idx].copy()
                    doc['score'] = float(score)
                    results.append(doc)
            
            # Rerank if available
            if self.reranker and results:
                results = self._rerank(query, results)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []
    
    def _rerank(self, query, results):
        """Rerank results using cross-encoder."""
        try:
            pairs = [[query, r.get('text') or r.get('content', '')] for r in results]
            scores = self.reranker.predict(pairs)
            
            for doc, s in zip(results, scores):
                doc['original_score'] = doc['score']
                doc['score'] = float(s)
            
            return sorted(results, key=lambda x: x['score'], reverse=True)
        except:
            return results
    
    def retrieve_context(self, query, top_k=5, min_score=0.0):
        """Get formatted context string from retrieval."""
        results = self.retrieve(query, top_k, min_score)
        if not results:
            return "No relevant documentation found."
        
        parts = []
        for i, doc in enumerate(results, 1):
            text = doc.get('text') or doc.get('content', '')
            header = doc.get('header', '')
            parts.append(f"[{i}] {header}\n{text}" if header else f"[{i}]\n{text}")
        
        return "\n---\n".join(parts)
    
    def get_status(self):
        return {
            "model": self.model_name,
            "dim": self.dim,
            "docs": len(self.docs),
            "indexed": self.index is not None,
            "reranker": self.reranker is not None
        }
    
    def clear_cache(self):
        for f in Path(self.cache_dir).glob("idx_*.pkl"):
            try:
                f.unlink()
            except:
                pass


class HybridRetriever:
    """Combines semantic search with keyword filtering."""
    
    def __init__(self, semantic_retriever):
        self.semantic = semantic_retriever
    
    def retrieve(self, query, top_k=5, min_score=0.0, required_keywords=None, boost_keywords=None):
        # Get semantic results
        k = top_k * 3 if required_keywords else top_k
        results = self.semantic.retrieve(query, k, min_score)
        
        # Filter by required keywords
        if required_keywords:
            results = [r for r in results 
                       if any(kw.lower() in (r.get('text') or '').lower() for kw in required_keywords)]
        
        # Boost scores for matching keywords
        if boost_keywords:
            for r in results:
                text = (r.get('text') or '').lower()
                matches = sum(1 for kw in boost_keywords if kw.lower() in text)
                if matches:
                    r['score'] *= (1 + 0.1 * matches)
            results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_k]


# Factory function
def create_semantic_retriever(documents=None, model_name=None, use_reranker=True, **kwargs):
    retriever = SemanticRetriever(model_name=model_name, use_reranker=use_reranker, **kwargs)
    if documents:
        retriever.index_documents(documents)
    return retriever


# Backward compatibility
class SimpleRetriever:
    """Deprecated - use SemanticRetriever instead."""
    
    def __init__(self):
        self._retriever = None
        self.chunks = []
    
    def index_chunks(self, chunks):
        self.chunks = chunks
        if HAS_DEPS:
            self._retriever = create_semantic_retriever(chunks)
    
    def retrieve(self, query, top_k=3):
        if self._retriever:
            results = self._retriever.retrieve(query, top_k)
            return [{'chunk': r, 'score': r.get('score', 0), 'rank': i+1} for i, r in enumerate(results)]
        return []
