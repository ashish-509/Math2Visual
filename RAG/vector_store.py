"""
CREATES SEARCHABLE EMBEDDINGS
This module handles creating and managing vector embeddings for the processed Manim documentation using ChromaDB.
"""

import os
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from uuid import uuid4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ManimeVectorStore:
    def __init__(self, 
                 collection_name: str = "manim_docs",
                 model_name: str = "all-MiniLM-L6-v2",
                 persist_directory: str = "./chroma_db"):
                 
        """
        Initialize the vector store for Manim documentation.
        Args:
            collection_name: Name of the ChromaDB collection
            model_name: Sentence transformer model name
            persist_directory: Directory to persist ChromaDB
        """

        self.collection_name = collection_name
        self.persist_directory = os.path.abspath(persist_directory)
        
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self._get_or_create_collection()
        
        logger.info(f"Vector store initialized with collection: {collection_name}")
    
    def _get_or_create_collection(self):
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Manim documentation embeddings"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def chunk_document(self, doc: Dict[str, Any], chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Split a document into smaller chunks for better retrieval.
        
        Args:
            doc: Document to chunk
            chunk_size: Maximum size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of document chunks
        """
        content = doc['content']
        chunks = []
        
        sentences = content.split('. ')
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append({
                    'id': f"{doc['url']}#chunk_{chunk_index}",
                    'content': current_chunk.strip(),
                    'title': doc['title'],
                    'url': doc['url'],
                    'chunk_index': chunk_index,
                    'type': 'content'
                })
                
                overlap_text = '. '.join(current_chunk.split('. ')[-2:]) if '. ' in current_chunk else ""
                current_chunk = overlap_text + ". " + sentence if overlap_text else sentence
                chunk_index += 1
            else:
                current_chunk += ". " + sentence if current_chunk else sentence
        
        if current_chunk.strip():
            chunks.append({
                'id': f"{doc['url']}#chunk_{chunk_index}",
                'content': current_chunk.strip(),
                'title': doc['title'],
                'url': doc['url'],
                'chunk_index': chunk_index,
                'type': 'content'
            })
        
        for i, code in enumerate(doc.get('code_examples', [])):
            if len(code) > 50: 
                chunks.append({
                    'id': f"{doc['url']}#code_{i}",
                    'content': f"Code example from {doc['title']}:\n\n{code}",
                    'title': doc['title'],
                    'url': doc['url'],
                    'chunk_index': f"code_{i}",
                    'type': 'code'
                })
        
        return chunks
    
    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 50):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            batch_size: Batch size for processing
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        # Process in batches
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            
            # Extract content for embedding
            texts = [chunk['content'] for chunk in batch]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Prepare data for ChromaDB
            ids = [chunk['id'] for chunk in batch]
            metadatas = [{
                'title': chunk['title'],
                'url': chunk['url'],
                'chunk_index': str(chunk['chunk_index']),
                'type': chunk['type']
            } for chunk in batch]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas
            )
        
        logger.info(f"Successfully added {len(all_chunks)} chunks to vector store")
    
    def search(self, query: str, n_results: int = 5, filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using semantic similarity.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_type: Filter by document type ('content' or 'code')
            
        Returns:
            List of relevant documents with scores
        """

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Prepare where clause for filtering
        where_clause = {"type": filter_type} if filter_type else None
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where=where_clause
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        count = self.collection.count()
        return {
            'total_chunks': count,
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model._modules['0'].__class__.__name__
        }
    
    def clear_collection(self):
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()
        logger.info("Collection cleared")

if __name__ == "__main__":
    from document_processor import ManimeDocumentProcessor
    
    vector_store = ManimeVectorStore()
    
    processor = ManimeDocumentProcessor()
    docs = processor.load_processed_docs()
    
    if docs:
        print(f"Adding {len(docs)} documents to vector store...")
        vector_store.add_documents(docs)
        
        query = "how to create animation with manim"
        results = vector_store.search(query, n_results=3)
        
        print(f"\nSearch results for: '{query}'")
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result['metadata']['title']}")
            print(f"   Content: {result['content'][:200]}...")
            print(f"   URL: {result['metadata']['url']}")
    else:
        print("No documents found. Please run document_processor.py first.")
