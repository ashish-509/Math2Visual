import os
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from typing import List, Dict, Any, Optional
import logging
from vector_store import ManimeVectorStore
from document_processor import ManimeDocumentProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Math2VisualRAG:
    def __init__(self, 
                 model_path: str = "./finetuned-codellama-manim",
                 vector_store: Optional[ManimeVectorStore] = None):
        
        """
        Initialize the RAG pipeline for Math2Visual.
        
        Args:
            model_path: Path to the fine-tuned CodeLlama model
            vector_store: Pre-initialized vector store (optional)
        """

        self.model_path = model_path
        
        self.vector_store = vector_store or ManimeVectorStore() # Initialize vector store
        
        self.pipe = self._load_model() # Load the fine-tuned model
        
        logger.info("Math2Visual RAG pipeline initialized")
    
    def _load_model(self):
        try:
            logger.info(f"Loading model from {self.model_path}")
            tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
            model = LlamaForCausalLM.from_pretrained(
                self.model_path, 
                torch_dtype=torch.float16, 
                device_map="auto"
            )
            return pipeline("text-generation", model=model, tokenizer=tokenizer)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def retrieve_relevant_docs(self, 
                             query: str, 
                             n_results: int = 5,
                             include_code: bool = True) -> List[Dict[str, Any]]:
        
        """
        Retrieve relevant documentation based on the query.
        
        Args:
            query: User's input query
            n_results: Number of documents to retrieve
            include_code: Whether to include code examples
            
        Returns:
            List of relevant documents
        """

        results = []
        
        content_results = self.vector_store.search(
            query, 
            n_results=max(1, n_results // 2), 
            filter_type="content"
        )
        results.extend(content_results)
        
        # Search for code examples if requested
        if include_code:
            code_results = self.vector_store.search(
                query, 
                n_results=max(1, n_results // 2), 
                filter_type="code"
            )
            results.extend(code_results)
        
        # Sort by relevance (distance) and limit results
        results.sort(key=lambda x: x.get('distance', float('inf')))
        return results[:n_results]
    
    def create_enhanced_prompt(self, 
                             user_query: str, 
                             retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Create an enhanced prompt using retrieved documentation.
        
        Args:
            user_query: Original user query
            retrieved_docs: Retrieved relevant documents
            
        Returns:
            Enhanced prompt with context
        """
        # Start with context from documentation
        context_parts = []
        
        for doc in retrieved_docs:
            title = doc['metadata'].get('title', 'Unknown')
            content = doc['content']
            doc_type = doc['metadata'].get('type', 'content')
            
            if doc_type == 'code':
                context_parts.append(f"Code Example from '{title}':\n{content}\n")
            else:
                context_parts.append(f"Documentation from '{title}':\n{content}\n")
        
        context = "\n".join(context_parts)
        
        # Create the enhanced prompt
        enhanced_prompt = f"""# Manim Documentation Context:
{context}

# User Request:
{user_query}

# Instructions:
Based on the above Manim documentation and examples, generate Python code using the Manim library that fulfills the user's request. 
The code should:
1. Follow Manim best practices shown in the documentation
2. Use appropriate Manim classes and methods
3. Be complete and runnable
4. Include proper imports and scene structure

### Manim Code:
"""
        return enhanced_prompt
    
    def generate_manim_code(self, user_query: str, max_new_tokens: int = 500,temperature: float = 0.7,top_p: float = 0.9) -> Dict[str, Any]:

        """
        Generate Manim code using RAG pipeline.
        
        Args:
            user_query: User's natural language query
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Dictionary containing generated code and metadata
        """
        
        if not self.pipe:
            return {
                'success': False,
                'error': 'Model not loaded',
                'code': '',
                'retrieved_docs': []
            }
        
        try:
            # Step 1: Retrieve relevant documentation
            logger.info("Retrieving relevant documentation...")
            retrieved_docs = self.retrieve_relevant_docs(user_query, n_results=5)
            
            # Step 2: Create enhanced prompt
            enhanced_prompt = self.create_enhanced_prompt(user_query, retrieved_docs)
            
            # Step 3: Generate code using the fine-tuned model
            logger.info("Generating Manim code...")
            output = self.pipe(
                enhanced_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=self.pipe.tokenizer.eos_token_id
            )
            
            # Step 4: Extract generated code
            generated = output[0]["generated_text"]
            
            # Find the generated code (after "### Manim Code:")
            if "### Manim Code:" in generated:
                manim_code = generated.split("### Manim Code:")[-1].strip()
            else:
                manim_code = generated[len(enhanced_prompt):].strip()
            
            return {
                'success': True,
                'code': manim_code,
                'retrieved_docs': retrieved_docs,
                'enhanced_prompt': enhanced_prompt,
                'user_query': user_query
            }
            
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return {
                'success': False,
                'error': str(e),
                'code': '',
                'retrieved_docs': []
            }
    
    def setup_knowledge_base(self, max_docs: int = 30):
        """
        Set up the knowledge base by processing Manim documentation.
        
        Args:
            max_docs: Maximum number of documents to process
        """
        logger.info("Setting up knowledge base...")
        
        # Check if documents are already processed
        processor = ManimeDocumentProcessor()
        docs = processor.load_processed_docs()
        
        if not docs:
            logger.info("Processing Manim documentation...")
            docs = processor.process_documentation(max_docs=max_docs)
            processor.save_processed_docs()
        else:
            logger.info(f"Loaded {len(docs)} existing documents")
        
        # Check if vector store is populated
        stats = self.vector_store.get_collection_stats()
        if stats['total_chunks'] == 0:
            logger.info("Populating vector store...")
            self.vector_store.add_documents(docs)
        else:
            logger.info(f"Vector store already contains {stats['total_chunks']} chunks")
        
        logger.info("Knowledge base setup complete")

def test_rag_pipeline():
    """Test the RAG pipeline with sample queries."""
    rag = Math2VisualRAG()
    
    # Setup knowledge base if needed
    rag.setup_knowledge_base(max_docs=10)  # Reduced for testing
    
    # Test queries
    test_queries = [
        "Create a sine wave animation",
        "Draw a circle that transforms into a square",
        "Animate the graph of y = x^2",
        "Show the Pythagorean theorem with animation"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        result = rag.generate_manim_code(query, max_new_tokens=300)
        
        if result['success']:
            print(f"Generated Code:\n{result['code']}")
            print(f"\nRetrieved {len(result['retrieved_docs'])} relevant documents")
        else:
            print(f"Error: {result['error']}")

if __name__ == "__main__":
    test_rag_pipeline()
