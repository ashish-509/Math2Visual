import os
import logging
from typing import Optional, Dict, List
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ManimSyntaxChatbot:
    """
    Attributes:
        rag_pipeline: The RAG pipeline for document retrieval
        llm_client: The LLM client for generating responses
        is_ready: Whether the chatbot is initialized and ready
    """
    
    # CONFIGURATION
    
    # Path to Manim documentation
    DEFAULT_DOCS_PATH = "crawler/markdown_output.md"
    
    # RAG settings for optimal performance
    RAG_SETTINGS = {
        "chunk_size": 600,       # Smaller chunks = faster retrieval
        "chunk_overlap": 100,    # Overlap to preserve context
        "top_k_chunks": 3,       # Number of chunks to retrieve 
        "max_context_tokens": 1500  # Limit context size for speed
    }
    
    # LLM settings for fast responses
    LLM_SETTINGS = {
        "max_tokens": 512,       # Shorter responses = faster
        "temperature": 0.3,      # Lower = more focused answers
    }
    
    # System prompt for syntax-focused answers
    SYSTEM_PROMPT = """You are a helpful Manim syntax assistant. Your job is to:
1. Answer questions about Manim syntax clearly and concisely
2. Provide working code examples when relevant
3. Use the CORRECT modern Manim methods (not deprecated ones)
4. Keep explanations simple and beginner-friendly

IMPORTANT RULES:
- Use Create() not ShowCreation()
- Use MathTex() for math formulas, Text() for plain text
- Always include necessary imports in examples
- If unsure, say so honestly

Format your responses with:
- A brief explanation first
- Then a code example if applicable
- Keep it short and practical"""

    # INITIALIZATION
    
    def __init__(self, docs_path: Optional[str] = None):
        
        self.docs_path = docs_path
        self.rag_pipeline = None
        self.llm_client = None
        self.is_ready = False
        
        # Response cache for repeated questions (speeds up common queries)
        self._response_cache: Dict[str, Dict] = {}
        self._cache_max_size = 50  # Keep last 50 responses
        
        logger.info("ManimSyntaxChatbot created. Call initialize() to set up.")
    
    def initialize(self) -> bool:
        # Set up the chatbot (load docs, create index, connect to LLM).

        logger.info("Initializing ManimSyntaxChatbot...")
        
        # Step 1: Set up RAG pipeline
        if not self._setup_rag():
            logger.error("Failed to set up RAG pipeline")
            return False
        
        # Step 2: Set up LLM client
        if not self._setup_llm():
            logger.error("Failed to set up LLM client")
            return False
        
        self.is_ready = True
        logger.info(" ManimSyntaxChatbot ready!")
        return True
    
    def _setup_rag(self) -> bool:
        try:
            # Import here to avoid circular imports
            from src.rag.rag_pipeline import RAGPipeline
            
            # Create RAG pipeline with optimized settings
            self.rag_pipeline = RAGPipeline(
                chunk_size=self.RAG_SETTINGS["chunk_size"],
                chunk_overlap=self.RAG_SETTINGS["chunk_overlap"],
                max_context_tokens=self.RAG_SETTINGS["max_context_tokens"],
                top_k_chunks=self.RAG_SETTINGS["top_k_chunks"]
            )
            
            # Find the docs file
            docs_path = self._find_docs_path()
            if not docs_path:
                logger.error("Could not find Manim documentation file")
                return False
            
            # Load and index the documentation
            logger.info(f"Loading docs from: {docs_path}")
            success = self.rag_pipeline.load_and_index(docs_path)
            
            if success:
                logger.info(" RAG pipeline ready")
            return success
            
        except ImportError as e:
            logger.error(f"Could not import RAG modules: {e}")
            return False
        except Exception as e:
            logger.error(f"RAG setup error: {e}")
            return False
    
    def _setup_llm(self) -> bool:
        try:
            # Import here to avoid circular imports
            from src.llm.groq_client import GroqClient
            
            # Use the faster model for quick responses
            self.llm_client = GroqClient(model_type="phi2")
            logger.info(" LLM client ready (using fast model)")
            return True
            
        except ImportError as e:
            logger.error(f"Could not import LLM modules: {e}")
            return False
        except Exception as e:
            logger.error(f"LLM setup error: {e}")
            return False
    
    def _find_docs_path(self) -> Optional[str]:
        """Find the Manim documentation file."""
        # If custom path provided, use it
        if self.docs_path and os.path.exists(self.docs_path):
            return self.docs_path
        
        # Try to find project root and default docs
        possible_roots = [
            os.getcwd(),
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ]
        
        for root in possible_roots:
            docs_path = os.path.join(root, self.DEFAULT_DOCS_PATH)
            if os.path.exists(docs_path):
                return docs_path
        
        return None

    # MAIN API 
    
    def ask(self, question: str) -> Dict:
        
        # Check if chatbot is ready
        if not self.is_ready:
            return {
                "answer": "Chatbot not initialized. Call initialize() first.",
                "sources": "",
                "cached": False,
                "error": True
            }
        
        # Check cache for repeated questions
        cache_key = question.strip().lower()
        if cache_key in self._response_cache:
            logger.info("Returning cached response")
            cached = self._response_cache[cache_key].copy()
            cached["cached"] = True
            return cached
        
        # Generate new response
        response = self._generate_response(question)
        
        # Cache the response (if successful)
        if not response.get("error"):
            self._cache_response(cache_key, response)
        
        return response
    
    def ask_with_examples(self, question: str) -> Dict:
        # Similar to ask(), but explicitly requests code examples in the response.
      
        # Append request for examples
        enhanced_question = f"{question}\n\nPlease include a simple code example."
        return self.ask(enhanced_question)
    
    def get_syntax(self, manim_class: str) -> Dict:
        
        question = f"What is the syntax for {manim_class} in Manim? Show constructor parameters and a simple example."
        return self.ask(question)
    
    def list_animations(self) -> Dict:
        
        question = "List the most common animation methods in Manim with brief descriptions."
        return self.ask(question)
    
    def clear_cache(self):
        self._response_cache.clear()
        logger.info("Response cache cleared")


    # INTERNAL METHODS
    
    def _generate_response(self, question: str) -> Dict:
        # Generate a response using RAG + LLM.
        try:
            # Step 1: Retrieve relevant context from docs
            context = self.rag_pipeline.retrieve_context(question)
            
            # Step 2: Build the prompt
            prompt = self._build_prompt(question, context)
            
            # Step 3: Generate response with LLM
            answer = self.llm_client.generate(
                prompt=prompt,
                max_tokens=self.LLM_SETTINGS["max_tokens"],
                temperature=self.LLM_SETTINGS["temperature"]
            )
            
            return {
                "answer": answer,
                "sources": context[:500] if context else "No specific docs found",
                "cached": False,
                "error": False
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "sources": "",
                "cached": False,
                "error": True
            }
    
    def _build_prompt(self, question: str, context: str) -> str:
        prompt = f"""{self.SYSTEM_PROMPT}

=== Relevant Manim Documentation ===
{context if context else "No specific documentation found for this query."}

=== User Question ===
{question}

=== Your Answer ===
"""
        return prompt
    
    def _cache_response(self, key: str, response: Dict):
        """Cache a response (with size limit)."""
        # Remove oldest entries if cache is full
        if len(self._response_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._response_cache))
            del self._response_cache[oldest_key]
        
        self._response_cache[key] = response


# CONVENIENCE FUNCTION 

# Cached singleton instance
_chatbot_instance: Optional[ManimSyntaxChatbot] = None


def get_manim_chatbot() -> ManimSyntaxChatbot:
    # This function creates and initializes the chatbot once, then returns the same instance on subsequent calls.
    
    global _chatbot_instance
    
    if _chatbot_instance is None:
        _chatbot_instance = ManimSyntaxChatbot()
        _chatbot_instance.initialize()
    
    return _chatbot_instance


# QUICK TEST

if __name__ == "__main__":
    # Quick test when running this file directly
    print("Testing ManimSyntaxChatbot...")
    
    chatbot = ManimSyntaxChatbot()
    
    if chatbot.initialize():
        print("\nChatbot initialized successfully!\n")
        
        # Test question
        test_question = "How do I create and animate a circle in Manim?"
        print(f"Question: {test_question}\n")
        
        response = chatbot.ask(test_question)
        print(f"Answer:\n{response['answer']}")
    else:
        print("Failed to initialize chatbot")
