# This combines document retrieval with the finetuned Mistral model

import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGWithFinetunedModel:
    """
    Combines RAG with the finetuned Mistral model.
    
    Flow:
    1. User provides a prompt (like "Create a circle animation")
    2. RAG retrieves relevant Manim documentation
    3. The prompt + documentation context is sent to the finetuned model
    4. The model generates accurate Manim code
    """
    
    def __init__(self, doc_path=None):
        """
        Initialize the RAG + Finetuned Model pipeline.
        
        doc_path: Path to the Manim documentation file.
                 If None, uses default at crawler/markdown_output.md
        """
        self.rag_pipeline = None
        self.llm_client = None
        self.is_initialized = False
        
        # Set default doc path
        if doc_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            doc_path = os.path.join(project_root, "crawler", "markdown_output.md")
        
        self.doc_path = doc_path
        logger.info(f"RAGWithFinetunedModel created. Doc path: {doc_path}")
    
    def initialize(self):
        """
        Initialize both RAG and the finetuned model.
        Call this before generating code.
        """
        if self.is_initialized:
            return True
        
        try:
            # Import and set up RAG pipeline
            from src.rag.rag_pipeline import RAGPipeline
            
            logger.info("Initializing RAG pipeline...")
            self.rag_pipeline = RAGPipeline(
                chunk_size=800,
                chunk_overlap=150,
                max_context_tokens=2000,
                top_k_chunks=3
            )
            
            # Load and index the documentation
            if os.path.exists(self.doc_path):
                success = self.rag_pipeline.load_and_index(self.doc_path)
                if success:
                    logger.info("RAG pipeline indexed successfully")
                else:
                    logger.warning("RAG indexing failed - will proceed without context")
            else:
                logger.warning(f"Doc file not found: {self.doc_path}")
                logger.warning("Will proceed without RAG context")
            
            # Import and set up the finetuned model client
            from src.llm.finetuned_client import get_finetuned_client
            
            logger.info("Getting finetuned model client...")
            self.llm_client = get_finetuned_client()
            
            self.is_initialized = True
            logger.info("RAG + Finetuned Model pipeline initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False
    
    def generate_manim_code(self, user_prompt, use_rag=True):
        """
        Generate Manim code for the given prompt.
        
        user_prompt: What the user wants to visualize
        use_rag: Whether to use RAG for context (default True)
        
        Returns: Generated Manim code as a string
        """
        # Make sure we are initialized
        if not self.is_initialized:
            success = self.initialize()
            if not success:
                return "# Error: Failed to initialize the pipeline"
        
        logger.info(f"Generating code for: {user_prompt[:100]}...")
        
        # Step 1: Get relevant context from RAG
        context = ""
        if use_rag and self.rag_pipeline and self.rag_pipeline.is_indexed:
            logger.info("Retrieving relevant documentation...")
            context = self.rag_pipeline.retrieve_context(user_prompt)
            
            if context:
                logger.info(f"Retrieved context: {len(context)} characters")
            else:
                logger.info("No relevant context found")
        
        # Step 2: Build the augmented prompt
        augmented_prompt = self._build_augmented_prompt(user_prompt, context)
        
        logger.info(f"Augmented prompt length: {len(augmented_prompt)} characters")
        
        # Step 3: Generate code with the finetuned model
        code = self.llm_client.generate(
            augmented_prompt,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9
        )
        
        return code
    
    def _build_augmented_prompt(self, user_prompt, context):
        """
        Build the final prompt that combines user request with RAG context.
        """
        layout_rules = """
CRITICAL LAYOUT RULES TO PREVENT TEXT OVERLAP AND OVERFLOW:

1. TEXT SIZING (MANDATORY):
   - Title: .scale(0.6) at top with .to_edge(UP, buff=0.3)
   - Main text: .scale(0.5) MAXIMUM
   - Math equations: .scale(0.6) for main, .scale(0.4) for secondary
   - Labels: .scale(0.4)

2. LINE LENGTH: Max 40 characters per line. Use "\\n" for longer text.

3. FRAME BOUNDARIES (safe zone):
   - Horizontal: -6 to +6
   - Vertical: -3.5 to +3.5
   
4. PREVENT OVERLAP: ALWAYS FadeOut previous content BEFORE showing new content.

5. PATTERN:
   self.play(FadeOut(old_content))  # Remove old first
   new_content = Text("...").scale(0.5)  # Create scaled
   self.play(Write(new_content))  # Show new
"""
        
        if not context:
            # No context available, just use the user prompt
            return f"""Create Manim code for the following request. Write clear, readable code.
{layout_rules}
Request: {user_prompt}"""
        
        # Build prompt with RAG context
        prompt = f"""Create Manim code for the following request.
Use only standard keyboard characters. Write clear, readable code.
Use the documentation below as reference for correct Manim syntax.
{layout_rules}
=== MANIM DOCUMENTATION REFERENCE ===
{context}
=== END DOCUMENTATION ===

User Request: {user_prompt}

Generate clean Manim code that implements this request. Include helpful comments."""
        
        return prompt
    
    def get_status(self):
        """
        Get the current status of the pipeline.
        """
        status = {
            "initialized": self.is_initialized,
            "doc_path": self.doc_path,
            "doc_exists": os.path.exists(self.doc_path) if self.doc_path else False
        }
        
        if self.rag_pipeline:
            status["rag"] = self.rag_pipeline.get_stats()
        else:
            status["rag"] = "not loaded"
        
        if self.llm_client:
            status["model"] = self.llm_client.get_status()
        else:
            status["model"] = "not loaded"
        
        return status


# Singleton instance for caching across Streamlit reruns
_cached_pipeline = None


def get_rag_finetuned_pipeline():
    """
    Get a cached instance of the RAG + Finetuned Model pipeline.
    This is used by the Streamlit UI to avoid reloading on every interaction.
    """
    global _cached_pipeline
    if _cached_pipeline is None:
        _cached_pipeline = RAGWithFinetunedModel()
    return _cached_pipeline
