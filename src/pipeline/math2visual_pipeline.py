# Math2Visual Pipeline
# Converts spoken math to LaTeX, builds a prompt, and gets Manim code from an LLM.
# Now with RAG support for retrieving relevant Manim documentation

import sys
import os

# handle the math-parser directory name with dash
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from importlib import import_module
math_parser = import_module('math-parser.parser', package='src')
spoken_math_to_latex = math_parser.spoken_math_to_latex

from src.llm.client import LLMClient, LLMClientError
from src.rag.rag_pipeline import create_rag_pipeline
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Math2VisualPipeline:
    def __init__(self, llm_client=None, use_rag=True, doc_path=None, preferred_model='mistral'):
        """
        Initialize the Math2Visual pipeline
        
        llm_client: optional custom LLM client (if None, creates one with multi-model support)
        use_rag: whether to use RAG for documentation retrieval
        doc_path: path to documentation file
        preferred_model: which LLM to prefer ('mistral', 'codellama', or 'phi2')
        """
        # setup LLM client with multi-model support
        if llm_client:
            self.llm = llm_client
        else:
            # create client with automatic fallback between models
            self.llm = LLMClient(preferred_model=preferred_model)
        
        # RAG setup
        self.use_rag = use_rag
        self.rag = None
        
        if use_rag:
            try:
                logger.info("Initializing RAG system...")
                self.rag = create_rag_pipeline(doc_path)
                if self.rag.is_indexed:
                    logger.info("RAG enabled and ready")
                else:
                    logger.warning("RAG initialization incomplete - continuing without docs")
                    self.use_rag = False
            except Exception as e:
                logger.warning(f"Could not initialize RAG: {e}")
                self.use_rag = False

    def build_prompt(self, raw_text, latex, context=""):
        # Create a prompt for the LLM to generate Manim code
        # Now includes retrieved documentation context if available
        
        prompt = ""
        
        # add retrieved documentation context first
        if context:
            prompt += "RELEVANT MANIM DOCUMENTATION\n"
            prompt += context
            prompt += "\nEND DOCUMENTATION\n\n"
        
        prompt += (
            "Task: Generate a Manim Community-compatible Python script that visualizes the following request.\n\n"
            f"User speech: \"{raw_text}\"\n\n"
        )
        if latex:
            prompt += f"Interpreted math (LaTeX-like): `{latex}`\n\n"
        
        prompt += "Constraints:\n"
        if context:
            prompt += "- Use the documentation above to write accurate, up-to-date Manim code.\n"
        prompt += (
            "- Output only the Python file contents.\n"
            "- Keep code simple and well-commented.\n"
            "- Avoid external dependencies other than manim.\n"
        )
        return prompt

    def generate_code(self, raw_text):
        # Convert speech to LaTeX, retrieve docs, build prompt, and get code from LLM
        latex = spoken_math_to_latex(raw_text)
        
        # build prompt with RAG context if enabled
        if self.use_rag and self.rag:
            try:
                logger.info("Retrieving relevant documentation...")
                # use RAG's optimized prompt building with token limits
                base_prompt = (
                    "Task: Generate a Manim Community-compatible Python script that visualizes the following request.\n\n"
                    f"User speech: \"{raw_text}\"\n\n"
                )
                if latex:
                    base_prompt += f"Interpreted math (LaTeX-like): `{latex}`\n\n"
                base_prompt += (
                    "Constraints:\n"
                    "- Output only the Python file contents.\n"
                    "- Keep code simple and well-commented.\n"
                    "- Avoid external dependencies other than manim.\n"
                )
                
                prompt = self.rag.augment_prompt(raw_text, base_prompt, max_total_tokens=4000)
                
                if len(prompt) > len(base_prompt):  # context was added
                    logger.info(f"Added context to prompt ({len(prompt)} chars total)")
                else:
                    logger.info("No relevant docs found, proceeding without context")
                    
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")
                # fallback to basic prompt
                prompt = self.build_prompt(raw_text, latex, "")
        else:
            # no RAG, use basic prompt
            prompt = self.build_prompt(raw_text, latex, "")
        
        try:
            code = self.llm.generate(prompt)
            return code, True, ""
        except LLMClientError as e:
            logger.error(f"LLM generation failed: {e}")
            
            # Fallback code if LLM fails - provides user feedback via Manim scene
            fallback = (
                "# Fallback Manim snippet\n"
                "from manim import *\n\n"
                "class FallbackScene(Scene):\n"
                "    def construct(self):\n"
                "        self.play(Write(Text('Failed to generate code - LLM error')))\n"
                "        self.wait(1)\n"
            )
            return fallback, False, str(e)
    
    def get_rag_stats(self):
        """Get RAG pipeline statistics if available"""
        if self.rag:
            return self.rag.get_stats()
        return {"status": "disabled"}
    
    def get_model_status(self):
        """
        Get health status of all available LLM models
        Useful for debugging which models are working
        """
        return self.llm.get_model_status()
    
    def switch_model(self, model_name):
        """
        Switch preferred model (mistral, codellama, or phi2)
        The pipeline will try this model first on next generation
        """
        if model_name in self.llm.models:
            self.llm.preferred_model = model_name
            logger.info(f"Switched preferred model to: {model_name}")
            return True
        else:
            logger.warning(f"Model '{model_name}' not available")
            return False
