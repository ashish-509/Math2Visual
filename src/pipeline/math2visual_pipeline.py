# Math2Visual Pipeline
# Converts spoken math to LaTeX, builds a prompt, and gets Manim code from an LLM.

from src.math_parser.parser import spoken_math_to_latex
from src.llm.client import LLMClient, LLMClientError
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Math2VisualPipeline:
    def __init__(self, llm_client=None):
        # Use provided LLM client or make a new one
        self.llm = llm_client or LLMClient()

    def build_prompt(self, raw_text, latex):
        # Create a prompt for the LLM to generate Manim code
        prompt = (
            "Task: Generate a Manim Community-compatible Python script that visualizes the following request.\n\n"
            f"User speech: \"{raw_text}\"\n\n"
        )
        if latex:
            prompt += f"Interpreted math (LaTeX-like): `{latex}`\n\n"
        prompt += (
            "Constraints:\n"
            "- Output only the Python file contents.\n"
            "- Keep code simple and well-commented.\n"
            "- Avoid external dependencies other than manim.\n"
        )
        return prompt

    def generate_code(self, raw_text):
        # Convert speech to LaTeX, build prompt, and get code from LLM
        latex = spoken_math_to_latex(raw_text)
        prompt = self.build_prompt(raw_text, latex)
        try:
            code = self.llm.generate(prompt)
            return code, True, ""
        except LLMClientError as e:
            logger.error(f"LLM generation failed: {e}")
            # Fallback code if LLM fails
            fallback = (
                "# Fallback Manim snippet\n"
                "from manim import *\n\n"
                "class FallbackScene(Scene):\n"
                "    def construct(self):\n"
                "        self.play(Write(Text('Failed to generate code - LLM error')))\n"
                "        self.wait(1)\n"
            )
            return fallback, False, str(e)
