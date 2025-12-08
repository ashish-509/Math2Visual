from src.math_parser.parser import spoken_math_to_latex
from src.llm.client import LLMClient, LLMClientError
from typing import Tuple
import logging
logger = logging.getLogger(__name__)

class Math2VisualPipeline:
    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient()
    
    def build_prompt(self, raw_text: str, latex: str) -> str:
        prompt = "Task: Generate a Manim Community-compatible Python script that visualizes the following request.\n\n"
        prompt += f"User speech: \"{raw_text}\"\n\n"
        if latex:
            prompt += f"Interpreted math (LaTeX-like): `{latex}`\n\n"
        prompt += ("Constraints:\n- Output only the Python file contents.\n- Keep code simple and well-commented.\n- Avoid external dependencies other than manim.\n")
        return prompt

    def generate_code(self, raw_text: str) -> Tuple[str, bool, str]:
        latex = spoken_math_to_latex(raw_text)
        prompt = self.build_prompt(raw_text, latex)
        try:
            code = self.llm.generate(prompt)
            return code, True, ""
        except LLMClientError as e:
            logger.exception("LLM generation failed")
            # return a simple fallback code snippet so the UI can still show something
            fallback = f"""# Fallback Manim snippet\nfrom manim import *\n\nclass FallbackScene(Scene):\n    def construct(self):\n        self.play(Write(Text('Failed to generate code - LLM error'))) \n        self.wait(1)\n"""
            return fallback, False, str(e)
