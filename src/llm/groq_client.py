# Groq API client for CodeLlama and other models
# Groq provides fast inference for Llama models via API

import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroqClient:
    """
    Client for Groq API to generate Manim code using Llama models.
    Supports different models for different use cases.
    """
    
    # Available models on Groq
    MODELS = {
        "codellama": "llama-3.3-70b-versatile",  # Best for code generation
        "phi2": "llama-3.1-8b-instant"           # Faster, lighter model
    }
    
    def __init__(self, api_key=None, model_type="codellama"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        
        # Get model from env or use default based on type
        if model_type == "codellama":
            self.model = os.getenv("GROQ_MODEL_CODELLAMA", self.MODELS["codellama"])
        elif model_type == "phi2":
            self.model = os.getenv("GROQ_MODEL_PHI2", self.MODELS["phi2"])
        else:
            self.model = os.getenv("GROQ_MODEL_CODELLAMA", self.MODELS["codellama"])
        
        self.model_type = model_type
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Add it to .env file")
        
        # Import groq library (install with: pip install groq)
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Groq client initialized with model: {self.model} ({model_type})")
        except ImportError:
            raise ImportError("groq library not installed. Run: pip install groq")
    
    def generate(self, prompt, max_tokens=2048, temperature=0.7):
        """
        Generate Manim code using Groq API.
        
        prompt: The user's request (optionally with RAG context)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0-1)
        
        Returns: Generated code as string
        """
        try:
            # Build a system prompt for better code generation
            system_prompt = """You are an expert Manim code generator. Generate clean, valid Python code using the Manim Community Edition library.

IMPORTANT - Use these CORRECT modern Manim methods (NOT deprecated ones):
- Use Create() instead of ShowCreation() 
- Use Uncreate() instead of ShowDestruction()
- Use FadeIn() and FadeOut() for fading
- Use Transform() for morphing between objects
- Use ReplacementTransform() to replace one object with another
- Use Write() for text and equations
- Use DrawBorderThenFill() for shapes with fill
- Use GrowFromCenter() to grow objects
- Use MoveToTarget() with .generate_target() for movement
- Use Indicate() to highlight objects
- Use Circumscribe() to draw attention to objects
- Use self.play() to animate
- Use self.wait() to pause
- Use self.add() to add without animation

For math equations:
- Use MathTex(r"...") for LaTeX math (use raw strings with r prefix)
- Use Tex(r"...") for LaTeX text
- Use Text("...") for plain text (no special characters)
- Escape backslashes properly in LaTeX strings

Rules:
1. Always start with: from manim import *
2. Use standard ASCII characters only in Text() - no unicode symbols like pi, use words instead
3. Create a class that inherits from Scene
4. Implement the construct(self) method
5. Include helpful comments explaining each step
6. Never use deprecated methods like ShowCreation, ShowDestruction
7. Make code beginner-friendly and readable
8. Ensure the code is complete and runnable
9. For pi symbol in math, use MathTex(r"\\pi") not Text()

Output only the Python code, no explanations or markdown."""

            logger.info(f"Generating code with Groq ({self.model})...")
            
            # Call Groq API
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Generate Manim code for:\n{prompt}"
                    }
                ],
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )
            
            # Extract the generated code
            code = chat_completion.choices[0].message.content
            
            # Clean up the code (remove markdown fences if present)
            code = self._clean_code(code)
            
            logger.info("Code generation complete")
            return code
            
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return f"# Error generating code: {str(e)}"
    
    def _clean_code(self, code):
        # Remove markdown code fences
        if "```python" in code:
            code = code.split("```python")[1]
            if "```" in code:
                code = code.split("```")[0]
        elif "```" in code:
            parts = code.split("```")
            if len(parts) >= 2:
                code = parts[1]
        
        # Ensure imports are present
        code = code.strip()
        if not code.startswith("from manim import") and not code.startswith("import"):
            code = "from manim import *\n\n" + code
        
        # Fix deprecated Manim API calls
        code = self._fix_deprecated_manim_calls(code)
        
        return code
    
    def _fix_deprecated_manim_calls(self, code):
        """
        Replace deprecated Manim API calls with modern equivalents.
        """
        import re
        
        # Dictionary of deprecated -> modern replacements
        replacements = {
            r'\bShowCreation\b': 'Create',
            r'\bShowDestruction\b': 'Uncreate',
            r'\bShowPassingFlash\b': 'ShowPassingFlash',
            r'\bShowSubmobjectsOneByOne\b': 'ShowSubmobjectsOneByOne',
            r'\bShowIncreasingSubsets\b': 'ShowIncreasingSubsets',
            r'\bApplyWave\b': 'ApplyWave',
            r'\bWiggleOutThenIn\b': 'Wiggle',
            r'\bTurnInsideOut\b': 'TurnInsideOut',
        }
        
        for old_pattern, new_name in replacements.items():
            code = re.sub(old_pattern, new_name, code)
        
        # Fix unicode characters in Text() calls - replace with ASCII or MathTex
        # Replace pi symbol with the word "pi" in Text()
        code = code.replace('Text("π', 'Text("pi')
        code = code.replace("Text('π", "Text('pi")
        code = code.replace('"\u03c0"', '"pi"')
        code = code.replace("'\u03c0'", "'pi'")
        
        return code
    
    def get_status(self):
        """
        Get the current status of the Groq client.
        """
        return {
            "api_key_set": bool(self.api_key),
            "model": self.model,
            "model_type": self.model_type,
            "service": "Groq API"
        }


# Cache for different model clients
_groq_clients = {}


def get_groq_client(model_type="codellama"):
    """
    Get a cached instance of the Groq client for the specified model type.
    
    model_type: 'codellama' or 'phi2'
    """
    global _groq_clients
    if model_type not in _groq_clients:
        _groq_clients[model_type] = GroqClient(model_type=model_type)
    return _groq_clients[model_type]
