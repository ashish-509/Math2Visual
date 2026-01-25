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
- Use MathTex(r"...") for ANY formula with math symbols (=, +, -, ×, fractions, greek letters, etc.)
- Use Text("...") for plain text labels (no LaTeX, no special characters)
- NEVER use Tex() for formulas - always use MathTex() for math
- For multiplication use MathTex(r"a \\times b") NOT Tex()
- Escape backslashes properly: \\times, \\frac, \\pi, \\sqrt

CRITICAL LAYOUT RULES - ZERO OVERLAP ALLOWED:
1. SEQUENCE IS CRITICAL - NEVER show new content while old content is still in center:
   a) FIRST: Move old content away with self.play(old.animate.scale(0.25).to_corner(UL))
   b) THEN: Show new content in center with self.play(Create(new)) or self.play(Write(new))
   c) NEVER animate both in the same self.play() call

2. POSITIONING OLD CONTENT (use these exact positions):
   - 1st old item: to_corner(UL) with buff=0.3
   - 2nd old item: move_to(LEFT*6)
   - 3rd old item: to_corner(DL) with buff=0.3
   - If more than 3 old items: FadeOut the oldest ones

3. CENTER ZONE IS SACRED:
   - New content appears at ORIGIN or with small shifts (UP*0.5, DOWN*0.5)
   - Scale new content to 0.7 so it fits comfortably
   - Nothing else should be in the center region (LEFT*3 to RIGHT*3, UP*2 to DOWN*2)

4. CODE PATTERN FOR EACH STEP:
   ```
   # Step N: First move old content away
   self.play(old_content.animate.scale(0.25).to_corner(UL))
   
   # Now center is clear - show new content
   new_content = MathTex(r"...").scale(0.7)
   self.play(Write(new_content))
   self.wait(1)
   ```

5. GROUPING: Use VGroup() to move related items together as one unit

6. TITLE: Keep title at top edge, scale 0.6, never move it

7. TIMING: self.wait(1) after each major animation for narration

8. TEXT LENGTH: Max 30 characters per line, use Text("Line1\nLine2") for longer

Rules:
1. Always start with: from manim import *
2. Use standard ASCII characters only in Text() - no unicode symbols like pi
3. Create a class that inherits from Scene
4. Implement the construct(self) method
5. Include helpful comments explaining each step
6. Never use deprecated methods like ShowCreation, ShowDestruction
7. Make code beginner-friendly and readable
8. Ensure the code is complete and runnable
9. For pi symbol in math, use MathTex(r"\\pi") not Text()
10. Total animation should be 15-30 seconds with appropriate wait times
11. NEVER use self.play(self.add(...)) - self.add() adds without animation, self.play() animates
12. Use self.add(obj) for instant appearance, self.play(FadeIn(obj)) or self.play(Write(obj)) for animated appearance
13. self.play() takes Animation objects like Create(), Write(), FadeIn(), Transform() - NOT self.add()

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
