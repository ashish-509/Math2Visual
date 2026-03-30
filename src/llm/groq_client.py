# Groq API client for CodeLlama and other models
# Groq provides fast inference for Llama models via API

import os
import re
import ast
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
            system_prompt = """You are an expert Manim Community Edition code generator. Your job is to create PERFECT, ERROR-FREE animations.

======================================================================================
                              CRITICAL OUTPUT RULES
======================================================================================

1. Output ONLY valid Python code - no explanations, no markdown, no "Here is the code"
2. First line MUST be: from manim import *
3. Create exactly ONE class inheriting from Scene
4. The class MUST have a construct(self) method
5. Generate COMPLETE code - never truncate or use "..." or comments like "continue..."

======================================================================================
                     !!!! ABSOLUTE BAN ON LATEX OBJECTS !!!!
======================================================================================

*** NEVER use MathTex(), Tex(), or any LaTeX-based object. ***
LaTeX is NOT installed. Any MathTex/Tex call WILL crash the render.

INSTEAD of MathTex/Tex, ALWAYS use Text():
- Text("y = x²").scale(0.32)
- Text("f(x) = x² + 2x + 1").scale(0.32)
- Text("Area = pi × r²").scale(0.32)
- Text("dy/dx = 2x").scale(0.32)
- Text("a² + b² = c²").scale(0.32)
- Text("integral of f(x) dx").scale(0.32)

Write math in plain readable form using Unicode symbols where possible:
  ² ³ × ÷ ≤ ≥ ≠ ≈ → π θ α β Σ ∫ √ ∞

======================================================================================
                           LAYOUT RULES (PREVENT OVERLAP)
======================================================================================

SAFE ZONE: x=[-5.5, 5.5], y=[-3.2, 3.2] - content outside gets cut off

TEXT SIZING (STRICT):
- Title: .scale(0.4).to_edge(UP, buff=0.3)
- Body/Math: .scale(0.3)
- Labels on shapes: .scale(0.22)
- Step numbers/headers: .scale(0.28)
- Max 30 chars per line, use \\n to break longer text
- NEVER use .scale() above 0.45 for ANY text

!!! ABSOLUTE RULE — PREVENT TEXT OVERLAP !!!
- BEFORE showing any new text/content, ALWAYS FadeOut ALL previous text first
- Use self.play(*[FadeOut(mob) for mob in self.mobjects]) to clear the screen between sections
- Maximum 3-4 text elements visible at the SAME TIME on screen
- If you have a list of steps, show ONE or TWO steps at a time, then clear and show next
- NEVER stack more than 3 lines of text vertically without clearing previous ones
- Use .move_to() or .to_edge() to position text — NEVER let text auto-stack
- Keep generous spacing: buff=0.4 minimum between text elements
- When showing steps, use a "slide" approach: show step, wait, fade out, show next step

======================================================================================
              STEP-BY-STEP PROBLEM SOLVING (CRITICAL FOR MATH PROBLEMS)
======================================================================================

When solving a math problem, ALWAYS follow this slide-based pattern:
1. Show the problem statement alone first (title + problem text)
2. FadeOut everything, then show Step 1 with its work
3. FadeOut Step 1, then show Step 2 with its work
4. Continue: FadeOut previous step, show next step
5. FadeOut last step, show the final answer/result
6. End with cleanup

Each "slide" should have:
- A step header: Text("Step 1: ...").scale(0.28).to_edge(UP, buff=0.3)
- The math work: Text("...").scale(0.3).move_to(ORIGIN)
- A brief self.wait(1.5) for reading time
- Then FadeOut EVERYTHING before the next step

NEVER show all steps at once. NEVER stack steps below each other.
Treat each step as a separate slide/screen.

ALWAYS END WITH:
    self.wait(2)
    self.play(*[FadeOut(mob) for mob in self.mobjects])

======================================================================================
                            PYTHON SYNTAX RULES
======================================================================================

*** LIST SYNTAX - MUST BE CORRECT: ***
CORRECT:   points = [(1, 2), (3, 4), (5, 6)]
WRONG:     points = []\\n        (1, 2),\\n    ]

*** DEPRECATED CALLS - USE MODERN API: ***
- Create() not ShowCreation()
- Uncreate() not ShowDestruction()
- FadeIn() / FadeOut() for appearing/disappearing

======================================================================================
                  COMPLETE WORKING TEMPLATE — STEP-BY-STEP SOLVING
======================================================================================

from manim import *

class SolveEquation(Scene):
    def construct(self):
        # --- Slide 1: Problem Statement ---
        title = Text("Solving 2x + 5 = 13").scale(0.4).to_edge(UP, buff=0.3)
        problem = Text("Find the value of x").scale(0.3).move_to(ORIGIN)
        self.play(Write(title), Write(problem))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(problem))

        # --- Slide 2: Step 1 ---
        step1_title = Text("Step 1: Subtract 5 from both sides").scale(0.28).to_edge(UP, buff=0.3)
        step1_work = Text("2x + 5 - 5 = 13 - 5\\n2x = 8").scale(0.3).move_to(ORIGIN)
        self.play(Write(step1_title), Write(step1_work))
        self.wait(1.5)
        self.play(FadeOut(step1_title), FadeOut(step1_work))

        # --- Slide 3: Step 2 ---
        step2_title = Text("Step 2: Divide both sides by 2").scale(0.28).to_edge(UP, buff=0.3)
        step2_work = Text("2x / 2 = 8 / 2\\nx = 4").scale(0.3).move_to(ORIGIN)
        self.play(Write(step2_title), Write(step2_work))
        self.wait(1.5)
        self.play(FadeOut(step2_title), FadeOut(step2_work))

        # --- Slide 4: Final Answer ---
        result_title = Text("Solution").scale(0.4).to_edge(UP, buff=0.3)
        result = Text("x = 4").scale(0.4).set_color(GREEN).move_to(ORIGIN)
        check = Text("Check: 2(4) + 5 = 8 + 5 = 13  ✓").scale(0.25).next_to(result, DOWN, buff=0.4)
        self.play(Write(result_title), Write(result))
        self.play(Write(check))
        self.wait(2)

        # --- Cleanup ---
        self.play(*[FadeOut(mob) for mob in self.mobjects])

======================================================================================
                  COMPLETE WORKING TEMPLATE — CONCEPT VISUALIZATION
======================================================================================

from manim import *

class ConceptVisualization(Scene):
    def construct(self):
        # 1. Title
        title = Text("Topic Title Here").scale(0.4).to_edge(UP, buff=0.3)
        self.play(Write(title))
        self.wait(1)

        # 2. Explanation (replace title first)
        self.play(FadeOut(title))
        explanation = Text("Key concept explained\\nin simple terms").scale(0.3).move_to(UP * 1)
        self.play(Write(explanation))
        self.wait(1.5)

        # 3. Visual (clear old content, show graph)
        self.play(FadeOut(explanation))
        axes = Axes(x_range=[-3,3,1], y_range=[-2,2,1], x_length=5, y_length=3)
        axes.scale(0.7).move_to(DOWN * 0.8)
        self.play(Create(axes))
        graph = axes.plot(lambda x: x**2, color=BLUE)
        self.play(Create(graph))
        self.wait(1)

        # 4. Label (use Text, NOT MathTex)
        label = Text("y = x²").scale(0.22).next_to(axes, UP, buff=0.15)
        self.play(Write(label))
        self.wait(1.5)

        # 5. Clean ending
        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])

======================================================================================

Now generate the complete Manim code. Output ONLY the Python code, starting with 'from manim import *'.
REMEMBER: NEVER use MathTex or Tex. Use Text() for ALL text including math formulas."""

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
                        "content": f"Generate complete Manim code for:\n{prompt}\n\nCRITICAL REMINDERS:\n1. Output COMPLETE code only. No truncation. No markdown. No explanations.\n2. NEVER use MathTex or Tex — use Text() for ALL text including formulas.\n3. PREVENT OVERLAP: FadeOut ALL previous content before showing new text/steps.\n4. For problem solving: show ONE step per screen, FadeOut before next step.\n5. Keep text small: .scale(0.3) for body, .scale(0.4) max for titles.\n6. Maximum 3 text elements visible at once."
                    }
                ],
                model=self.model,
                max_tokens=4096, 
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
    
    def generate_narration(self, prompt, max_tokens=1024, temperature=0.8, manim_code=None):
       
        try:
            # System prompt specifically for narration (NOT code)
            system_prompt = """You are a math teacher creating a voiceover script for an educational animation video.

YOUR TASK: Write ONLY the spoken narration text that will be read aloud.

CRITICAL RULES:
1. Output ONLY plain spoken words - no code, no formatting, no markdown
2. Start speaking IMMEDIATELY about the math concept - no introductions
3. DO NOT output any Python code, imports, class definitions, or programming syntax
4. DO NOT say "Here is the narration" or similar - just start the narration directly
5. DO NOT use bullet points, numbers, or formatting - just flowing speech
6. Sound like an enthusiastic teacher explaining to students
7. If animation code is provided, narrate EXACTLY what appears on screen — do not invent extra content

FORBIDDEN (never include):
- "from manim import" or any code
- "def", "class", "self.", "import"
- "Here is...", "Sure...", "Certainly..."
- "Hello everyone", "Welcome", "In this video"
- Markdown like ** or # or ```
- Timestamps like [0:00]
- Technical terms: speed, quality, resolution, fps, render, version, pixels, codec, bitrate
- Visual descriptions: "we see", "notice how", "as you can see", "appearing on screen"

EXAMPLE OUTPUT:
"Local minima and maxima are fascinating points on a curve. At these special locations, the function momentarily stops increasing or decreasing. Think of it like a ball rolling on a hill - at the very top, it pauses before rolling down the other side. That peak is a local maximum. Similarly, the bottom of a valley is a local minimum. Mathematically, we find these points where the derivative equals zero."

Write natural, flowing speech that teaches the concept. Be concise and clear.
When code is included, read every formula/text that appears on screen in spoken form and explain it."""

            logger.info(f"Generating narration with Groq ({self.model})...")
            
            # Call Groq API with narration-specific prompt
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                max_tokens=max_tokens, 
                temperature=temperature,
                top_p=0.9
            )
            
            # Extract the generated narration
            narration = chat_completion.choices[0].message.content
            
            # Clean up - remove any accidental code or markdown
            narration = self._clean_narration(narration)
            
            logger.info("Narration generation complete")
            return narration
            
        except Exception as e:
            logger.error(f"Groq API error generating narration: {e}")
            return f"Error generating narration: {str(e)}"
    
    def _clean_narration(self, text):
        """Clean narration text - remove any code, markdown, or technical junk."""
        if not text:
            return ""
        
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)
        
        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold -> plain
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Italic -> plain
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # Headers
        text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)  # Bullet points
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # Numbered lists
        
        # Remove any lines that look like code or technical junk
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip code-like lines
            if any(pattern in stripped for pattern in [
                'from manim', 'import ', 'def ', 'class ', 'self.', 
                '```', '.scale(', '.play(', '.wait(', '.move_to(', '.next_to(',
                'FadeIn', 'FadeOut', 'Create', 'Write', 'Axes(',
                'MathTex', 'Text(', '= ', 'lambda', 'VGroup', '.plot(',
            ]):
                continue
            # Skip lines starting with common meta-text
            if stripped.lower().startswith(('here is', 'here\'s', 'sure', 'certainly', 'of course',
                                            'note:', 'tip:', 'warning:', 'error:')):
                continue
            # Skip lines with rendering / technical junk
            if re.search(r'\b(fps|resolution|render(ing|ed|s)?|pixel|codec|bitrate|frame\s*rate|version\s*[\d.]|v\d+\.\d+)\b',
                         stripped, re.IGNORECASE):
                continue
            if re.search(r'\b(low|medium|high)\s*(quality|speed)\b', stripped, re.IGNORECASE):
                continue
            if re.search(r'\b(480|720|1080|1440|2160|4k)\s*p?\b', stripped, re.IGNORECASE):
                continue
            if re.search(r'\bspeed\b', stripped, re.IGNORECASE):
                continue
            if re.search(r'\b(animation|animate|playback|slow\s*down|fast\s*forward|encoding)\b',
                         stripped, re.IGNORECASE):
                continue
            # Skip separator lines
            if re.match(r'^[-=*]{3,}$', stripped):
                continue
            if stripped:
                clean_lines.append(stripped)
        
        result = ' '.join(clean_lines)
        
        # Normalize whitespace
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def generate_chat_response(self, prompt, system_prompt, max_tokens=1024, temperature=0.3):
 
        try:
            logger.info(f"Generating chat response with Groq ({self.model})...")
            
            # Call Groq API with custom system prompt
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=self.model,
                max_tokens=max_tokens, 
                temperature=temperature,
                top_p=0.9
            )
            
            # Extract the response
            response = chat_completion.choices[0].message.content
            
            logger.info("Chat response generation complete")
            return response
            
        except Exception as e:
            logger.error(f"Groq API error generating chat response: {e}")
            return f"I'm sorry, I encountered an error: {str(e)}"
    
    def _clean_code(self, code):
        """Clean and validate the generated code comprehensively."""
        # Remove markdown code fences
        if "```python" in code:
            code = code.split("```python")[1]
            if "```" in code:
                code = code.split("```")[0]
        elif "```" in code:
            parts = code.split("```")
            if len(parts) >= 2:
                code = parts[1]
        
        code = code.strip()
        
        # Ensure imports are present
        if not code.startswith("from manim import") and not code.startswith("import"):
            code = "from manim import *\n\n" + code
        
        # Apply all fixes in sequence
        code = self._fix_deprecated_manim_calls(code)
        code = self._fix_common_syntax_errors(code)
        code = self._fix_manim_specific_errors(code)
        code = self._ensure_scene_class_exists(code)
        code = self._validate_and_fix_python_syntax(code)
        code = self._replace_mathtex_with_text(code)
        
        return code
    
    def _fix_deprecated_manim_calls(self, code):
        """Replace deprecated Manim API calls with modern equivalents."""
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
            r'\bDrawBorderThenFill\b': 'DrawBorderThenFill',
        }
        
        for old_pattern, new_name in replacements.items():
            code = re.sub(old_pattern, new_name, code)
        
        # Fix unicode characters in Text() calls
        code = code.replace('Text("π', 'Text("pi')
        code = code.replace("Text('π", "Text('pi")
        code = code.replace('"\u03c0"', '"pi"')
        code = code.replace("'\u03c0'", "'pi'")
        
        return code
    
    def _fix_common_syntax_errors(self, code):
        """Fix common Python syntax errors in generated code."""
        lines = code.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            original_line = line
            
            # Fix unclosed parentheses in self.play() calls
            if 'self.play(' in line:
                open_count = line.count('(')
                close_count = line.count(')')
                if open_count > close_count:
                    line = line.rstrip() + ')' * (open_count - close_count)
            
            # Fix unclosed brackets in lists
            if '[' in line:
                open_count = line.count('[')
                close_count = line.count(']')
                if open_count > close_count:
                    line = line.rstrip() + ']' * (open_count - close_count)
            
            # Fix unclosed braces in dicts
            if '{' in line:
                open_count = line.count('{')
                close_count = line.count('}')
                if open_count > close_count:
                    line = line.rstrip() + '}' * (open_count - close_count)
            
            # Fix missing colons after class/def/if/for/while/else/elif/try/except/finally
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                # Check for class definition without colon
                if re.match(r'^class\s+\w+.*\)\s*$', stripped) and not stripped.endswith(':'):
                    line = line.rstrip() + ':'
                # Check for def without colon
                elif re.match(r'^def\s+\w+.*\)\s*$', stripped) and not stripped.endswith(':'):
                    line = line.rstrip() + ':'
                # Check for control structures
                elif re.match(r'^(if|elif|else|for|while|try|except|finally|with)\b', stripped):
                    if not stripped.endswith(':') and not stripped.endswith(','):
                        # Check if the line seems complete (not a multiline statement)
                        if ')' in stripped or stripped in ['else', 'try', 'finally']:
                            line = line.rstrip() + ':'
            
            # Fix common typos
            line = line.replace('slef.', 'self.')
            line = line.replace('sefl.', 'self.')
            line = line.replace('sel.', 'self.')
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _fix_manim_specific_errors(self, code):
        """Fix Manim-specific coding errors."""
        # Fix self.play(self.add(...)) - this is a common error
        code = re.sub(
            r'self\.play\s*\(\s*self\.add\s*\(',
            'self.play(FadeIn(',
            code
        )
        
        # Graph() is for network graphs, not mathematical function plots
        if re.search(r'Graph\s*\(\s*x_min|Graph\s*\(\s*x_range|Graph\s*\(\s*y_min', code):
            logger.warning("Detected incorrect Graph() usage for function plotting. This cannot be auto-fixed - Graph() is for network graphs only.")
            # Add a comment warning about this
            if 'Graph(' in code:
                code = "# WARNING: This code uses Graph() incorrectly. Graph() is for network graphs only.\n# For plotting mathematical functions, use Axes with .plot() method.\n# Example:\n# axes = Axes(x_range=[-4, 4, 1], y_range=[-5, 5, 1])\n# graph = axes.plot(lambda x: x**2, color=BLUE)\n\n" + code
        
        # Fix Write() used on shapes (should be Create())
        shape_classes = ['Circle', 'Square', 'Rectangle', 'Triangle', 'Line', 'Arrow', 
                        'Dot', 'Ellipse', 'Arc', 'Polygon', 'RegularPolygon', 'Star',
                        'Annulus', 'Sector', 'AnnularSector', 'Axes', 'NumberPlane']
        for shape in shape_classes:
            # Fix Write(Circle(...)) -> Create(Circle(...))
            pattern = rf'Write\s*\(\s*({shape}\s*\([^)]*\))'
            code = re.sub(pattern, r'Create(\1)', code)
        
        # Fix FadeIn used directly on text content instead of objects
        # e.g., FadeIn("text") -> FadeIn(Text("text"))
        code = re.sub(
            r'FadeIn\s*\(\s*(["\'][^"\']+["\'])\s*\)',
            r'FadeIn(Text(\1))',
            code
        )
        
        # Fix incorrect MathTex escaping (single backslash should be double in raw strings)
        # But be careful not to double-escape already escaped ones
        def fix_mathtex_escaping(match):
            content = match.group(1)
            # Common LaTeX commands that need proper escaping
            latex_commands = ['frac', 'sqrt', 'pi', 'times', 'div', 'pm', 'mp',
                            'cdot', 'ldots', 'cdots', 'vdots', 'ddots',
                            'alpha', 'beta', 'gamma', 'delta', 'epsilon',
                            'theta', 'lambda', 'mu', 'sigma', 'omega',
                            'infty', 'partial', 'nabla', 'sum', 'prod',
                            'int', 'oint', 'lim', 'sin', 'cos', 'tan',
                            'log', 'ln', 'exp', 'left', 'right', 'text']
            for cmd in latex_commands:
                # Replace single backslash with double (but not if already doubled)
                content = re.sub(rf'(?<!\\)\\{cmd}\b', rf'\\\\{cmd}', content)
            return f'MathTex(r"{content}"'
        
        # Apply to MathTex without raw string
        code = re.sub(r'MathTex\s*\(\s*"([^"]+)"', fix_mathtex_escaping, code)
        
        # Fix .animate used incorrectly
        # .animate should be followed by a method call, not a property assignment
        code = re.sub(
            r'\.animate\s*=',
            '.animate.set_color(',
            code
        )
        
        # Fix missing run_time in complex animations (add default if very long chain)
        if code.count('self.play(') > 10 and 'run_time' not in code:
            # Add a comment suggesting run_time but don't break working code
            pass
        
        # Fix VGroup with incorrect syntax
        # VGroup(a, b, c).arrange() is correct
        # VGroup([a, b, c]) should be VGroup(*[a, b, c]) or VGroup(a, b, c)
        code = re.sub(
            r'VGroup\s*\(\s*\[([^\]]+)\]\s*\)',
            r'VGroup(\1)',
            code
        )
        
        # Fix common positioning errors
        # UP*2 is correct, UP * 2 is also correct
        # But UP2 or UP 2 is wrong
        code = re.sub(r'\b(UP|DOWN|LEFT|RIGHT|ORIGIN)\s+(\d)', r'\1*\2', code)
        
        # Fix Tex vs MathTex confusion - Tex is for text, MathTex for math
        # Pattern: Tex(r"\frac...") should be MathTex
        code = re.sub(
            r'Tex\s*\(\s*r?["\'].*\\(frac|sqrt|int|sum|prod|alpha|beta|pi)',
            lambda m: m.group(0).replace('Tex(', 'MathTex(', 1),
            code
        )
        
        return code
    
    def _ensure_scene_class_exists(self, code):
        """Ensure the code has a valid Scene class."""
        # Check if a Scene class exists
        if not re.search(r'class\s+\w+\s*\(.*Scene.*\)\s*:', code):
            # Try to find any class definition and make it inherit from Scene
            match = re.search(r'class\s+(\w+)\s*\([^)]*\)\s*:', code)
            if match:
                class_name = match.group(1)
                code = re.sub(
                    rf'class\s+{class_name}\s*\([^)]*\)\s*:',
                    f'class {class_name}(Scene):',
                    code
                )
            else:
                # No class found, wrap the code in a Scene class
                # Find where the main code starts (after imports)
                lines = code.split('\n')
                import_end = 0
                for i, line in enumerate(lines):
                    if line.strip().startswith('from ') or line.strip().startswith('import ') or line.strip() == '':
                        import_end = i + 1
                    else:
                        break
                
                # Check if there's actual animation code
                remaining_code = '\n'.join(lines[import_end:])
                if 'self.play' in remaining_code or 'self.add' in remaining_code or 'self.wait' in remaining_code:
                    # There's animation code without a class, wrap it
                    imports = '\n'.join(lines[:import_end])
                    # Indent the remaining code
                    indented_code = '\n'.join('        ' + line if line.strip() else line for line in lines[import_end:])
                    code = f"""{imports}

class GeneratedScene(Scene):
    def construct(self):
{indented_code}
"""
        
        # Ensure construct method exists
        if 'def construct(self)' not in code and 'def construct(self):' not in code:
            # Find the Scene class and add construct method
            match = re.search(r'(class\s+\w+\s*\(.*Scene.*\)\s*:)', code)
            if match:
                class_def = match.group(1)
                code = code.replace(
                    class_def,
                    f"{class_def}\n    def construct(self):"
                )
        
        return code
    
    def _validate_and_fix_python_syntax(self, code):
        """Validate Python syntax and attempt to fix errors."""
        try:
            ast.parse(code)
            return code  # Code is valid
        except SyntaxError as e:
            logger.warning(f"Syntax error in generated code: {e}")
            
            # Try to fix common issues based on the error
            error_line = e.lineno
            lines = code.split('\n')
            
            if error_line and error_line <= len(lines):
                problem_line = lines[error_line - 1]
                
                # Fix: unexpected EOF - usually means unclosed brackets
                if 'unexpected EOF' in str(e) or 'EOF while scanning' in str(e):
                    # Count all brackets in entire code
                    total_open_paren = code.count('(')
                    total_close_paren = code.count(')')
                    total_open_bracket = code.count('[')
                    total_close_bracket = code.count(']')
                    total_open_brace = code.count('{')
                    total_close_brace = code.count('}')
                    
                    # Add missing closing brackets at the end
                    suffix = ')' * (total_open_paren - total_close_paren)
                    suffix += ']' * (total_open_bracket - total_close_bracket)
                    suffix += '}' * (total_open_brace - total_close_brace)
                    
                    if suffix:
                        code = code.rstrip() + suffix
                
                # Fix: expected ':'
                elif "expected ':'" in str(e):
                    if not problem_line.rstrip().endswith(':'):
                        lines[error_line - 1] = problem_line.rstrip() + ':'
                        code = '\n'.join(lines)
                
                # Fix: invalid syntax with common patterns
                elif 'invalid syntax' in str(e):
                    # Try removing the problematic line if it looks like garbage
                    if len(problem_line.strip()) < 3 or problem_line.strip() in [')', ']', '}', ',']:
                        lines.pop(error_line - 1)
                        code = '\n'.join(lines)
            
            # Try parsing again after fixes
            try:
                ast.parse(code)
                logger.info("Successfully fixed syntax error")
                return code
            except SyntaxError as e2:
                logger.error(f"Could not auto-fix syntax error: {e2}")
                # Return original code with error comment
                return f"# WARNING: Code may have syntax errors - please review\n# Error: {e}\n\n{code}"
    
    def _replace_mathtex_with_text(self, code):
        """Replace any MathTex/Tex calls with Text() as a safety net.
        
        Even though the system prompt bans MathTex, LLMs sometimes ignore
        instructions.  This catches any that slip through so compilation
        doesn't fail on systems without LaTeX installed.
        """
        if 'MathTex' not in code and 'Tex(' not in code:
            return code

        def _strip_latex(s: str) -> str:
            """Convert a LaTeX expression to readable plain text."""
            s = s.replace('\\\\', '')
            s = re.sub(r'\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'\1/\2', s)
            s = re.sub(r'\\sqrt\s*\{([^{}]*)\}', r'sqrt(\1)', s)
            s = re.sub(r'\\text\s*\{([^{}]*)\}', r'\1', s)
            for fn in ['sin','cos','tan','log','ln','exp','lim','int','sum']:
                s = re.sub(rf'\\{fn}\b', fn, s)
            s = re.sub(r'\\(alpha|beta|gamma|delta|theta|pi|sigma|omega|lambda|phi|psi|mu|epsilon)\b',
                        r'\1', s, flags=re.IGNORECASE)
            symbol_map = {r'\\infty': '\u221e', r'\\times': '\u00d7', r'\\cdot': '\u00b7',
                          r'\\pm': '\u00b1', r'\\leq': '\u2264', r'\\geq': '\u2265',
                          r'\\neq': '\u2260', r'\\rightarrow': '\u2192', r'\\Rightarrow': '\u21d2',
                          r'\\partial': '\u2202', r'\\nabla': '\u2207', r'\\approx': '\u2248'}
            for pat, repl in symbol_map.items():
                s = re.sub(pat, repl, s)
            s = re.sub(r'\^\{([^{}]*)\}', r'^\1', s)
            s = re.sub(r'_\{([^{}]*)\}', r'_\1', s)
            s = s.replace('{', '').replace('}', '')
            s = re.sub(r'\\(\w+)', r'\1', s)
            s = s.replace('\\', '')
            s = re.sub(r'\s+', ' ', s).strip()
            if len(s) > 55:
                s = s[:52] + '...'
            return s

        def _replacer_dq(m):
            plain = _strip_latex(m.group(2)).replace('"', "'")
            return f'Text("{plain}"'

        def _replacer_sq(m):
            plain = _strip_latex(m.group(2)).replace("'", '"')
            return f"Text('{plain}'"

        code = re.sub(r'\b(MathTex|Tex)\s*\(\s*r?"((?:[^"\\]|\\.)*)"', _replacer_dq, code)
        code = re.sub(r"\b(MathTex|Tex)\s*\(\s*r?'((?:[^'\\]|\\.)*)'", _replacer_sq, code)
        
        logger.info("_replace_mathtex_with_text: converted MathTex/Tex → Text()")
        return code

    def regenerate_with_error(self, original_prompt, context, error_message, 
                               max_tokens=2048, temperature=0.5):
        # Regenerate code using feedback from a previous error.
        
        try:
            # Build a focused prompt that includes the error for correction
            error_feedback_prompt = f"""The previously generated Manim code failed with this error:

=== ERROR MESSAGE ===
{error_message}
=== END ERROR ===

Original request: {original_prompt}

{f"Reference documentation:{chr(10)}{context}" if context else ""}

Please generate CORRECTED Manim code that fixes this error. 
Pay careful attention to:
1. The specific error message above
2. Correct Manim syntax and API usage
3. Proper Python indentation and structure

Generate complete, working code that avoids this error."""

            logger.info("Regenerating code with error feedback...")
            
            # Call the API with focused prompt
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """You are fixing Manim code that had an error.
Focus on:
1. Understanding the exact error
2. Writing correct code that avoids this error
3. Using proper Manim Community Edition syntax

CRITICAL: NEVER use MathTex() or Tex() — LaTeX is NOT installed.
Use Text() for ALL text including math formulas. Write math in plain readable form.

PREVENT OVERLAP:
- ALWAYS FadeOut ALL previous text/content before showing new text
- For step-by-step solutions: show ONE step at a time, FadeOut before next step
- Keep text scales small: .scale(0.3) for body, .scale(0.4) max for titles
- Maximum 3 text elements visible at the same time
- NEVER stack steps vertically — use a slide approach

Output ONLY the corrected Python code, starting with 'from manim import *'."""
                    },
                    {
                        "role": "user",
                        "content": error_feedback_prompt
                    }
                ],
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )
            
            code = chat_completion.choices[0].message.content
            code = self._clean_code(code)
            
            logger.info("Code regeneration with error feedback complete")
            return code
            
        except Exception as e:
            logger.error(f"Error during regeneration: {e}")
            return f"# Error regenerating code: {str(e)}"

    def extract_from_image(self, image_base64, mime_type="image/png"):
        # Use a vision model to read a photo of a math problem and return a plain-text description of what it contains.

        # Llama 4 Scout supports native multimodal (vision) input on Groq
        vision_model = "meta-llama/llama-4-scout-17b-16e-instruct"

        system_prompt = (
            "You are a helpful assistant that reads photos of math problems. "
            "Your job is to look at the image and output a clear, concise "
            "natural-language description of what the problem is about, "
            "what is being asked, and outline the solution steps shown.\n\n"
            "RULES:\n"
            "1. Describe the math topic (algebra, geometry, calculus, etc.)\n"
            "2. State the problem clearly in words\n"
            "3. List the solution steps if visible\n"
            "4. Use plain English — no LaTeX, no code\n"
            "5. If anything is unclear in the image, make a reasonable guess "
            "and note what was hard to read\n"
            "6. Keep the description under 200 words"
        )

        # Build the multimodal message with image
        data_uri = f"data:{mime_type};base64,{image_base64}"

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Look at this photo of a math problem. "
                            "Describe what it contains so I can turn it "
                            "into an animated video."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                ],
            },
        ]

        try:
            logger.info(f"Extracting math from image with {vision_model}...")
            response = self.client.chat.completions.create(
                model=vision_model,
                messages=messages,
                max_tokens=1024,
                temperature=0.3,
            )
            extracted = response.choices[0].message.content.strip()
            logger.info("Image extraction complete")
            return extracted

        except Exception as e:
            logger.error(f"Vision API error: {e}")
            return f"Error reading image: {str(e)}"

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
