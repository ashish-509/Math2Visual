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
                           LAYOUT RULES (PREVENT OVERLAP)
======================================================================================

SAFE ZONE: x=[-5.5, 5.5], y=[-3.2, 3.2] - content outside gets cut off

TEXT SIZING:
- Title: .scale(0.45).to_edge(UP)
- Body: .scale(0.32)
- Math: .scale(0.5)
- Labels: .scale(0.22)
- Max 35 chars per line, use \\n to break

PREVENT OVERLAP: Always FadeOut old content before showing new content.
Max 3 elements visible at once.

ALWAYS END WITH:
    self.wait(2)
    self.play(*[FadeOut(mob) for mob in self.mobjects])

======================================================================================
                            PYTHON SYNTAX RULES
======================================================================================

*** LIST SYNTAX - MUST BE CORRECT: ***
CORRECT:
    points = [(1, 2), (3, 4), (5, 6)]
    
CORRECT (multiline):
    points = [
        (1, 2),
        (3, 4),
    ]

WRONG (syntax error):
    points = []
        (1, 2),
    ]

*** MATHTEX SYNTAX: ***
- Use raw strings: MathTex(r"\\frac{a}{b}")
- Double backslashes: \\frac, \\sqrt, \\pi, \\times, \\div

*** DEPRECATED CALLS - USE MODERN API: ***
- Create() not ShowCreation()
- Uncreate() not ShowDestruction()
- FadeIn() / FadeOut() for appearing/disappearing

======================================================================================
                          COMPLETE WORKING TEMPLATE
======================================================================================

from manim import *

class ConceptVisualization(Scene):
    def construct(self):
        # 1. Title (stays briefly)
        title = Text("Topic Title Here").scale(0.45).to_edge(UP, buff=0.4)
        self.play(Write(title))
        self.wait(1)
        
        # 2. Explanation text (replace title)
        self.play(FadeOut(title))
        explanation = Text("Key concept explained\\nin simple terms").scale(0.32).move_to(UP * 1)
        self.play(Write(explanation))
        self.wait(1.5)
        
        # 3. Show visual (keep explanation, add graph below)
        axes = Axes(x_range=[-3,3,1], y_range=[-2,2,1], x_length=5, y_length=3)
        axes.scale(0.7).move_to(DOWN * 0.8)
        self.play(FadeOut(explanation))
        self.play(Create(axes))
        
        graph = axes.plot(lambda x: x**2, color=BLUE)
        self.play(Create(graph))
        self.wait(1)
        
        # 4. Add small label
        label = Text("y = x²").scale(0.22).next_to(axes, UP, buff=0.15)
        self.play(Write(label))
        self.wait(1.5)
        
        # 5. Clean ending
        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])

======================================================================================

Now generate the complete Manim code. Output ONLY the Python code, starting with 'from manim import *'."""

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
                        "content": f"Generate complete Manim code for:\n{prompt}\n\nREMINDER: Output COMPLETE code only. No truncation. No markdown. No explanations."
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
    
    def generate_narration(self, prompt, max_tokens=1024, temperature=0.8):
       
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

FORBIDDEN (never include):
- "from manim import" or any code
- "def", "class", "self.", "import"
- "Here is...", "Sure...", "Certainly..."
- "Hello everyone", "Welcome", "In this video"
- Markdown like ** or # or ```
- Timestamps like [0:00]

EXAMPLE OUTPUT:
"Local minima and maxima are fascinating points on a curve. At these special locations, the function momentarily stops increasing or decreasing. Think of it like a ball rolling on a hill - at the very top, it pauses before rolling down the other side. That peak is a local maximum. Similarly, the bottom of a valley is a local minimum. Mathematically, we find these points where the derivative equals zero."

Write natural, flowing speech that teaches the concept."""

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
        """Clean narration text - remove any code or markdown that slipped through."""
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
        
        # Remove any lines that look like code
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip code-like lines
            if any(pattern in stripped for pattern in [
                'from manim', 'import ', 'def ', 'class ', 'self.', 
                '```', 'python', '.scale(', '.play(', '.wait(', 
                'FadeIn', 'FadeOut', 'Create', 'Write', 'Axes(',
                'MathTex', 'Text(', '= ', 'lambda'
            ]):
                continue
            # Skip lines starting with common meta-text
            if stripped.lower().startswith(('here is', 'here\'s', 'sure', 'certainly', 'of course')):
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
