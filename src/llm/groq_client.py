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

=== CRITICAL: TEXT AND LAYOUT RULES TO PREVENT OVERLAP/OVERFLOW ===

**FRAME BOUNDARIES - NOTHING OUTSIDE THESE LIMITS:**
- Horizontal: LEFT edge is -7, RIGHT edge is +7 (safe zone: -6 to +6)
- Vertical: TOP edge is +4, BOTTOM edge is -4 (safe zone: -3.5 to +3.5)
- ALWAYS check positions stay within safe zone

**TEXT SIZING - MANDATORY SCALING:**
- Title text: .scale(0.6) and position at TOP with .to_edge(UP, buff=0.3)
- Main explanation text: .scale(0.5) MAXIMUM - smaller for longer text
- Math equations (MathTex): .scale(0.6) for main, .scale(0.4) for secondary
- Labels and small text: .scale(0.4)
- NEVER use scale > 0.7 for any text element

**LINE LENGTH LIMITS - MUST FOLLOW:**
- Maximum 40 characters per line for Text()
- For longer text, SPLIT into multiple lines using "\n"
- Example: Text("This is the first line\nThis is the second line").scale(0.5)
- For very long explanations, use multiple separate Text objects stacked vertically

**MULTI-LINE TEXT PATTERN:**
text = Text("Line 1 here\nLine 2 here\nLine 3 here", line_spacing=0.8).scale(0.5)
text.move_to(ORIGIN)  # or specific position

**PREVENTING OVERLAP - STRICT SEQUENCE:**
1. ALWAYS FadeOut or move away previous content BEFORE showing new content
2. NEVER have more than 2-3 objects visible at once unless they are small and positioned apart
3. Use these positions for multiple elements:
   - Title: .to_edge(UP, buff=0.3)
   - Main content: ORIGIN or .move_to(UP*0.5)
   - Secondary content: .move_to(DOWN*1.5)
   - Old content (if keeping): .scale(0.3).to_corner(UL, buff=0.2)

**CORRECT PATTERN FOR SEQUENTIAL CONTENT:**
```python
# Show first concept
concept1 = Text("First explanation here").scale(0.5)
self.play(Write(concept1))
self.wait(2)

# MUST fade out before showing next
self.play(FadeOut(concept1))

# Now show second concept
concept2 = Text("Second explanation").scale(0.5)
self.play(Write(concept2))
self.wait(2)
```

**FOR STEP-BY-STEP EXPLANATIONS:**
```python
# Keep title fixed at top
title = Text("Topic Title").scale(0.6).to_edge(UP, buff=0.3)
self.play(Write(title))

# Show step 1
step1 = Text("Step 1: Do this").scale(0.5).move_to(ORIGIN)
self.play(Write(step1))
self.wait(2)

# Move step1 aside, show step 2
self.play(step1.animate.scale(0.5).to_corner(UL, buff=0.2))
step2 = Text("Step 2: Then this").scale(0.5).move_to(ORIGIN)
self.play(Write(step2))
self.wait(2)
```

**GROUPING RELATED ITEMS:**
- Use VGroup() to group related items
- Scale the entire group: VGroup(item1, item2).scale(0.5)
- Arrange vertically: group.arrange(DOWN, buff=0.3)
- Check group fits in frame before displaying

**EQUATIONS WITH EXPLANATIONS:**
```python
# Equation and label together, properly sized
eq = MathTex(r"E = mc^2").scale(0.6)
label = Text("Energy-mass equivalence").scale(0.4).next_to(eq, DOWN, buff=0.3)
group = VGroup(eq, label).move_to(ORIGIN)
self.play(Write(group))
```

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
14. ALWAYS scale text with .scale(0.5) or smaller to prevent overflow
15. ALWAYS fade out previous content before showing new content to prevent overlap
16. Keep all elements within the safe frame zone (-6 to +6 horizontal, -3.5 to +3.5 vertical)

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
        
        # Fix Write() used on shapes (should be Create())
        shape_classes = ['Circle', 'Square', 'Rectangle', 'Triangle', 'Line', 'Arrow', 
                        'Dot', 'Ellipse', 'Arc', 'Polygon', 'RegularPolygon', 'Star',
                        'Annulus', 'Sector', 'AnnularSector']
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
