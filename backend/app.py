"""
This backend handles all the heavy lifting:
- Code generation using different LLM models
- Video compilation with Manim
- Teaching script generation
- Text-to-speech conversion

The Streamlit frontend calls these endpoints via HTTP.
"""

import os
import sys
import re
import ast
import logging
import tempfile
import shutil
import subprocess
from typing import Optional, Tuple
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from gtts import gTTS

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Import our custom modules

# RAG + Finetuned Model pipeline
try:
    from src.pipeline.rag_finetuned_pipeline import get_rag_finetuned_pipeline
    RAG_FINETUNED_AVAILABLE = True
except ImportError as e:
    RAG_FINETUNED_AVAILABLE = False
    logger.warning(f"RAG Finetuned pipeline not available: {e}")

# Groq client for CodeLlama and Phi-2
try:
    from src.llm.groq_client import get_groq_client
    GROQ_AVAILABLE = True
except ImportError as e:
    GROQ_AVAILABLE = False
    logger.warning(f"Groq client not available: {e}")

# Standalone RAG pipeline
try:
    from src.rag.rag_pipeline import create_rag_pipeline
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    logger.warning(f"RAG pipeline not available: {e}")

# Manim Syntax Chatbot
try:
    from src.chatbot.manim_chatbot import get_manim_chatbot
    CHATBOT_AVAILABLE = True
except ImportError as e:
    CHATBOT_AVAILABLE = False
    logger.warning(f"Manim chatbot not available: {e}")


# Create FastAPI app

app = FastAPI(
    title="Math2Visual Backend",
    description="Backend API for Math2Visual - generates Manim animations from text",
    version="1.0.0"
)

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request and Response Models

class CodeGenerationRequest(BaseModel):
    prompt: str
    model_choice: str = "CodeLlama-34B"


class CodeGenerationResponse(BaseModel):
    success: bool
    code: str
    message: str = ""


class VideoCompilationRequest(BaseModel):
    code: str
    quality: str = "medium"


class VideoCompilationResponse(BaseModel):
    success: bool
    video_path: Optional[str] = None
    error_message: Optional[str] = None


class TeachingScriptRequest(BaseModel):
    animation_description: str
    manim_code: str
    model_choice: str = "CodeLlama-34B"
    video_duration: float = 0.0  # Duration in seconds for script timing


class TeachingScriptResponse(BaseModel):
    success: bool
    script: str
    message: str = ""


class TTSRequest(BaseModel):
    text: str


class TTSResponse(BaseModel):
    success: bool
    audio_path: Optional[str] = None
    message: str = ""


class MergeVideoAudioRequest(BaseModel):
    video_path: str
    audio_text: str


class MergeVideoAudioResponse(BaseModel):
    success: bool
    final_video_path: Optional[str] = None
    message: str = ""


class HealthResponse(BaseModel):
    status: str
    models: dict


# Chatbot Request/Response Models
class ChatbotRequest(BaseModel):
    question: str
    include_examples: bool = False


class ChatbotResponse(BaseModel):
    success: bool
    answer: str
    sources: str = ""
    cached: bool = False
    message: str = ""


# Cached Resources (singleton pattern)

# Cache for pipeline instances
_rag_pipeline_cache = None
_rag_finetuned_cache = None
_groq_clients_cache = {}
_chatbot_cache = None


def get_chatbot_cached():
    """Get or create the Manim syntax chatbot (singleton)."""
    global _chatbot_cache
    
    if _chatbot_cache is not None:
        return _chatbot_cache
    
    if not CHATBOT_AVAILABLE:
        return None
    
    try:
        chatbot = get_manim_chatbot()
        if chatbot.is_ready:
            _chatbot_cache = chatbot
            return _chatbot_cache
        return None
    except Exception as e:
        logger.error(f"Failed to create chatbot: {e}")
        return None


def get_rag_pipeline_cached():
    global _rag_pipeline_cache
    
    if _rag_pipeline_cache is not None:
        return _rag_pipeline_cache
    
    if not RAG_AVAILABLE:
        return None
    
    try:
        _rag_pipeline_cache = create_rag_pipeline()
        return _rag_pipeline_cache
    except Exception as e:
        logger.error(f"Failed to create RAG pipeline: {e}")
        return None


def get_rag_finetuned_cached():
    global _rag_finetuned_cache
    
    if _rag_finetuned_cache is not None:
        return _rag_finetuned_cache
    
    if not RAG_FINETUNED_AVAILABLE:
        return None
    
    try:
        pipeline = get_rag_finetuned_pipeline()
        if pipeline.initialize():
            _rag_finetuned_cache = pipeline
            return _rag_finetuned_cache
        return None
    except Exception as e:
        logger.error(f"Failed to create RAG Finetuned pipeline: {e}")
        return None


def get_groq_client_cached(model_type: str):
    global _groq_clients_cache
    
    if model_type in _groq_clients_cache:
        return _groq_clients_cache[model_type]
    
    if not GROQ_AVAILABLE:
        return None
    
    try:
        client = get_groq_client(model_type)
        _groq_clients_cache[model_type] = client
        return client
    except Exception as e:
        logger.error(f"Failed to create Groq client: {e}")
        return None


# Helper Functions

def extract_class_name(code: str) -> Optional[str]:
    pattern = r'class\s+(\w+)\s*\(.*Scene.*\)'
    matches = re.findall(pattern, code)
    if matches:
        return matches[0]
    return None


def clean_text_for_tts(text: str) -> str:
    """
    Clean text for TTS by removing code, warnings, LLM meta-text, and technical content.
    Only keep natural language that should be spoken aloud.
    """
    if not text:
        return ""
    
    # First, remove common LLM meta-text patterns (these should NEVER be spoken)
    llm_meta_patterns = [
        r'^Here is the.*?:?\s*',
        r'^Here\'s the.*?:?\s*',
        r'^Below is the.*?:?\s*',
        r'^The following is.*?:?\s*',
        r'^This is the.*?narration.*?:?\s*',
        r'^This is the.*?script.*?:?\s*',
        r'^I\'ll provide.*?:?\s*',
        r'^I will provide.*?:?\s*',
        r'^Let me.*?:?\s*',
        r'^Sure,.*?:?\s*',
        r'^Certainly,.*?:?\s*',
        r'^Of course,.*?:?\s*',
        r'^Here you go.*?:?\s*',
        r'^\*\*.*?\*\*\s*',  # Markdown bold headers
        r'^#+\s+.*?\n',  # Markdown headers
        r'^Note:.*?\n',
        r'^NOTE:.*?\n',
        r'^Warning:.*?\n',
        r'^WARNING:.*?\n',
        r'^Error:.*?\n',
        r'^ERROR:.*?\n',
        r'^Tip:.*?\n',
        r'^TIP:.*?\n',
    ]
    
    result = text
    for pattern in llm_meta_patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE | re.MULTILINE)
    
    lines = result.split('\n')
    cleaned_lines = []
    
    # Patterns to skip entirely
    skip_patterns = [
        r'^#',  # Comments
        r'^from\s+\w+\s+import',  # Import statements
        r'^import\s+',  # Import statements
        r'^\s*class\s+\w+',  # Class definitions
        r'^\s*def\s+\w+',  # Function definitions
        r'^\s*self\.',  # Self references
        r'^```',  # Code fences
        r'^\s*WARNING',  # Warnings
        r'^\s*Error',  # Errors
        r'^\s*#\s*WARNING',  # Code comments with warnings
        r'^\s*\w+\s*=\s*\w+\(',  # Variable assignments like x = Circle()
        r'^\s*return\s+',  # Return statements
        r'^\s*if\s+.*:',  # If statements
        r'^\s*for\s+.*:',  # For loops
        r'^\s*while\s+.*:',  # While loops
        r'^\s*try:',  # Try blocks
        r'^\s*except',  # Except blocks
        r'^\s*\[',  # Lists
        r'^\s*\{',  # Dicts
        r'\.scale\(',  # Manim scale
        r'\.move_to\(',  # Manim positioning
        r'\.next_to\(',  # Manim positioning
        r'MathTex\(',  # MathTex
        r'Text\(',  # Text objects
        r'Axes\(',  # Axes
        r'\.plot\(',  # Plot
        r'VGroup\(',  # VGroup
        r'FadeIn\(',  # Animations
        r'FadeOut\(',  # Animations
        r'Create\(',  # Animations
        r'Write\(',  # Animations
        r'python',  # Code language markers
        r'```',  # Code fences
        r'manim',  # Manim references in code
    ]
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
        
        # Skip very short lines (likely artifacts)
        if len(stripped) < 3:
            continue
        
        # Check if line matches any skip pattern
        should_skip = False
        for pattern in skip_patterns:
            if re.search(pattern, stripped, re.IGNORECASE):
                should_skip = True
                break
        
        if should_skip:
            continue
        
        # Skip lines that look like code (have parentheses with parameters)
        if re.search(r'\w+\([^)]*\)', stripped) and '=' in stripped:
            continue
        
        # Skip lines that are mostly symbols/punctuation
        alpha_chars = sum(1 for c in stripped if c.isalpha() or c.isspace())
        if len(stripped) > 0 and alpha_chars / len(stripped) < 0.6:
            continue
        
        # Skip lines that look like code variable names or technical terms
        if re.match(r'^[a-z_]+[A-Z]', stripped):  # camelCase
            continue
        if re.match(r'^[a-z]+_[a-z]+', stripped):  # snake_case
            continue
        
        # This line looks like natural language, keep it
        cleaned_lines.append(stripped)
    
    # Join and clean up
    result = ' '.join(cleaned_lines)
    
    # Remove any remaining code-like patterns
    result = re.sub(r'```[\s\S]*?```', '', result)  # Remove code blocks
    result = re.sub(r'`[^`]+`', '', result)  # Remove inline code
    result = re.sub(r'\*\*[^*]+\*\*', '', result)  # Remove markdown bold
    result = re.sub(r'\*[^*]+\*', '', result)  # Remove markdown italic
    result = re.sub(r'\[[^\]]+\]\([^)]+\)', '', result)  # Remove markdown links
    result = re.sub(r'https?://\S+', '', result)  # Remove URLs
    result = re.sub(r'\s+', ' ', result)  # Normalize whitespace
    
    # Final cleanup - remove any remaining technical artifacts
    result = re.sub(r'\b(def|class|import|from|self|return|if|else|for|while|try|except)\b', '', result)
    result = re.sub(r'\s+', ' ', result)  # Normalize whitespace again
    
    return result.strip()


def fix_layout_issues_in_code(code: str) -> str:
    """
    Post-process generated code to fix common layout issues:
    - Text too large (enforce strict scale limits)
    - Axes too large
    - Missing scale on text elements
    - Long text not split
    """
    if not code:
        return code
    
    lines = code.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        fixed_line = line
        
        
        scale_match = re.search(r'\.scale\s*\(\s*([\d.]+)\s*\)', fixed_line)
        if scale_match:
            try:
                scale_val = float(scale_match.group(1))
                
                # Check if this is a Text element
                if re.search(r'\bText\s*\(', fixed_line):
                    # Title text (to_edge(UP) or first few lines)
                    if 'to_edge(UP' in fixed_line:
                        max_scale = 0.45
                    # Labels (next_to, small text)
                    elif 'next_to' in fixed_line or 'label' in fixed_line.lower():
                        max_scale = 0.22
                    # Body text
                    else:
                        max_scale = 0.32
                    
                    if scale_val > max_scale:
                        fixed_line = re.sub(
                            r'\.scale\s*\(\s*[\d.]+\s*\)',
                            f'.scale({max_scale})',
                            fixed_line
                        )
                
                # Check if this is a MathTex element
                elif re.search(r'\b(MathTex|Tex)\s*\(', fixed_line):
                    if 'next_to' in fixed_line or 'label' in fixed_line.lower():
                        max_scale = 0.22
                    else:
                        max_scale = 0.5
                    
                    if scale_val > max_scale:
                        fixed_line = re.sub(
                            r'\.scale\s*\(\s*[\d.]+\s*\)',
                            f'.scale({max_scale})',
                            fixed_line
                        )
                
                # Check if this is Axes - limit scale
                elif re.search(r'\bAxes\s*\(', fixed_line) or re.search(r'\baxes\.', fixed_line):
                    if scale_val > 0.75:
                        fixed_line = re.sub(
                            r'\.scale\s*\(\s*[\d.]+\s*\)',
                            '.scale(0.7)',
                            fixed_line
                        )
            except ValueError:
                pass
        
        # If Text has no scale, add one
        if re.search(r'\bText\s*\([^)]+\)\s*$', fixed_line.strip()):
            if '.scale' not in fixed_line:
                fixed_line = fixed_line.rstrip() + '.scale(0.32)'
        
        # If MathTex has no scale, add one
        if re.search(r'\b(MathTex|Tex)\s*\([^)]+\)\s*$', fixed_line.strip()):
            if '.scale' not in fixed_line:
                fixed_line = fixed_line.rstrip() + '.scale(0.5)'
        
        # Fix axes that are too large - add size constraints
        if 'Axes(' in fixed_line:
            # Check for x_length
            if 'x_length' in fixed_line:
                # Reduce if too large
                x_len_match = re.search(r'x_length\s*=\s*([\d.]+)', fixed_line)
                if x_len_match:
                    x_len = float(x_len_match.group(1))
                    if x_len > 6:
                        fixed_line = re.sub(r'x_length\s*=\s*[\d.]+', 'x_length=5', fixed_line)
            
            if 'y_length' in fixed_line:
                y_len_match = re.search(r'y_length\s*=\s*([\d.]+)', fixed_line)
                if y_len_match:
                    y_len = float(y_len_match.group(1))
                    if y_len > 4:
                        fixed_line = re.sub(r'y_length\s*=\s*[\d.]+', 'y_length=3', fixed_line)
        
        fixed_lines.append(fixed_line)
    
    return '\n'.join(fixed_lines)


def validate_and_fix_manim_code(code: str) -> Tuple[bool, str, str]:
    """
    Comprehensive validation and fixing of Manim code before compilation.
    Returns: (is_valid, fixed_code, error_message)
    """
    if not code or not code.strip():
        return False, code, "Empty code provided"
    
    original_code = code
    errors_fixed = []
    
    # Step 0: Fix layout issues first
    code = fix_layout_issues_in_code(code)
    
    # Step 1: Basic cleanup
    code = code.strip()
    
    # Step 2: Ensure proper imports
    if 'from manim import' not in code and 'import manim' not in code:
        code = "from manim import *\n\n" + code
        errors_fixed.append("Added missing manim import")
    
    # Step 3: Fix deprecated Manim calls
    deprecated_replacements = {
        r'\bShowCreation\b': 'Create',
        r'\bShowDestruction\b': 'Uncreate',
        r'\bWiggleOutThenIn\b': 'Wiggle',
        r'\bDrawBorderThenFill\b': 'DrawBorderThenFill',
    }
    for old, new in deprecated_replacements.items():
        if re.search(old, code):
            code = re.sub(old, new, code)
            errors_fixed.append(f"Replaced deprecated {old} with {new}")
    
    # Step 4: Fix self.play(self.add(...)) error
    if 'self.play(self.add(' in code or re.search(r'self\.play\s*\(\s*self\.add\s*\(', code):
        code = re.sub(r'self\.play\s*\(\s*self\.add\s*\(', 'self.play(FadeIn(', code)
        errors_fixed.append("Fixed self.play(self.add()) -> self.play(FadeIn())")
    
    # Step 4b: Check for incorrect Graph() usage (Graph is for network graphs, not function plots)
    if re.search(r'Graph\s*\(\s*(x_min|x_range|y_min|y_range|x_max|y_max)', code):
        return False, original_code, """ERROR: Incorrect use of Graph() detected.

Graph() in Manim is for NETWORK GRAPHS (nodes and edges), NOT for plotting mathematical functions.

To plot mathematical functions like y = f(x), use Axes with .plot():

    axes = Axes(x_range=[-4, 4, 1], y_range=[-5, 5, 1], x_length=6, y_length=4)
    graph = axes.plot(lambda x: x**3 - 6*x**2 + 9*x + 2, color=BLUE)
    self.play(Create(axes), Create(graph))

Please regenerate the code with the correct approach."""

    # Step 4c: Fix broken list syntax (var = [] followed by indented items)
    # Pattern: "var = []" on one line, then indented items, then "]"
    # This handles cases like:
    #   labels = []
    #       Text("..."),
    #       Text("..."),
    #   ]
    
    def fix_broken_lists(code_text):
        """Fix broken list syntax where = [] is followed by indented items then ]"""
        lines = code_text.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if line matches "var = []" pattern
            match = re.match(r'^(\s*)(\w+)\s*=\s*\[\]\s*$', line)
            if match:
                indent = match.group(1)
                var_name = match.group(2)
                
                # Look ahead for indented items followed by ]
                items = []
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    stripped = next_line.strip()
                    
                    # Check if this is the closing bracket
                    if stripped == ']':
                        # Found the pattern - fix it
                        if items:
                            fixed_lines.append(f'{indent}{var_name} = [')
                            for item in items:
                                fixed_lines.append(item)
                            fixed_lines.append(f'{indent}]')
                            i = j + 1
                            break
                        else:
                            # Empty list, keep as is
                            fixed_lines.append(line)
                            i += 1
                            break
                    elif stripped and not stripped.startswith('#'):
                        # This is a list item
                        items.append(next_line)
                        j += 1
                    elif not stripped:
                        # Empty line, skip
                        j += 1
                    else:
                        # Something else, not our pattern
                        fixed_lines.append(line)
                        i += 1
                        break
                else:
                    # Reached end without finding ], keep original
                    fixed_lines.append(line)
                    i += 1
            else:
                fixed_lines.append(line)
                i += 1
        
        return '\n'.join(fixed_lines)
    
    old_code = code
    code = fix_broken_lists(code)
    if code != old_code:
        errors_fixed.append("Fixed broken list syntax (var = [] followed by items)")

    # Step 5: Fix common syntax issues
    lines = code.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        # Fix typos
        line = line.replace('slef.', 'self.')
        line = line.replace('sefl.', 'self.')
        line = line.replace('sel.', 'self.')
        
        # Fix MathMathTex -> MathTex typo (LLM sometimes doubles it)
        line = line.replace('MathMathTex', 'MathTex')
        line = line.replace('TextText', 'Text')
        
        # Fix unclosed parentheses in single lines
        if 'self.play(' in line or 'self.add(' in line:
            open_count = line.count('(')
            close_count = line.count(')')
            if open_count > close_count:
                line = line.rstrip() + ')' * (open_count - close_count)
                errors_fixed.append(f"Fixed unclosed parentheses on line {i+1}")
        
        # Fix missing colons
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            if re.match(r'^class\s+\w+.*\)\s*$', stripped) and not stripped.endswith(':'):
                line = line.rstrip() + ':'
                errors_fixed.append("Added missing colon to class definition")
            elif re.match(r'^def\s+\w+.*\)\s*$', stripped) and not stripped.endswith(':'):
                line = line.rstrip() + ':'
                errors_fixed.append("Added missing colon to function definition")
        
        fixed_lines.append(line)
    
    code = '\n'.join(fixed_lines)
    
    # Step 6: Ensure Scene class exists
    if not re.search(r'class\s+\w+\s*\(.*Scene.*\)\s*:', code):
        # Try to find class without Scene inheritance
        match = re.search(r'class\s+(\w+)\s*\([^)]*\)\s*:', code)
        if match:
            class_name = match.group(1)
            code = re.sub(
                rf'class\s+{class_name}\s*\([^)]*\)\s*:',
                f'class {class_name}(Scene):',
                code
            )
            errors_fixed.append(f"Made {class_name} inherit from Scene")
        else:
            # Check if there's animation code without a class
            if 'self.play' in code or 'self.add' in code:
                return False, original_code, "Animation code found outside of a Scene class. Please wrap your code in a class that inherits from Scene."
    
    # Step 7: Ensure construct method exists
    if re.search(r'class\s+\w+\s*\(.*Scene.*\)\s*:', code):
        if 'def construct(self)' not in code:
            match = re.search(r'(class\s+\w+\s*\(.*Scene.*\)\s*:)', code)
            if match:
                class_def = match.group(1)
                parts = code.split(class_def)
                if len(parts) == 2:
                    after_class = parts[1]
                    if 'def construct' not in after_class and 'def ' not in after_class[:50]:
                        code = parts[0] + class_def + "\n    def construct(self):" + after_class
                        errors_fixed.append("Added missing construct method")
    
    # Step 8: Validate Python syntax
    try:
        ast.parse(code)
    except SyntaxError as e:
        logger.warning(f"Syntax error in code: {e}")
        error_msg = str(e)
        
        # Fix unexpected EOF (unclosed brackets)
        if 'unexpected EOF' in error_msg or 'EOF while scanning' in error_msg:
            open_p = code.count('(')
            close_p = code.count(')')
            open_b = code.count('[')
            close_b = code.count(']')
            open_c = code.count('{')
            close_c = code.count('}')
            
            code = code.rstrip()
            if open_p > close_p:
                code += ')' * (open_p - close_p)
            if open_b > close_b:
                code += ']' * (open_b - close_b)
            if open_c > close_c:
                code += '}' * (open_c - close_c)
            errors_fixed.append("Fixed unclosed brackets")
            
            try:
                ast.parse(code)
            except SyntaxError as e2:
                return False, code, f"Syntax error: {e2}"
        else:
            return False, code, f"Syntax error: {e}"
    
    # Step 9: Fix self.wait without parentheses
    if re.search(r'self\.wait[^(]', code):
        code = re.sub(r'self\.wait\s*$', 'self.wait()', code, flags=re.MULTILINE)
        code = re.sub(r'self\.wait\s+', 'self.wait() ', code)
        errors_fixed.append("Fixed self.wait -> self.wait()")
    
    if errors_fixed:
        logger.info(f"Code validation fixed {len(errors_fixed)} issues: {errors_fixed}")
    
    return True, code, ""


def generate_code_with_model(prompt: str, model_choice: str) -> tuple:
    if not prompt or not prompt.strip():
        return False, "Please enter a description of what you want to visualize"
    
    # Use finetuned Mistral model
    if model_choice == "Mistral-7B (Finetuned)":
        if not RAG_FINETUNED_AVAILABLE:
            return False, "Finetuned model not available"
        
        try:
            pipeline = get_rag_finetuned_cached()
            if pipeline is None:
                return False, "Could not initialize the RAG + Finetuned model pipeline"
            
            code = pipeline.generate_manim_code(prompt, use_rag=True)
            
            # Validate and fix the generated code
            is_valid, fixed_code, error_msg = validate_and_fix_manim_code(code)
            if not is_valid:
                logger.warning(f"Finetuned model code validation failed: {error_msg}")
                return False, f"# Code validation error: {error_msg}\n# Original code below may have errors:\n\n{code}"
            
            return True, fixed_code
            
        except Exception as e:
            logger.warning(f"Finetuned model failed: {e}")
            # Fall back to CodeLlama
            model_choice = "CodeLlama-34B"
    
    # Use Groq API for CodeLlama or Phi-2
    if model_choice in ["CodeLlama-34B", "Phi-2"]:
        if not GROQ_AVAILABLE:
            return False, "Groq client not available"
        
        try:
            # Get RAG context first
            rag = get_rag_pipeline_cached()
            context = ""
            if rag and rag.is_indexed:
                context = rag.retrieve_context(prompt)
            
            # Layout rules to prevent text overlap and overflow
            layout_rules = """
==============================================
                    MANDATORY LAYOUT RULES (VIOLATIONS CAUSE ERRORS)
==============================================

*** SCREEN SAFE ZONE: ***
- Horizontal: -5.5 to +5.5 only
- Vertical: -3.2 to +3.2 only
- Content OUTSIDE these bounds gets CUT OFF

*** TEXT SCALE LIMITS: ***
- Titles: .scale(0.45) maximum
- Body text: .scale(0.32) maximum
- Math: .scale(0.5) maximum
- Labels: .scale(0.22) maximum
- Split text longer than 35 chars with \\n

*** OVERLAP PREVENTION: ***
- ALWAYS FadeOut previous content before showing new content
- Maximum 3 visible elements at once
- Pattern: FadeOut(old) -> then -> Write(new)

*** AXES/GRAPHS: ***
- Use Axes with x_length=5, y_length=3 maximum
- Scale axes with .scale(0.7)
- Position graphs at DOWN * 0.5 to leave room for labels above

*** TIMING: ***
- Total animation: 20-40 seconds
- self.wait(1) after each element
- End with: self.play(*[FadeOut(mob) for mob in self.mobjects])
"""
            
            # Build augmented prompt with RAG context
            if context:
                augmented_prompt = f"""Use the following Manim documentation as reference:

=== MANIM DOCUMENTATION ===
{context}
=== END DOCUMENTATION ===
{layout_rules}

User Request: {prompt}

Generate complete, working Manim code based on the documentation above."""
            else:
                augmented_prompt = f"""{layout_rules}

User Request: {prompt}"""
            
            # Get Groq client
            model_type = "codellama" if model_choice == "CodeLlama-34B" else "phi2"
            groq_client = get_groq_client_cached(model_type)
            
            if groq_client is None:
                return False, "Failed to initialize Groq client"
            
            code = groq_client.generate(augmented_prompt, max_tokens=2048, temperature=0.7)
            
            # Validate and fix the generated code
            is_valid, fixed_code, error_msg = validate_and_fix_manim_code(code)
            if not is_valid:
                logger.warning(f"Generated code validation failed: {error_msg}")
                # Return the error but still provide the code for debugging
                return False, f"# Code validation error: {error_msg}\n# Original code below may have errors:\n\n{code}"
            
            return True, fixed_code
            
        except Exception as e:
            return False, f"Groq API error: {str(e)}"
    
    return False, f"Unknown model choice: {model_choice}"


def compile_video(code: str, quality: str = "medium") -> tuple:
    # First, validate and fix the code
    is_valid, code, error_msg = validate_and_fix_manim_code(code)
    if not is_valid:
        return False, f"Code validation failed: {error_msg}"
    
    # Extract the class name
    class_name = extract_class_name(code)
    if not class_name:
        return False, "Could not find a Scene class in the code"
    
    # Quality flags
    quality_flags = {
        "low": "-ql",
        "medium": "-qm",
        "high": "-qh"
    }
    quality_flag = quality_flags.get(quality, "-qm")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="manim_")
    script_path = os.path.join(temp_dir, "generated_scene.py")
    
    try:
        # Write the code to temp file
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
        
        logger.info(f"Saved script to: {script_path}")
        logger.info(f"Compiling class: {class_name}")
        
        # Build Manim command
        cmd = [
            "manim",
            quality_flag,
            script_path,
            class_name,
            "--media_dir", temp_dir
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        # Run Manim
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.stdout:
            logger.info(f"Manim stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"Manim stderr: {result.stderr}")
        
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            return False, f"Manim compilation failed:\n{error_msg}"
        
        # Find the output video
        video_dir = os.path.join(temp_dir, "videos", "generated_scene")
        video_path = None
        
        for root, dirs, files in os.walk(video_dir):
            for file in files:
                if file.endswith(".mp4"):
                    video_path = os.path.join(root, file)
                    break
            if video_path:
                break
        
        if not video_path or not os.path.exists(video_path):
            return False, "Video compiled but output file not found"
        
        # Copy to persistent location
        output_dir = os.path.join(PROJECT_ROOT, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        final_path = os.path.join(output_dir, f"{class_name}.mp4")
        shutil.copy2(video_path, final_path)
        
        logger.info(f"Video saved to: {final_path}")
        return True, final_path
        
    except subprocess.TimeoutExpired:
        return False, "Compilation timed out (exceeded 5 minutes)"
    except FileNotFoundError:
        return False, "Manim not installed. Run: pip install manim"
    except Exception as e:
        logger.error(f"Error compiling: {e}")
        return False, f"Compilation error: {str(e)}"
    finally:
        # Cleanup temp directory
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


def generate_teaching_script_text(description: str, code: str, model_choice: str, video_duration: float = 0.0) -> tuple:
    if not description or not description.strip():
        return False, "Please provide a description first"
    
    if not code or not code.strip():
        return False, "Please generate the animation code first"
    
    # Calculate target word count based on video duration
    # Average speaking rate is about 150 words per minute (2.5 words per second)
    if video_duration > 0:
        target_words = int(video_duration * 2.5)
        duration_instruction = f"""
CRITICAL TIMING REQUIREMENT:
- The video is exactly {video_duration:.1f} seconds long
- Your script must be approximately {target_words} words (speaking at normal pace)
- This ensures the narration matches the video perfectly
- Count your words carefully to match this target
- Do not exceed {target_words + 20} words or go below {max(target_words - 20, 30)} words"""
    else:
        duration_instruction = ""
    
    # Build the teaching prompt - designed for perfect audio-video sync
    teaching_prompt = f"""You are creating a voiceover script for a math teaching video animation.

==============================================================================
                         CRITICAL OUTPUT REQUIREMENTS
==============================================================================

*** OUTPUT FORMAT: ***
- Output ONLY the spoken narration text
- Start speaking IMMEDIATELY - no introductions
- NO meta-text like "Here is the narration:" or "Sure, here's..."
- NO markdown, NO headers, NO bullet points
- NO timestamps like [0:00] or (pause)
- Just plain spoken words, nothing else

*** WHAT YOU ARE NARRATING: ***
Topic: "{description}"

Animation code (study this to understand WHEN visuals appear):
```python
{code}
```
{duration_instruction}

==============================================
                              TIMING RULES
==============================================

*** CRITICAL FOR AUDIO-VIDEO SYNC: ***
- Speaking rate: approximately 2.5 words per second
- Each self.wait(N) in code = N seconds of speaking time
- Each self.play() takes about 1 second
- Your script length MUST match the video duration

*** PACING: ***
- When title appears → speak 2-3 sentences about the topic
- When explanation text appears → elaborate on that specific point
- When graph/visual appears → describe what it represents mathematically
- When formula appears → explain what each part means
- At the end → brief concluding thought (1 sentence)

==============================================
                            CONTENT RULES
==============================================

*** DO: ***
- Teach the concept - explain WHY and HOW
- When formula shows: "The area formula pi r squared tells us..."
- When graph shows: "This parabola represents how..."
- Use simple, clear language a student would understand
- Sound like an enthusiastic teacher

*** DO NOT: ***
- "We see a circle appearing" (describing visuals)
- "Hello everyone" or "Welcome" (greetings)
- "In this video" or "Today we'll learn" (intros)
- "So to summarize" or "In conclusion" (summaries)
- "As you can see" or "Notice how" (visual references)
- Any code, any technical syntax, any warnings

==============================================

Now write ONLY the narration text. Start speaking immediately about the topic."""
    
    # Use the selected model
    if model_choice == "Mistral-7B (Finetuned)":
        if not RAG_FINETUNED_AVAILABLE:
            # Fall back to CodeLlama
            model_choice = "CodeLlama-34B"
        else:
            try:
                pipeline = get_rag_finetuned_cached()
                if pipeline is None:
                    model_choice = "CodeLlama-34B"
                else:
                    script = pipeline.generate_manim_code(teaching_prompt, use_rag=False)
                    return True, script
            except Exception as e:
                logger.warning(f"Finetuned model failed: {e}")
                model_choice = "CodeLlama-34B"
    
    # Use Groq API
    if model_choice in ["CodeLlama-34B", "Phi-2"]:
        if not GROQ_AVAILABLE:
            return False, "Groq client not available"
        
        try:
            model_type = "codellama" if model_choice == "CodeLlama-34B" else "phi2"
            groq_client = get_groq_client_cached(model_type)
            
            if groq_client is None:
                return False, "Failed to initialize Groq client"
            
            script = groq_client.generate(teaching_prompt, max_tokens=1024, temperature=0.8)
            return True, script
            
        except Exception as e:
            return False, f"Error generating script: {str(e)}"
    
    return False, f"Unknown model: {model_choice}"


def generate_tts_audio_file(text: str) -> Optional[str]:
    if not text or not text.strip():
        return None
    
    # Clean the text - remove any code, warnings, or technical content that shouldn't be read
    cleaned_text = clean_text_for_tts(text)
    
    if not cleaned_text or not cleaned_text.strip():
        return None
    
    try:
        tts = gTTS(text=cleaned_text, lang="en", slow=False)
        
        # Save to outputs directory
        output_dir = os.path.join(PROJECT_ROOT, "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        # Use temp file first, then move
        with NamedTemporaryFile(delete=False, suffix=".mp3", dir=output_dir) as fp:
            temp_path = fp.name
            tts.save(temp_path)
        
        return temp_path
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None


def get_media_duration(file_path: str) -> float:                                      
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        logger.error(f"Error getting duration: {e}")
    return 0.0


def merge_video_with_audio(video_path: str, audio_text: str) -> tuple:
    """
    Merge a video with generated audio narration.
    Adjusts video speed to match audio length for perfect sync.
    
    Returns: (success, final_video_path or error_message)
    """
    if not os.path.exists(video_path):
        return False, "Video file not found"
    
    if not audio_text or not audio_text.strip():
        return False, "No audio text provided"
    
    # Clean the audio text - remove any code, warnings, or technical content
    cleaned_audio_text = clean_text_for_tts(audio_text)
    
    if not cleaned_audio_text or not cleaned_audio_text.strip():
        return False, "No speakable content in the audio text after cleaning"
    
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate audio file
    audio_path = None
    try:
        tts = gTTS(text=cleaned_audio_text, lang="en", slow=False)
        audio_path = os.path.join(output_dir, "temp_narration.mp3")
        tts.save(audio_path)
        
        # Get durations
        video_duration = get_media_duration(video_path)
        audio_duration = get_media_duration(audio_path)
        
        if video_duration <= 0 or audio_duration <= 0:
            return False, "Could not determine media durations"
        
        logger.info(f"Video duration: {video_duration:.2f}s, Audio duration: {audio_duration:.2f}s")
        
        # Calculate speed factor to sync video with audio
        # If audio is longer, slow down video. If shorter, speed up video.
        speed_factor = video_duration / audio_duration
        
        # Limit speed adjustment to reasonable range (0.5x to 2x)
        speed_factor = max(0.5, min(2.0, speed_factor))
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        final_video_path = os.path.join(output_dir, f"{base_name}_with_audio.mp4")
        
        # Build ffmpeg command to merge video and audio with speed adjustment
        # Using setpts for video speed and atempo for audio normalization
        if abs(speed_factor - 1.0) < 0.05:
            # No significant speed change needed, just merge
            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-i", audio_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                final_video_path
            ]
        else:
            # Adjust video speed to match audio length
            pts_value = 1.0 / speed_factor
            cmd = [
                "ffmpeg",
                "-y",
                "-i", video_path,
                "-i", audio_path,
                "-filter_complex",
                f"[0:v]setpts={pts_value:.4f}*PTS[v]",
                "-map", "[v]",
                "-map", "1:a",
                "-c:v", "libx264",
                "-preset", "fast",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                final_video_path
            ]
        
        logger.info(f"Running ffmpeg: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            return False, f"FFmpeg error: {result.stderr[:500]}"
        
        if not os.path.exists(final_video_path):
            return False, "Output video was not created"
        
        logger.info(f"Merged video saved to: {final_video_path}")
        return True, final_video_path
        
    except FileNotFoundError:
        return False, "FFmpeg not installed. Please install FFmpeg."
    except subprocess.TimeoutExpired:
        return False, "Video merging timed out"
    except Exception as e:
        logger.error(f"Error merging video and audio: {e}")
        return False, f"Error: {str(e)}"
    finally:
        # Cleanup temp audio file
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass


# API Endpoints

@app.get("/")
def root():
    return {"message": "Math2Visual Backend API", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse)
def health_check():
    models_status = {}
    
    # Check RAG Finetuned model
    if RAG_FINETUNED_AVAILABLE:
        try:
            pipeline = get_rag_finetuned_cached()
            models_status["Mistral-7B (Finetuned)"] = "healthy" if pipeline else "unhealthy"
        except:
            models_status["Mistral-7B (Finetuned)"] = "unhealthy"
    else:
        models_status["Mistral-7B (Finetuned)"] = "not available"
    
    # Check Groq API
    if GROQ_AVAILABLE:
        for model_type, label in [("codellama", "CodeLlama-34B"), ("phi2", "Phi-2")]:
            try:
                client = get_groq_client_cached(model_type)
                if client:
                    status = client.get_status()
                    models_status[label] = "healthy" if status.get("api_key_set") else "no API key"
                else:
                    models_status[label] = "unhealthy"
            except:
                models_status[label] = "unhealthy"
    else:
        models_status["CodeLlama-34B"] = "not available"
        models_status["Phi-2"] = "not available"
    
    return HealthResponse(status="running", models=models_status)


@app.post("/generate_code", response_model=CodeGenerationResponse)
def generate_code_endpoint(request: CodeGenerationRequest):
    logger.info(f"Code generation request: model={request.model_choice}, prompt={request.prompt[:50]}...")
    
    success, result = generate_code_with_model(request.prompt, request.model_choice)
    
    if success:
        return CodeGenerationResponse(success=True, code=result, message="Code generated successfully")
    else:
        return CodeGenerationResponse(success=False, code=f"# Error: {result}", message=result)


@app.post("/compile_video", response_model=VideoCompilationResponse)
def compile_video_endpoint(request: VideoCompilationRequest):
    logger.info(f"Video compilation request: quality={request.quality}")
    
    success, result = compile_video(request.code, request.quality)
    
    if success:
        return VideoCompilationResponse(success=True, video_path=result)
    else:
        return VideoCompilationResponse(success=False, error_message=result)


@app.get("/video/{filename}")
def get_video(filename: str): # Download a compiled video file.
    video_path = os.path.join(PROJECT_ROOT, "outputs", filename)
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(video_path, media_type="video/mp4", filename=filename)


class VideoDurationRequest(BaseModel):
    video_path: str


@app.post("/get_video_duration")
def get_video_duration_endpoint(request: VideoDurationRequest):
    """Get the duration of a video file in seconds."""
    if not os.path.exists(request.video_path):
        return {"duration": 0.0, "error": "Video file not found"}
    
    duration = get_media_duration(request.video_path)
    return {"duration": duration}


@app.post("/generate_teaching_script", response_model=TeachingScriptResponse)
def generate_teaching_script_endpoint(request: TeachingScriptRequest):
    logger.info(f"Teaching script request: model={request.model_choice}, duration={request.video_duration}s")
    
    success, result = generate_teaching_script_text(
        request.animation_description,
        request.manim_code,
        request.model_choice,
        request.video_duration
    )
    
    if success:
        return TeachingScriptResponse(success=True, script=result, message="Script generated successfully")
    else:
        return TeachingScriptResponse(success=False, script="", message=result)


@app.post("/generate_tts", response_model=TTSResponse)
def generate_tts_endpoint(request: TTSRequest):
    logger.info(f"TTS request: text length={len(request.text)}")
    
    audio_path = generate_tts_audio_file(request.text)
    
    if audio_path:
        return TTSResponse(success=True, audio_path=audio_path, message="Audio generated successfully")
    else:
        return TTSResponse(success=False, message="Failed to generate audio")


@app.get("/audio/{filename}")
def get_audio(filename: str): # Download an audio file.
    audio_path = os.path.join(PROJECT_ROOT, "outputs", filename)
    
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio not found")
    
    return FileResponse(audio_path, media_type="audio/mp3", filename=filename)


@app.post("/merge_video_audio", response_model=MergeVideoAudioResponse)
def merge_video_audio_endpoint(request: MergeVideoAudioRequest):
    logger.info(f"Merge request: video={request.video_path}")
    
    success, result = merge_video_with_audio(request.video_path, request.audio_text)
    
    if success:
        return MergeVideoAudioResponse(
            success=True,
            final_video_path=result,
            message="Video and audio merged successfully"
        )
    else:
        return MergeVideoAudioResponse(
            success=False,
            message=result
        )


# CHATBOT ENDPOINTS - Manim Syntax Assistant

@app.post("/chatbot/ask", response_model=ChatbotResponse)
def chatbot_ask(request: ChatbotRequest):
  
    logger.info(f"Chatbot question: {request.question[:50]}...")
    
    # Check if chatbot is available
    if not CHATBOT_AVAILABLE:
        return ChatbotResponse(
            success=False,
            answer="Chatbot service is not available. Please check server logs.",
            message="CHATBOT_AVAILABLE is False"
        )
    
    try:
        # Get the chatbot instance
        chatbot = get_chatbot_cached()
        
        if chatbot is None:
            return ChatbotResponse(
                success=False,
                answer="Could not initialize the chatbot. Please try again later.",
                message="Chatbot initialization failed"
            )
        
        # Ask the question
        if request.include_examples:
            response = chatbot.ask_with_examples(request.question)
        else:
            response = chatbot.ask(request.question)
        
        return ChatbotResponse(
            success=not response.get("error", False),
            answer=response.get("answer", "No response generated"),
            sources=response.get("sources", ""),
            cached=response.get("cached", False),
            message="Success" if not response.get("error") else "Error occurred"
        )
        
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return ChatbotResponse(
            success=False,
            answer=f"An error occurred: {str(e)}",
            message=str(e)
        )


@app.get("/chatbot/syntax/{class_name}")
def chatbot_get_syntax(class_name: str):
 
    logger.info(f"Syntax lookup: {class_name}")
    
    if not CHATBOT_AVAILABLE:
        return ChatbotResponse(
            success=False,
            answer="Chatbot service is not available.",
            message="CHATBOT_AVAILABLE is False"
        )
    
    try:
        chatbot = get_chatbot_cached()
        
        if chatbot is None:
            return ChatbotResponse(
                success=False,
                answer="Could not initialize the chatbot.",
                message="Chatbot initialization failed"
            )
        
        response = chatbot.get_syntax(class_name)
        
        return ChatbotResponse(
            success=not response.get("error", False),
            answer=response.get("answer", "No information found"),
            sources=response.get("sources", ""),
            cached=response.get("cached", False)
        )
        
    except Exception as e:
        logger.error(f"Syntax lookup error: {e}")
        return ChatbotResponse(
            success=False,
            answer=f"Error looking up {class_name}: {str(e)}",
            message=str(e)
        )


@app.get("/chatbot/status")
def chatbot_status():
    
    if not CHATBOT_AVAILABLE:
        return {
            "available": False,
            "ready": False,
            "message": "Chatbot module not imported"
        }
    
    chatbot = get_chatbot_cached()
    
    return {
        "available": True,
        "ready": chatbot is not None and chatbot.is_ready,
        "message": "Chatbot ready" if chatbot and chatbot.is_ready else "Chatbot not initialized"
    }


@app.get("/available_models")
def get_available_models():
    models = []
    
    # Check GPU for finetuned model
    try:
        import torch
        if torch.cuda.is_available() and RAG_FINETUNED_AVAILABLE:
            models.append("Mistral-7B (Finetuned)")
    except:
        pass
    
    # Groq models are always available if API key is set
    if GROQ_AVAILABLE:
        models.append("CodeLlama-34B")
        models.append("Phi-2")
    
    return {"models": models}


# Run the server

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
