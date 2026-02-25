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
import asyncio
from typing import Optional, Tuple
from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor

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

# Documentation sync (for fetching latest Manim docs from GitHub)
try:
    from crawler.docs_sync import get_docs_sync, sync_manim_docs
    DOCS_SYNC_AVAILABLE = True
except ImportError as e:
    DOCS_SYNC_AVAILABLE = False
    logger.warning(f"Docs sync not available: {e}")


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


# Thread pool for running blocking operations without blocking the event loop
executor = ThreadPoolExecutor(max_workers=4)


# Startup event - preload models to eliminate cold start delay
@app.on_event("startup")
async def preload_models():
    # Load all ML models at startup instead of on first request. This runs in parallel to speed up initialization.
    logger.info("Starting model preloading...")
    
    loop = asyncio.get_event_loop()
    
    # Define preloading tasks
    def load_rag():
        if RAG_AVAILABLE:
            try:
                get_rag_pipeline_cached()
                logger.info("RAG pipeline loaded")
            except Exception as e:
                logger.warning(f"RAG preload failed: {e}")
    
    def load_groq_codellama():
        if GROQ_AVAILABLE:
            try:
                get_groq_client_cached("codellama")
                logger.info("Groq CodeLlama client loaded")
            except Exception as e:
                logger.warning(f"Groq CodeLlama preload failed: {e}")
    
    def load_groq_phi2():
        if GROQ_AVAILABLE:
            try:
                get_groq_client_cached("phi2")
                logger.info("Groq Phi-2 client loaded")
            except Exception as e:
                logger.warning(f"Groq Phi-2 preload failed: {e}")
    
    def load_chatbot():
        if CHATBOT_AVAILABLE:
            try:
                get_chatbot_cached()
                logger.info("Chatbot loaded")
            except Exception as e:
                logger.warning(f"Chatbot preload failed: {e}")
    
    # Run all preloading tasks in parallel
    preload_tasks = [load_rag, load_groq_codellama, load_groq_phi2, load_chatbot]
    await asyncio.gather(*[loop.run_in_executor(executor, task) for task in preload_tasks])
    
    logger.info("Model preloading complete!")



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


# Docs sync response
class DocsSyncResponse(BaseModel):
    success: bool
    message: str
    last_sync: Optional[str] = None
    last_commit: Optional[str] = None
    files_synced: int = 0
    update_available: bool = False


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


# Smart Generation with Retry - Request/Response Models
class SmartGenerateRequest(BaseModel):
    prompt: str
    model_choice: str = "CodeLlama-34B"
    quality: str = "medium"
    max_retries: int = 3


class SmartGenerateResponse(BaseModel):
    success: bool
    video_path: Optional[str] = None
    code: str = ""
    attempts: int = 0
    message: str = ""


# Error-feedback regeneration — Request/Response Models
class RegenerateAndCompileRequest(BaseModel):
    prompt: str                        
    current_code: str                 
    error_message: str              
    model_choice: str = "CodeLlama-34B" 
    quality: str = "medium"            
    max_retries: int = 2               


class RegenerateAndCompileResponse(BaseModel):
    success: bool
    video_path: Optional[str] = None 
    code: str = ""                    # The final (fixed) code that was compiled
    message: str = ""                 # Human-readable result summary


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


def check_latex_available() -> bool:
    """Check if LaTeX (pdflatex / latex) is installed and accessible on the system."""
    for latex_cmd in ["latex", "pdflatex", "xelatex"]:
        try:
            result = subprocess.run(
                [latex_cmd, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False


def convert_mathtex_to_text(code: str) -> str:
    """
    Automatically convert MathTex() and Tex() calls into plain Text() calls.

    This is a fallback when LaTeX is NOT installed on the system.
    It strips LaTeX markup and converts it to readable plain text so the
    animation can still render without needing a LaTeX installation.
    """
    if not code:
        return code

    def clean_latex_string(s: str) -> str:
        # Strip LaTeX commands and return a readable plain-text equivalent.
        
        # Remove raw string prefix characters that might appear in the extracted string
        s = s.replace("\\\\", "DOUBLE_BACKSLASH_PLACEHOLDER")

        # Fractions: \frac{a}{b} -> a/b
        s = re.sub(r'\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}', r'\1/\2', s)

        # Square root: \sqrt{x} -> sqrt(x)
        s = re.sub(r'\\sqrt\s*\{([^{}]*)\}', r'sqrt(\1)', s)
        s = re.sub(r'\\sqrt', 'sqrt', s)

        # \text{...} -> ...
        s = re.sub(r'\\text\s*\{([^{}]*)\}', r'\1', s)

        # Greek letters
        greek = {
            'alpha': 'alpha', 'beta': 'beta', 'gamma': 'gamma', 'delta': 'delta',
            'epsilon': 'epsilon', 'zeta': 'zeta', 'eta': 'eta', 'theta': 'theta',
            'iota': 'iota', 'kappa': 'kappa', 'lambda': 'lambda', 'mu': 'mu',
            'nu': 'nu', 'xi': 'xi', 'pi': 'pi', 'rho': 'rho', 'sigma': 'sigma',
            'tau': 'tau', 'upsilon': 'upsilon', 'phi': 'phi', 'chi': 'chi',
            'psi': 'psi', 'omega': 'omega',
            'Alpha': 'Alpha', 'Beta': 'Beta', 'Gamma': 'Gamma', 'Delta': 'Delta',
            'Theta': 'Theta', 'Lambda': 'Lambda', 'Pi': 'Pi', 'Sigma': 'Sigma',
            'Phi': 'Phi', 'Psi': 'Psi', 'Omega': 'Omega',
        }
        for latex_name, plain_name in greek.items():
            s = re.sub(rf'\\{latex_name}\b', plain_name, s)

        # Trig and math functions
        for fn in ['sin', 'cos', 'tan', 'sec', 'csc', 'cot',
                   'arcsin', 'arccos', 'arctan',
                   'log', 'ln', 'exp', 'lim', 'max', 'min',
                   'sum', 'prod', 'int', 'oint']:
            s = re.sub(rf'\\{fn}\b', fn, s)

        # Symbols
        symbol_map = {
            r'\\infty': 'infinity',
            r'\\times': 'x',
            r'\\cdot': '.',
            r'\\pm': '+/-',
            r'\\mp': '-/+',
            r'\\leq': '<=',
            r'\\geq': '>=',
            r'\\neq': '!=',
            r'\\approx': '~',
            r'\\equiv': '=',
            r'\\rightarrow': '->',
            r'\\leftarrow': '<-',
            r'\\Rightarrow': '=>',
            r'\\Leftarrow': '<=',
            r'\\leftrightarrow': '<->',
            r'\\partial': 'd',
            r'\\nabla': 'del',
            r'\\ldots': '...',
            r'\\cdots': '...',
        }
        for pattern, repl in symbol_map.items():
            s = re.sub(pattern, repl, s)

        # \left and \right delimiters
        s = re.sub(r'\\left\s*[\[({|]', '(', s)
        s = re.sub(r'\\right\s*[\])}|]', ')', s)

        # Superscripts/subscripts: ^{expr} -> ^expr, _{expr} -> _expr
        s = re.sub(r'\^\{([^{}]*)\}', r'^\1', s)
        s = re.sub(r'_\{([^{}]*)\}', r'_\1', s)

        # Remove leftover curly braces from LaTeX grouping
        s = s.replace('{', '').replace('}', '')

        # Remove remaining backslashes
        s = re.sub(r'\\(\w+)', r'\1', s)
        s = s.replace('\\', '')

        # Restore placeholder
        s = s.replace("DOUBLE_BACKSLASH_PLACEHOLDER", "")

        # Clean up whitespace
        s = re.sub(r'\s+', ' ', s).strip()

        # Limit length to avoid text overflow on screen
        if len(s) > 55:
            s = s[:52] + '...'

        return s

    def replace_mathtex_in_line(line: str) -> str:
        """Replace all MathTex(...) and Tex(...) calls in a single line."""
        # Match: MathTex(r"...") or MathTex("...") or Tex(r"...") or Tex("...")
        # The regex uses a non-greedy match for the string content.
        # It handles both single and double quotes.
        def replacer(m):
            latex_content = m.group(2)  # Content inside the quotes
            plain_text = clean_latex_string(latex_content)
            # Escape any double quotes in the plain text for safety
            plain_text = plain_text.replace('"', "'")
            return f'Text("{plain_text}"'

        # Pattern: (MathTex|Tex)(  r?"content"  -- captures content inside quotes
        line = re.sub(
            r'\b(MathTex|Tex)\s*\(\s*r?"((?:[^"\\]|\\.)*)"',
            replacer,
            line
        )
        # Same for single-quoted strings
        line = re.sub(
            r"\b(MathTex|Tex)\s*\(\s*r?'((?:[^'\\]|\\.)*)'",
            lambda m: f'Text("{clean_latex_string(m.group(2).replace(chr(34), chr(39)))}"',
            line
        )
        return line

    lines = code.split('\n')
    converted_lines = []
    for line in lines:
        # Only process lines that actually contain MathTex or Tex calls
        if re.search(r'\b(MathTex|Tex)\s*\(', line):
            line = replace_mathtex_in_line(line)
        converted_lines.append(line)

    converted = '\n'.join(converted_lines)
    if converted != code:
        logger.info("convert_mathtex_to_text: replaced MathTex/Tex with Text() (LaTeX not available)")
    return converted


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
            
            # Build augmented prompt with RAG context
            if context:
                augmented_prompt = f"""Use the following Manim documentation as reference:

=== MANIM DOCUMENTATION ===
{context}
=== END DOCUMENTATION ===

User Request: {prompt}

Generate complete, working Manim code based on the documentation above."""
            else:
                augmented_prompt = f"""

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
    
    # Quality flags - use low quality for faster preview, medium/high for final
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
            "--media_dir", temp_dir,
            "--disable_caching", "False",  # Enable Manim's internal caching for faster reruns
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

            # Detect if the failure is because 'latex' binary is missing on this system.
            # This happens whenever the generated code uses MathTex() or Tex(), which both
            # need a full LaTeX installation to render equations as SVG.
            is_latex_error = (
                ("FileNotFoundError" in error_msg and "latex" in error_msg.lower()) or
                ("No such file or directory" in error_msg and "latex" in error_msg.lower())
            )

            if is_latex_error:
                logger.info(
                    "Compilation failed because LaTeX is not installed. "
                    "Attempting automatic MathTex -> Text() conversion and retrying..."
                )
                fixed_code = convert_mathtex_to_text(code)

                if fixed_code != code:
                    # Rewrite the temp script file with the auto-fixed code
                    with open(script_path, "w", encoding="utf-8") as f:
                        f.write(fixed_code)
                    code = fixed_code  # Keep track so we copy the right version later

                    logger.info("Retrying Manim compilation after MathTex to Text() conversion...")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                    if result.stdout:
                        logger.info(f"Auto-fix retry stdout: {result.stdout}")
                    if result.stderr:
                        logger.warning(f"Auto-fix retry stderr: {result.stderr}")

                    if result.returncode != 0:
                        retry_error = result.stderr or result.stdout or "Unknown error"
                        return False, (
                            "LaTeX is not installed on this system.\n"
                            "MathTex calls were automatically converted to Text() but "
                            "compilation still failed. Please regenerate without MathTex.\n\n"
                            f"Error details:\n{retry_error}"
                        )
                    # Auto-fix succeeded — fall through to the video-finding logic below
                else:
                    # The code had no MathTex/Tex to convert, so we can't auto-fix it.
                    return False, (
                        "LaTeX is not installed on this system (the 'latex' binary was not found).\n"
                        "Please avoid MathTex() and Tex() — use Text() for all text and math.\n\n"
                        f"Original error:\n{error_msg}"
                    )
            else:
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


def generate_and_compile_with_retry(prompt: str, model_choice: str, quality: str = "medium", 
                                    max_retries: int = 3) -> tuple:
    # Generate Manim code and compile it to video, with automatic retry on errors.

    # Get RAG context once (we'll reuse it for retries)
    context = ""
    if RAG_AVAILABLE:
        rag = get_rag_pipeline_cached()
        if rag and rag.is_indexed:
            context = rag.retrieve_context(prompt)
    
    # Track attempts for logging
    attempt = 0
    last_error = ""
    current_code = ""
    
    while attempt < max_retries:
        attempt += 1
        logger.info(f"Attempt {attempt}/{max_retries} for prompt: {prompt[:50]}...")
        
        # Step 1: Generate code (or regenerate with error feedback)
        if attempt == 1:
            # First attempt: normal generation
            success, current_code = generate_code_with_model(prompt, model_choice)
        else:
            # Retry: regenerate using previous error as feedback
            success, current_code = regenerate_code_with_error(
                prompt, model_choice, context, last_error
            )
        
        if not success:
            logger.warning(f"Code generation failed on attempt {attempt}: {current_code}")
            last_error = current_code
            continue
        
        # Step 2: Try to compile the code
        compile_success, compile_result = compile_video(current_code, quality)
        
        if compile_success:
            # Success! Return the video path and final code
            logger.info(f"Successfully compiled on attempt {attempt}")
            return True, compile_result, current_code
        
        # Compilation failed - save the error for feedback
        last_error = compile_result
        logger.warning(f"Compilation failed on attempt {attempt}: {last_error[:200]}...")
    
    # All retries exhausted
    error_msg = f"Failed after {max_retries} attempts. Last error: {last_error}"
    logger.error(error_msg)
    return False, error_msg, current_code


def regenerate_code_with_error(original_prompt: str, model_choice: str, 
                               context: str, error_message: str) -> tuple:
    # Regenerate code using the error from a failed compilation attempt.

    logger.info("Regenerating code with error feedback...")

    # Detect if the error is caused by LaTeX not being installed on this system.
    # This is signalled by a FileNotFoundError for the 'latex' binary inside the error.
    is_latex_missing_error = (
        ("FileNotFoundError" in error_message and "latex" in error_message.lower()) or
        ("No such file or directory" in error_message and "latex" in error_message.lower()) or
        # Also catch our own descriptive messages from compile_video()
        ("LaTeX is not installed" in error_message)
    )

    # When LaTeX is not available, we augment the error with an explicit instruction
    # so the model stops using MathTex/Tex and switches to Text() instead.
    if is_latex_missing_error:
        logger.info(
            "LaTeX missing error detected — injecting 'no MathTex' constraint into regeneration prompt"
        )
        latex_constraint = (
            "\n\n"
            "===========================================================\n"
            "CRITICAL SYSTEM CONSTRAINT — LaTeX is NOT installed here:\n"
            "===========================================================\n"
            "- DO NOT use MathTex() anywhere — it requires a LaTeX compiler.\n"
            "- DO NOT use Tex() anywhere — it also requires LaTeX.\n"
            "- Use ONLY Text() for every piece of text, including formulas.\n"
            "- For equations, write them in plain text inside Text():\n"
            "    Text('sin(theta) = opp/hyp').scale(0.4)\n"
            "    Text('E = mc^2').scale(0.4)\n"
            "    Text('a^2 + b^2 = c^2').scale(0.4)\n"
            "    Text('f(x) = x^2 + 2x + 1').scale(0.4)\n"
            "    Text('pi = 3.14159').scale(0.4)\n"
            "- Subscripts/superscripts: write as plain strings — 'x_0', 'a_n', 'x^2'\n"
            "===========================================================\n"
        )
        # Prepend the constraint so the model sees it first
        error_message = latex_constraint + error_message

    # Use Groq client for regeneration
    if model_choice in ["CodeLlama-34B", "Phi-2"]:
        if not GROQ_AVAILABLE:
            return False, "Groq client not available for regeneration"
        
        try:
            model_type = "codellama" if model_choice == "CodeLlama-34B" else "phi2"
            groq_client = get_groq_client_cached(model_type)
            
            if groq_client is None:
                return False, "Failed to get Groq client"
            
            # Call the regenerate method with error feedback
            code = groq_client.regenerate_with_error(
                original_prompt=original_prompt,
                context=context,
                error_message=error_message,
                max_tokens=2048,
                temperature=0.4
            )
            
            # Validate and fix the regenerated code
            is_valid, fixed_code, validation_error = validate_and_fix_manim_code(code)
            
            if not is_valid:
                return False, f"Regenerated code validation failed: {validation_error}"
            
            return True, fixed_code
            
        except Exception as e:
            return False, f"Error during regeneration: {str(e)}"
    
    # For finetuned model, use a combined prompt approach
    elif model_choice == "Mistral-7B (Finetuned)":
        if not RAG_FINETUNED_AVAILABLE:
            return False, "Finetuned model not available"
        
        try:
            pipeline = get_rag_finetuned_cached()
            if pipeline is None:
                return False, "Could not get finetuned pipeline"
            
            # Build error feedback prompt
            error_prompt = f"""{original_prompt}

IMPORTANT: The previous attempt failed with this error:
{error_message}

Please generate corrected code that fixes this error."""
            
            code = pipeline.generate_manim_code(error_prompt, use_rag=True)
            
            is_valid, fixed_code, validation_error = validate_and_fix_manim_code(code)
            if not is_valid:
                return False, f"Regenerated code validation failed: {validation_error}"
            
            return True, fixed_code
            
        except Exception as e:
            return False, f"Error during finetuned regeneration: {str(e)}"
    
    return False, f"Unknown model for regeneration: {model_choice}"



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
            
            # Use generate_narration() NOT generate() - this uses a narration-specific prompt
            script = groq_client.generate_narration(teaching_prompt, max_tokens=1024, temperature=0.8)
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


def get_available_video_encoder() -> str:
    # Detect available video encoder on the system. Returns the best available encoder for re-encoding.
    # List of encoders to try, in order of preference
    encoders_to_try = ["libx264", "h264", "libx265", "mpeg4"]
    
    for encoder in encoders_to_try:
        try:
            # Test if encoder is available
            cmd = ["ffmpeg", "-hide_banner", "-encoders"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if encoder in result.stdout:
                logger.info(f"Found video encoder: {encoder}")
                return encoder
        except:
            pass
    
    # Default fallback - mpeg4 is always available in FFmpeg
    logger.warning("No preferred encoder found, using mpeg4")
    return "mpeg4"


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
        
        # Get durations using ffprobe
        video_duration = get_media_duration(video_path)
        audio_duration = get_media_duration(audio_path)
        
        logger.info(f"Video duration: {video_duration:.2f}s, Audio duration: {audio_duration:.2f}s")
        
        if video_duration <= 0:
            return False, f"Could not determine video duration. Video path: {video_path}"
        
        if audio_duration <= 0:
            return False, "Could not determine audio duration"
        
        # Calculate speed factor to sync video with audio
        speed_factor = video_duration / audio_duration
        
        # Limit speed adjustment to reasonable range (0.5x to 2x)
        speed_factor = max(0.5, min(2.0, speed_factor))
        
        logger.info(f"Speed factor: {speed_factor:.4f}")
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        final_video_path = os.path.join(output_dir, f"{base_name}_with_audio.mp4")
        
        # Build ffmpeg command 
        # First, we'll adjust video speed if needed, then add audio
        if abs(speed_factor - 1.0) < 0.05:
            # No significant speed change needed, just merge video with audio
            # Use -an to ignore any existing audio from video, then add our audio
            cmd = [
                "ffmpeg",
                "-y",                      # Overwrite output
                "-i", video_path,          # Input video
                "-i", audio_path,          # Input audio (narration)
                "-map", "0:v:0",           # Take video stream from first input
                "-map", "1:a:0",           # Take audio stream from second input
                "-c:v", "copy",            # Copy video codec (no re-encode)
                "-c:a", "aac",             # Encode audio as AAC
                "-b:a", "192k",            # Audio bitrate
                final_video_path
            ]
        else:
            # Need to adjust video speed to match audio length
            # Detect available encoder since libx264 may not be installed
            video_encoder = get_available_video_encoder()
            pts_value = 1.0 / speed_factor
            
            # Build encoder-specific options
            if video_encoder in ["libx264", "h264"]:
                encoder_opts = ["-c:v", video_encoder, "-preset", "fast", "-crf", "23"]
            elif video_encoder == "libx265":
                encoder_opts = ["-c:v", video_encoder, "-preset", "fast", "-crf", "28"]
            else:
                # mpeg4 or other fallback
                encoder_opts = ["-c:v", video_encoder, "-q:v", "5"]
            
            cmd = [
                "ffmpeg",
                "-y",                      # Overwrite output
                "-i", video_path,          # Input video
                "-i", audio_path,          # Input audio (narration)
                "-filter_complex",
                f"[0:v]setpts={pts_value:.4f}*PTS[outv]",  # Adjust video speed
                "-map", "[outv]",          # Use the filtered video
                "-map", "1:a:0",           # Take audio from second input
            ] + encoder_opts + [
                "-c:a", "aac",             # Encode audio as AAC
                "-b:a", "192k",            # Audio bitrate
                final_video_path
            ]
        
        logger.info(f"Running ffmpeg: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            # Extract the actual error from stderr (skip the version/config header)
            stderr = result.stderr
            # Look for actual error lines (usually after "configuration:" lines)
            error_lines = []
            capture = False
            for line in stderr.split('\n'):
                # Skip empty lines and version/config info
                if not line.strip():
                    continue
                if line.strip().startswith('configuration:'):
                    capture = True  # Start capturing after config
                    continue
                if capture or 'error' in line.lower() or 'invalid' in line.lower():
                    error_lines.append(line.strip())
            
            actual_error = '\n'.join(error_lines[-10:]) if error_lines else stderr[-1000:]
            logger.error(f"FFmpeg failed with return code {result.returncode}")
            logger.error(f"FFmpeg stderr: {actual_error}")
            return False, f"FFmpeg error: {actual_error}"
        
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
async def generate_code_endpoint(request: CodeGenerationRequest):
    logger.info(f"Code generation request: model={request.model_choice}, prompt={request.prompt[:50]}...")
    
    # Run in thread pool to avoid blocking the event loop
    success, result = await asyncio.to_thread(
        generate_code_with_model, request.prompt, request.model_choice
    )
    
    if success:
        return CodeGenerationResponse(success=True, code=result, message="Code generated successfully")
    else:
        return CodeGenerationResponse(success=False, code=f"# Error: {result}", message=result)


@app.post("/compile_video", response_model=VideoCompilationResponse)
async def compile_video_endpoint(request: VideoCompilationRequest):
    logger.info(f"Video compilation request: quality={request.quality}")
    
    # Run in thread pool - Manim compilation is CPU-bound
    success, result = await asyncio.to_thread(
        compile_video, request.code, request.quality
    )
    
    if success:
        return VideoCompilationResponse(success=True, video_path=result)
    else:
        return VideoCompilationResponse(success=False, error_message=result)


@app.post("/smart_generate", response_model=SmartGenerateResponse)
async def smart_generate_endpoint(request: SmartGenerateRequest):
    # If compilation fails, it automatically retries by sending the error back to the LLM for correction.

    logger.info(f"Smart generate request: model={request.model_choice}, retries={request.max_retries}")
    
    # Validate max_retries (keep it reasonable)
    max_retries = min(max(1, request.max_retries), 5)
    
    # Run in thread pool - this involves LLM calls and subprocess compilation
    success, result, final_code = await asyncio.to_thread(
        generate_and_compile_with_retry,
        request.prompt,
        request.model_choice,
        request.quality,
        max_retries
    )
    
    if success:
        return SmartGenerateResponse(
            success=True,
            video_path=result,
            code=final_code,
            attempts=max_retries,  # Will be refined in future to track actual attempts
            message="Video generated successfully"
        )
    else:
        return SmartGenerateResponse(
            success=False,
            code=final_code,
            message=result
        )


@app.post("/regenerate_and_compile", response_model=RegenerateAndCompileResponse)
async def regenerate_and_compile_endpoint(request: RegenerateAndCompileRequest):
    # Error-feedback regeneration endpoint.
    logger.info(
        f"Regenerate-and-compile request: model={request.model_choice}, "
        f"quality={request.quality}, max_retries={request.max_retries}"
    )

    # Clamp retries to a sensible range (1 – 3)
    max_retries = min(max(1, request.max_retries), 3)

    # Fetch RAG documentation context once — reused across all retry attempts
    context = ""
    if RAG_AVAILABLE:
        rag = get_rag_pipeline_cached()
        if rag and rag.is_indexed:
            context = rag.retrieve_context(request.prompt)

    last_error = request.error_message
    current_code = request.current_code

    for attempt in range(max_retries):
        logger.info(f"Regeneration attempt {attempt + 1}/{max_retries}")

        # Step A: LaTeX quick-fix — try converting MathTex to Text() first.  
        is_latex_error = (
            ("FileNotFoundError" in last_error and "latex" in last_error.lower()) or
            ("No such file or directory" in last_error and "latex" in last_error.lower()) or
            ("LaTeX is not installed" in last_error)
        )

        if is_latex_error:
            fixed_code = convert_mathtex_to_text(current_code)
            if fixed_code != current_code:
                logger.info("Attempt: compiling LaTeX-auto-fixed code...")
                compile_ok, compile_result = await asyncio.to_thread(
                    compile_video, fixed_code, request.quality
                )
                if compile_ok:
                    logger.info("LaTeX auto-fix succeeded!")
                    return RegenerateAndCompileResponse(
                        success=True,
                        video_path=compile_result,
                        code=fixed_code,
                        message="Fixed by automatically converting MathTex to Text() (LaTeX not available on this system)"
                    )
                # Auto-fix didn't fully solve it — continue with LLM regeneration
                current_code = fixed_code
                last_error = compile_result

        # Step B: LLM regeneration with the error as feedback.
        regen_ok, new_code = await asyncio.to_thread(
            regenerate_code_with_error,
            request.prompt,
            request.model_choice,
            context,
            last_error
        )

        if not regen_ok:
            logger.warning(f"LLM regeneration failed on attempt {attempt + 1}: {new_code[:200]}")
            last_error = new_code
            continue

        current_code = new_code

        # Try to compile the freshly generated code
        compile_ok, compile_result = await asyncio.to_thread(
            compile_video, current_code, request.quality
        )
        if compile_ok:
            logger.info(f"Compilation succeeded on regeneration attempt {attempt + 1}")
            return RegenerateAndCompileResponse(
                success=True,
                video_path=compile_result,
                code=current_code,
                message=f"Code regenerated and compiled successfully (attempt {attempt + 1})"
            )

        # Still failing — save the error for the next iteration
        last_error = compile_result
        logger.warning(
            f"Compilation still failed after regeneration attempt {attempt + 1}: "
            f"{last_error[:200]}..."
        )

    # All retries exhausted
    logger.error(f"Regeneration failed after {max_retries} attempts. Last error: {last_error[:200]}")
    return RegenerateAndCompileResponse(
        success=False,
        code=current_code,
        message=f"Failed after {max_retries} regeneration attempt(s). Last error: {last_error}"
    )


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
async def generate_teaching_script_endpoint(request: TeachingScriptRequest):
    logger.info(f"Teaching script request: model={request.model_choice}, duration={request.video_duration}s")
    
    # Run in thread pool - involves LLM call
    success, result = await asyncio.to_thread(
        generate_teaching_script_text,
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
async def generate_tts_endpoint(request: TTSRequest):
    logger.info(f"TTS request: text length={len(request.text)}")
    
    # Run in thread pool - network call to Google TTS
    audio_path = await asyncio.to_thread(generate_tts_audio_file, request.text)
    
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
async def merge_video_audio_endpoint(request: MergeVideoAudioRequest):
    logger.info(f"Merge request: video={request.video_path}")
    
    # Run in thread pool - FFmpeg subprocess
    success, result = await asyncio.to_thread(
        merge_video_with_audio, request.video_path, request.audio_text
    )
    
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
async def chatbot_ask(request: ChatbotRequest):
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
        
        # Run chatbot query in thread pool - involves RAG retrieval and LLM call
        if request.include_examples:
            response = await asyncio.to_thread(chatbot.ask_with_examples, request.question)
        else:
            response = await asyncio.to_thread(chatbot.ask, request.question)
        
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

                                              
# Documentation Sync Endpoints

@app.post("/docs/refresh", response_model=DocsSyncResponse)
def refresh_documentation(force: bool = False):
    """Sync documentation from GitHub."""
    if not DOCS_SYNC_AVAILABLE:
        return DocsSyncResponse(success=False, message="Docs sync not available")
    
    try:
        success, msg = sync_manim_docs(force=force)
        status = get_docs_sync().get_status()
        
        if success:
            global _rag_pipeline_cache
            _rag_pipeline_cache = None
        
        return DocsSyncResponse(
            success=success, message=msg,
            last_sync=status.get("last_sync"),
            last_commit=status.get("last_commit"),
            files_synced=status.get("files_synced", 0),
            update_available=status.get("update_available", False)
        )
    except Exception as e:
        return DocsSyncResponse(success=False, message=str(e))


@app.get("/docs/status", response_model=DocsSyncResponse)
def get_docs_status():
    """Get documentation sync status."""
    if not DOCS_SYNC_AVAILABLE:
        return DocsSyncResponse(success=False, message="Docs sync not available")
    
    try:
        status = get_docs_sync().get_status()
        return DocsSyncResponse(
            success=True, message="OK",
            last_sync=status.get("last_sync"),
            last_commit=status.get("last_commit"),
            files_synced=status.get("files_synced", 0),
            update_available=status.get("update_available", False)
        )
    except Exception as e:
        return DocsSyncResponse(success=False, message=str(e))


@app.get("/docs/check_update")
def check_docs_update():
    """Check if new docs are available."""
    if not DOCS_SYNC_AVAILABLE:
        return {"update_available": False, "error": "Not available"}
    
    try:
        sync = get_docs_sync()
        has_update = sync.check_for_updates()
        return {
            "update_available": has_update,
            "last_commit": sync.metadata.get("last_commit"),
            "message": "Update available" if has_update else "Up to date"
        }
    except Exception as e:
        return {"update_available": False, "error": str(e)}


# Run the server

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
