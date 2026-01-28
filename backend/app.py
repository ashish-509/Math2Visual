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
import logging
import tempfile
import shutil
import subprocess
from typing import Optional
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
            return True, code
            
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
CRITICAL LAYOUT RULES - MUST FOLLOW:
1. TEXT SIZING: Title .scale(0.6), main text .scale(0.5), equations .scale(0.6), labels .scale(0.4)
2. LINE LENGTH: Max 40 chars per line. Use "\\n" for longer text.
3. FRAME SAFE ZONE: -6 to +6 horizontal, -3.5 to +3.5 vertical
4. PREVENT OVERLAP: ALWAYS FadeOut previous content BEFORE showing new content
5. PATTERN: FadeOut old -> Create new scaled -> Write/FadeIn new"""
            
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
            return True, code
            
        except Exception as e:
            return False, f"Groq API error: {str(e)}"
    
    return False, f"Unknown model choice: {model_choice}"


def compile_video(code: str, quality: str = "medium") -> tuple:
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
    teaching_prompt = f"""You are a math teacher giving a live lesson. The animation shows visuals while YOU TEACH the concept.

Topic: "{description}"

Animation code (this shows WHEN things appear - sync your teaching to these moments):
```python
{code}
```
{duration_instruction}

TEACHING RULES:
1. NO GREETING, NO SUMMARY - teach from first second to last
2. DO NOT describe visuals ("we see a circle") - TEACH the concept ("a circle is a shape where every point is the same distance from the center")
3. When a formula appears, EXPLAIN what it means, don't just read it
4. When a shape appears, TEACH why it matters to the concept
5. When something transforms, EXPLAIN the mathematical relationship

WHAT TO DO:
- Circle appears → "A circle is defined by all points equidistant from a center point"
- Formula A=πr² appears → "The area equals pi times the radius squared - pi is about 3.14"
- Triangle transforms to rectangle → "Notice how we can rearrange these pieces - the area stays the same"

WHAT NOT TO DO:
- "We see a circle on the screen" 
- "Now the formula appears" 
- "The text says..." 
- "Hello everyone" 
- "So to summarize" 

TIMING:
- Follow the order of animations in the code
- Pace your explanation to match self.wait() durations
- ~2.5 words per second of video

STYLE:
- Talk like a passionate teacher in a classroom
- Explain WHY, not just WHAT
- Use analogies from everyday life
- Simple words for beginners

Write ONLY the teaching narration. No timestamps, no brackets, no stage directions."""
    
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
    
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        
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
    
    output_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate audio file
    audio_path = None
    try:
        tts = gTTS(text=audio_text, lang="en", slow=False)
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
