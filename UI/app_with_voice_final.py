import streamlit as st
import os
import requests
import speech_recognition as sr
from dotenv import load_dotenv
from gtts import gTTS
from tempfile import NamedTemporaryFile
import torch
import sys
import logging
import subprocess
import re
import tempfile
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to Python path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import our custom LLM client for multi-model support
try:
    from src.llm.client import LLMClient
    llm_import_success = True
except ImportError as e:
    llm_import_success = False
    llm_import_error = str(e)

# Import the RAG + Finetuned Model pipeline
try:
    from src.pipeline.rag_finetuned_pipeline import get_rag_finetuned_pipeline
    rag_finetuned_import_success = True
except ImportError as e:
    rag_finetuned_import_success = False
    rag_finetuned_import_error = str(e)

# Import Groq client for CodeLlama API
try:
    from src.llm.groq_client import get_groq_client
    groq_import_success = True
except ImportError as e:
    groq_import_success = False
    groq_import_error = str(e)

# Import transformers pipeline
try:
    from transformers import pipeline
    transformers_import_success = True
except ImportError as e:
    transformers_import_success = False
    transformers_import_error = str(e)

# Load environment variables
load_dotenv()

# Configuration settings
STT_MODEL_ID = os.getenv("M2V_STT_MODEL", "distil-whisper/distil-medium.en")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set up the page
st.set_page_config(
    page_title="Math2Visual",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme styling
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stButton>button {
        width: 100%;
        border-radius: 4px;
        height: 3em;
        font-weight: 600;
    }
    .stTextInput>div>div>input {
        color: #FAFAFA;
    }
    /* Hide default Streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Load the speech recognition model once and cache it
@st.cache_resource
def load_speech_recognition_model():
    """
    Load the Whisper model for speech-to-text.
    This is cached so we don't reload it every time.
    """
    if not transformers_import_success:
        st.error(f"Transformers library not available: {transformers_import_error}")
        return None
    
    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=STT_MODEL_ID,
            device=DEVICE
        )
        return pipe
    except Exception as e:
        st.error(f"Error loading speech model: {e}")
        return None

# Create and cache the LLM client
@st.cache_resource
def get_llm_client(preferred_model='mistral'):
    """
    This handles switching between different AI models automatically.
    """
    return LLMClient(preferred_model=preferred_model)

# Create and cache the RAG pipeline (for retrieving Manim documentation)
@st.cache_resource
def get_standalone_rag():
    """
    Get a standalone RAG pipeline for retrieving Manim documentation context.
    This is used with Groq API models (CodeLlama, Phi-2).
    """
    try:
        from src.rag.rag_pipeline import create_rag_pipeline
        rag = create_rag_pipeline()
        return rag
    except Exception as e:
        logger.error(f"Failed to create RAG pipeline: {e}")
        return None

# Create and cache the RAG + Finetuned Model pipeline
@st.cache_resource
def get_rag_pipeline():
    """
    RAG retrieves relevant Manim documentation, then the finetuned model generates code.
    """
    if not rag_finetuned_import_success:
        return None
    
    pipeline = get_rag_finetuned_pipeline()
    # Initialize it (loads docs and prepares the model)
    if not pipeline.initialize():
        return None
    return pipeline

def capture_audio_input(pipeline_stt):
    """
    Record audio from the microphone and convert it to text.
    First tries the local Whisper model, then falls back to Google's service.
    """
    recognizer = sr.Recognizer()

    # Settings for better speech recognition
    recognizer.energy_threshold = 300  # Adjust for background noise
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 2.0   # Wait 2 seconds of silence before stopping

    try:
        with sr.Microphone() as source:
            st.info("Adjusting for background noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)

            st.info("Listening for 10 seconds... (Speak now)")
            # Listen for up to 10 seconds to start, then fixed 7 seconds speaking
            audio_data = recognizer.listen(source, timeout=10, phrase_time_limit=7)

            st.info("Processing your speech...")

            # Try the local Whisper model first
            if pipeline_stt:
                # Convert audio to the format Whisper expects
                raw_data = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
                import numpy as np
                audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0

                result = pipeline_stt({"raw": audio_np, "sampling_rate": 16000})
                return result["text"].strip()

            # If local model fails, use Google's web service
            else:
                return recognizer.recognize_google(audio_data).lower()

    except sr.WaitTimeoutError:
        return "Error: No speech detected within 10 seconds."
    except sr.UnknownValueError:
        return "Error: Could not understand the audio."
    except Exception as e:
        return f"Error: {str(e)}"

def generate_tts_audio(text):
    """
    Convert text to speech and save it as an audio file.
    Returns the path to the audio file.
    """
    try:
        if not text:
            return None
        tts = gTTS(text=text, lang="en", slow=False)
        # Create a temporary file to store the audio
        with NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_path = fp.name
            tts.save(temp_path)
        return temp_path
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")
        return None


def extract_class_name(code):
    """
    Extract the Scene class name from the generated Manim code.
    Returns the first class that inherits from Scene.
    """
    # Look for class definitions that inherit from Scene
    pattern = r'class\s+(\w+)\s*\(.*Scene.*\)'
    matches = re.findall(pattern, code)
    if matches:
        return matches[0]
    return None


def compile_manim_video(code, quality="medium"):
    """
    Compile the Manim script and generate a video.
    
    Args:
        code: The Manim Python code to compile
        quality: Video quality - 'low' (480p), 'medium' (720p), 'high' (1080p)
    
    Returns:
        tuple: (success: bool, video_path or error_message: str)
    """
    # Extract the class name from the code
    class_name = extract_class_name(code)
    if not class_name:
        return False, "Could not find a Scene class in the generated code. Make sure the code contains a class that inherits from Scene."
    
    # Quality flags for Manim
    quality_flags = {
        "low": "-ql",      # 480p, 15fps
        "medium": "-qm",   # 720p, 30fps  
        "high": "-qh"      # 1080p, 60fps
    }
    quality_flag = quality_flags.get(quality, "-qm")
    
    # Create a temporary directory for the script and output
    temp_dir = tempfile.mkdtemp(prefix="manim_")
    script_path = os.path.join(temp_dir, "generated_scene.py")
    
    try:
        # Write the code to a temporary file
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
        
        logger.info(f"Saved script to: {script_path}")
        logger.info(f"Compiling class: {class_name}")
        
        # Build the Manim command
        # Using --media_dir to specify output location
        cmd = [
            "manim",
            quality_flag,
            script_path,
            class_name,
            "--media_dir", temp_dir
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Run Manim
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Log output for debugging
        if result.stdout:
            logger.info(f"Manim stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"Manim stderr: {result.stderr}")
        
        # Check if compilation was successful
        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error during compilation"
            return False, f"Manim compilation failed:\n{error_msg}"
        
        # Find the output video file
        # Manim outputs to: media_dir/videos/script_name/quality/class_name.mp4
        video_dir = os.path.join(temp_dir, "videos", "generated_scene")
        
        # Search for the video file
        video_path = None
        for root, dirs, files in os.walk(video_dir):
            for file in files:
                if file.endswith(".mp4"):
                    video_path = os.path.join(root, file)
                    break
            if video_path:
                break
        
        if not video_path or not os.path.exists(video_path):
            # Try alternative location (current media folder)
            alt_video_dir = os.path.join(os.getcwd(), "media", "videos", "generated_scene")
            for root, dirs, files in os.walk(alt_video_dir):
                for file in files:
                    if file.endswith(".mp4"):
                        video_path = os.path.join(root, file)
                        break
                if video_path:
                    break
        
        if not video_path or not os.path.exists(video_path):
            return False, f"Video was compiled but output file not found. Check the logs for details.\nOutput: {result.stdout}"
        
        # Copy video to a persistent location
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        
        final_video_path = os.path.join(output_dir, f"{class_name}.mp4")
        shutil.copy2(video_path, final_video_path)
        
        logger.info(f"Video saved to: {final_video_path}")
        
        return True, final_video_path
        
    except subprocess.TimeoutExpired:
        return False, "Manim compilation timed out (exceeded 5 minutes). Try simplifying the animation."
    except FileNotFoundError:
        return False, "Manim is not installed or not in PATH. Please install Manim: pip install manim"
    except Exception as e:
        logger.error(f"Error compiling Manim: {e}")
        return False, f"Error during compilation: {str(e)}"
    finally:
        # Cleanup temp directory (but keep output if successful)
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass

def generate_manim_code(prompt, model_choice):
    """
    Generate Manim code using the selected model.
    All models integrate with RAG for Manim documentation context.
    
    For Mistral-7B (Finetuned): Uses RAG + finetuned model pipeline
    For CodeLlama-34B: Uses RAG + Groq API (llama-3.1-70b-versatile)
    For Phi-2: Uses RAG + Groq API (llama-3.1-8b-instant)
    """
    if not prompt or not prompt.strip():
        return "# Error: Please enter a description of what you want to visualize"
    
    # Use the finetuned model with RAG for Mistral
    if model_choice == "Mistral-7B (Finetuned)":
        if not rag_finetuned_import_success:
            return f"# Error: RAG + Finetuned model not available. {rag_finetuned_import_error}"
        
        try:
            # Get the RAG + Finetuned pipeline
            rag_pipeline = get_rag_pipeline()
            
            if rag_pipeline is None:
                return "# Error: Could not initialize the RAG + Finetuned model pipeline"
            
            # Generate code using RAG context + finetuned model
            code = rag_pipeline.generate_manim_code(prompt, use_rag=True)
            return code
            
        except Exception as e:
            st.warning(f"Finetuned model failed ({str(e)[:100]}...), falling back to CodeLlama-34B")
            model_choice = "CodeLlama-34B"
    
    # Use Groq API for CodeLlama or Phi-2 (with RAG context)
    if model_choice in ["CodeLlama-34B", "Phi-2"]:
        if not groq_import_success:
            return f"# Error: Groq client not available. {groq_import_error}"
        
        try:
            # Get RAG context first
            rag = get_standalone_rag()
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
                augmented_prompt = prompt
            
            # Get Groq client for the selected model
            model_type = "codellama" if model_choice == "CodeLlama-34B" else "phi2"
            groq_client = get_groq_client(model_type)
            
            # Generate code with Groq API
            code = groq_client.generate(augmented_prompt, max_tokens=2048, temperature=0.7)
            return code
            
        except Exception as e:
            st.error(f"Groq API error: {str(e)}")
            return f"# Error: {str(e)}\n# Make sure GROQ_API_KEY is set in .env file"
    
    # Fallback for unknown model
    return f"# Error: Unknown model choice: {model_choice}"

# Initialize session state variables
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""
if "tts_file_path" not in st.session_state:
    st.session_state.tts_file_path = None
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "CodeLlama-34B"  # Default to available model
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "video_error" not in st.session_state:
    st.session_state.video_error = None
if "video_quality" not in st.session_state:
    st.session_state.video_quality = "medium"

# Main UI Layout

# Sidebar for settings
with st.sidebar:
    st.header("Settings")

    st.subheader("AI Model Selection")
    # Only show finetuned model if GPU is available
    if torch.cuda.is_available():
        model_options = ["Mistral-7B (Finetuned)", "CodeLlama-34B", "Phi-2"]
    else:
        model_options = ["CodeLlama-34B", "Phi-2"]
    
    model_choice = st.selectbox(
        "Choose AI Model",
        model_options,
        index=model_options.index(st.session_state.model_choice) if st.session_state.model_choice in model_options else 0
    )

    # Update session state if model changed
    if model_choice != st.session_state.model_choice:
        st.session_state.model_choice = model_choice
    
    # Show info about the selected model
    if model_choice == "Mistral-7B (Finetuned)":
        st.success("Using finetuned Mistral with RAG")
        st.info("Local finetuned model + Manim docs")
    elif model_choice == "CodeLlama-34B":
        st.success("Using Groq API + RAG")
        st.info("llama-3.1-70b-versatile + Manim docs")
    elif model_choice == "Phi-2":
        st.success("Using Groq API + RAG")
        st.info("llama-3.1-8b-instant + Manim docs")
    

    # LLM Health Check Section
    st.subheader("LLM Health Status")

    # Health for Mistral Finetuned Model (local, GPU only)
    if torch.cuda.is_available():
        if rag_finetuned_import_success:
            try:
                rag_pipeline = get_rag_pipeline()
                if rag_pipeline is not None:
                    color = "#00FF00"
                    st.markdown(f"<span style='color:{color};font-weight:bold'>Mistral-7B (Finetuned) - Healthy</span>", unsafe_allow_html=True)
                else:
                    color = "#FF0000"
                    st.markdown(f"<span style='color:{color};font-weight:bold'>Mistral-7B (Finetuned) - Unhealthy</span>", unsafe_allow_html=True)
            except Exception as e:
                color = "#FF0000"
                st.markdown(f"<span style='color:{color};font-weight:bold'>Mistral-7B (Finetuned) - Unhealthy ({str(e)[:50]})</span>", unsafe_allow_html=True)
        else:
            color = "#FF0000"
            st.markdown(f"<span style='color:{color};font-weight:bold'>Mistral-7B (Finetuned) - Not Available</span>", unsafe_allow_html=True)
    
    # Health for Groq API models
    if groq_import_success:
        try:
            for model_type, label in [("codellama", "CodeLlama-34B"), ("phi2", "Phi-2")]:
                groq_client = get_groq_client(model_type=model_type)
                status = groq_client.get_status()
                color = "#00FF00" if status.get("api_key_set") else "#FF0000"
                status_text = "Healthy" if status.get("api_key_set") else "Unhealthy (No API Key)"
                st.markdown(f"<span style='color:{color};font-weight:bold'>{label} (Groq API) - {status_text}</span>", unsafe_allow_html=True)
        except Exception as e:
            color = "#FF0000"
            st.markdown(f"<span style='color:{color};font-weight:bold'>Groq API - Error: {str(e)[:50]}</span>", unsafe_allow_html=True)
    else:
        color = "#FF0000"
        st.markdown(f"<span style='color:{color};font-weight:bold'>Groq API - Not Available</span>", unsafe_allow_html=True)
    
    # Video Quality Settings
    st.markdown("---")
    st.subheader("Video Settings")
    video_quality = st.selectbox(
        "Video Quality",
        ["low", "medium", "high"],
        index=["low", "medium", "high"].index(st.session_state.video_quality),
        help="low: 480p 15fps, medium: 720p 30fps, high: 1080p 60fps"
    )
    if video_quality != st.session_state.video_quality:
        st.session_state.video_quality = video_quality

# Main Content
st.title("Math2Visual Studio")

# Input Section
col_input, col_actions = st.columns([3, 1])

# Speech recognition is off by default
speech_enabled = st.session_state.get("speech_enabled", False)

with col_input:
    user_input = st.text_area(
        "Describe your mathematical concept",
        value=st.session_state.transcribed_text,
        height=150,
        placeholder="e.g., Visualize the area of a circle formula derivation..."
    )

with col_actions:
    st.write("### Input Controls")
    # Button to enable speech recognition
    if st.button("🎤 Enable Speech Recognition"):
        st.session_state.speech_enabled = True
        st.rerun()
    # Only run speech recognition if enabled
    if speech_enabled:
        stt_pipeline = load_speech_recognition_model()
        text_result = capture_audio_input(stt_pipeline)
        if not text_result.startswith("Error"):
            st.session_state.transcribed_text = text_result
            st.session_state.speech_enabled = False
            st.rerun()
        else:
            st.error(text_result)
            st.session_state.speech_enabled = False

    # Generate code using the selected AI model
    if st.button(" Generate Code"):
        with st.spinner("Generating Manim script..."):
            code_result = generate_manim_code(user_input, model_choice)
            st.session_state.generated_code = code_result
            completion_msg = "Code generation complete. Please review the script below."
            audio_file = generate_tts_audio(completion_msg)
            st.session_state.tts_file_path = audio_file
        st.rerun()

# Output Section
st.markdown("---")

if st.session_state.generated_code:
    st.subheader("Generated Manim Code")
    
    # Code Display with editable option
    edited_code = st.text_area(
        "Edit code if needed:",
        value=st.session_state.generated_code,
        height=400,
        key="code_editor"
    )
    
    # Update session state if code was edited
    if edited_code != st.session_state.generated_code:
        st.session_state.generated_code = edited_code
    
    # Video Generation Section
    st.markdown("---")
    col_video_btn, col_video_info = st.columns([1, 2])
    
    with col_video_btn:
        if st.button(" Generate Video", type="primary"):
            with st.spinner(f"Compiling Manim animation ({st.session_state.video_quality} quality)..."):
                # Reset previous video state
                st.session_state.video_path = None
                st.session_state.video_error = None
                
                # Compile the video
                success, result = compile_manim_video(
                    st.session_state.generated_code,
                    quality=st.session_state.video_quality
                )
                
                if success:
                    st.session_state.video_path = result
                    st.session_state.video_error = None
                    # Generate success audio feedback
                    success_msg = "Video generated successfully! You can now watch the animation below."
                    st.session_state.tts_file_path = generate_tts_audio(success_msg)
                else:
                    st.session_state.video_error = result
                    st.session_state.video_path = None
            
            st.rerun()
    
    with col_video_info:
        st.info(f"Quality: {st.session_state.video_quality} | This may take a few moments depending on animation complexity.")
    
    # Display video error if any
    if st.session_state.video_error:
        st.error("Video Compilation Failed")
        with st.expander("View Error Details", expanded=True):
            st.code(st.session_state.video_error, language="text")
    
    # Display the rendered video
    if st.session_state.video_path and os.path.exists(st.session_state.video_path):
        st.markdown("---")
        st.subheader(" Generated Animation")
        
        # Display the video
        st.video(st.session_state.video_path)
        
        # Add download button for the video
        with open(st.session_state.video_path, "rb") as video_file:
            video_bytes = video_file.read()
            st.download_button(
                label=" Download Video",
                data=video_bytes,
                file_name=os.path.basename(st.session_state.video_path),
                mime="video/mp4"
            )
    
    # Audio Feedback Section
    st.markdown("---")
    if st.session_state.tts_file_path:
        st.subheader(" Audio Feedback")
        st.audio(st.session_state.tts_file_path, format="audio/mp3")
        
        # Option to read the code summary
        if st.button(" Read Code Explanation"):
            explanation_text = f"This code creates a Manim animation for {user_input[:100]}. The animation has been compiled and is ready to view."
            explanation_audio = generate_tts_audio(explanation_text)
            if explanation_audio:
                st.audio(explanation_audio, format="audio/mp3")
