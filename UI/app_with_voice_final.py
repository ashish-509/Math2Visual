import streamlit as st
import os
import requests
import speech_recognition as sr
from dotenv import load_dotenv
from gtts import gTTS
from tempfile import NamedTemporaryFile
from transformers import pipeline
import torch

# Import our custom LLM client for multi-model support
from src.llm.client import LLMClient

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
def get_llm_client():
    """
    Get the multi-model LLM client.
    This handles switching between different AI models automatically.
    """
    return LLMClient()

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

            st.info("Listening... (Speak now)")
            # Listen for up to 10 seconds to start, then unlimited speaking
            audio_data = recognizer.listen(source, timeout=10, phrase_time_limit=None)

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

def generate_manim_code(prompt, model_choice):
    """
    Generate Manim code using the selected AI model.
    The LLMClient handles model switching and fallbacks automatically.
    """
    try:
        # Map UI model names to client model names
        model_map = {
            "Mistral-7B": "mistral",
            "CodeLlama-34B": "codellama",
            "Phi-2": "phi2"
        }

        client = get_llm_client()
        model_name = model_map.get(model_choice, "mistral")

        # Generate the code using our multi-model client
        code = client.generate(prompt, model_name)
        return code

    except Exception as e:
        return f"# Error generating code: {str(e)}"

# Initialize session state variables
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""
if "tts_file_path" not in st.session_state:
    st.session_state.tts_file_path = None
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "Mistral-7B"  # Default model

# Main UI Layout

# Sidebar for settings
with st.sidebar:
    st.header("Settings")

    st.subheader("AI Model Selection")
    model_choice = st.selectbox(
        "Choose AI Model",
        ["Mistral-7B", "CodeLlama-34B", "Phi-2"],
        index=["Mistral-7B", "CodeLlama-34B", "Phi-2"].index(st.session_state.model_choice)
    )

    # Update session state if model changed
    if model_choice != st.session_state.model_choice:
        st.session_state.model_choice = model_choice

    stt_status = "Active (GPU)" if torch.cuda.is_available() else "Active (CPU)"
    st.info(f" STT Engine: {stt_status}")
    
    if torch.cuda.is_available():
        st.info(f" GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.warning(" No GPU detected - model running on CPU (slower)")

    st.markdown("---")
    st.subheader("Animation Parameters")
    resolution = st.select_slider("Resolution", options=["480p", "720p", "1080p", "4K"], value="1080p")
    duration = st.number_input("Max Duration (sec)", min_value=10, max_value=300, value=60)

# Main Content
st.title("Math2Visual Studio")
st.markdown("### Text/Voice to Mathematical Animation")

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
    if st.button("⚡ Generate Code"):
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
    
    # Code Display
    st.code(st.session_state.generated_code, language="python")
    
    # Audio Feedback Section
    if st.session_state.tts_file_path:
        st.subheader("Audio Feedback")
        st.audio(st.session_state.tts_file_path, format="audio/mp3")
        
        # Option to read the code summary (simulated)
        if st.button("📖 Read Code Explanation"):
            explanation_text = f"This code creates a scene specifically for {user_input[:50]}..."
            explanation_audio = generate_tts_audio(explanation_text)
            st.audio(explanation_audio, format="audio/mp3")
