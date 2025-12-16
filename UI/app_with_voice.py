import streamlit as st
import os
import requests
import speech_recognition as sr
from dotenv import load_dotenv
from gtts import gTTS
from tempfile import NamedTemporaryFile
from transformers import pipeline
import torch

# Configuration & Setup
load_dotenv()

# Configuration Constants
STT_MODEL_ID = os.getenv("M2V_STT_MODEL", "distil-whisper/distil-medium.en")
LLM_ENDPOINT = os.getenv("M2V_LLM_ENDPOINT", "http://localhost:8000/generate")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="Math2Visual",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme CSS
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
    /* Hide Streamlit default menu/footer for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Core Logic: AI Models & Audio Processing

@st.cache_resource
def load_speech_recognition_model():
    """
    Loads the local Whisper model for high-fidelity transcription.
    Cached to prevent reloading on every interaction.
    """
    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=STT_MODEL_ID,
            device=DEVICE
        )
        return pipe
    except Exception as e:
        st.error(f"Error loading local STT model: {e}")
        return None

def capture_audio_input(pipeline_stt):
    """
    Captures audio from the microphone and transcribes it.
    Prioritizes local Whisper model; falls back to Google Web API if local model fails.
    """
    recognizer = sr.Recognizer()
    
    # Critical settings for "lengthy" speech
    recognizer.energy_threshold = 300  # Adjust based on ambient noise
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 2.0   # Allow 2 seconds of silence before cutting off

    try:
        with sr.Microphone() as source:
            st.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            st.info("Listening... (Speak now)")
            # timeout=None waits indefinitely for speech to start
            # phrase_time_limit=None allows indefinite speaking length
            audio_data = recognizer.listen(source, timeout=10, phrase_time_limit=None)
            
            st.info("Processing audio...")

            # 1. Try Local Whisper Model (High Performance)
            if pipeline_stt:
                # Convert AudioData to raw bytes for transformers
                raw_data = audio_data.get_raw_data(convert_rate=16000, convert_width=2)
                import numpy as np
                audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                result = pipeline_stt({"raw": audio_np, "sampling_rate": 16000})
                return result["text"].strip()
            
            # 2. Fallback to Google (Standard Performance)
            else:
                return recognizer.recognize_google(audio_data).lower()

    except sr.WaitTimeoutError:
        return "Error: Listening timed out. No speech detected."
    except sr.UnknownValueError:
        return "Error: Could not understand audio."
    except Exception as e:
        return f"Error: {str(e)}"

def generate_tts_audio(text):
    """
    Converts text to speech and returns the file path.
    """
    try:
        if not text:
            return None
        tts = gTTS(text=text, lang="en", slow=False)
        # Create a temp file that persists long enough for the UI to render
        with NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_path = fp.name
            tts.save(temp_path)
        return temp_path
    except Exception as e:
        st.error(f"TTS Generation Error: {e}")
        return None

def query_llm(prompt):
    """
    Sends the prompt to the LLM endpoint to retrieve Manim code.
    """
    payload = {
        "prompt": f"Write a complete, error-free Manim (Python) script to animate the following concept. Return only the Python code:\n\n{prompt}",
        "temperature": 0.2
    }
    
    try:
        response = requests.post(LLM_ENDPOINT, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json().get("code", "")
        else:
            return f"# Error: LLM Provider returned status {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "# Error: Could not connect to LLM endpoint. Ensure the server is running."
    except Exception as e:
        return f"# Error: {str(e)}"

# Session State Management
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""
if "tts_file_path" not in st.session_state:
    st.session_state.tts_file_path = None

# UI Layout

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    
    st.subheader("Model Configuration")
    model_choice = st.selectbox("LLM Model", ["CodeLlama-34B", "Mistral-7B", "GPT-4-Turbo"])
    stt_status = "Active (Local)" if torch.cuda.is_available() else "Active (CPU/Cloud)"
    st.info(f"STT Engine: {stt_status}")

    st.markdown("---")
    st.subheader("Animation Parameters")
    resolution = st.select_slider("Resolution", options=["480p", "720p", "1080p", "4K"], value="1080p")
    duration = st.number_input("Max Duration (sec)", min_value=10, max_value=300, value=60)

# Main Content
st.title("Math2Visual Studio")
st.markdown("### Text/Voice to Mathematical Animation")

# Input Section
col_input, col_actions = st.columns([3, 1])

with col_input:
    user_input = st.text_area(
        "Describe your mathematical concept",
        value=st.session_state.transcribed_text,
        height=150,
        placeholder="e.g., Visualize the area of a circle formula derivation..."
    )

with col_actions:
    st.write("### Input Controls")
    
    # Load model in background
    stt_pipeline = load_speech_recognition_model()

    if st.button("Start Recording"):
        text_result = capture_audio_input(stt_pipeline)
        if not text_result.startswith("Error"):
            st.session_state.transcribed_text = text_result
            st.rerun()
        else:
            st.error(text_result)
            
    if st.button("Generate Code"):
        if not user_input.strip():
            st.warning("Please enter a prompt first.")
        else:
            with st.spinner("Generating Manim script..."):
                code_result = query_llm(user_input)
                st.session_state.generated_code = code_result
                
                # Generate TTS for completion
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
        if st.button("Read Code Explanation"):
            explanation_text = f"This code creates a scene specifically for {user_input[:50]}..."
            explanation_audio = generate_tts_audio(explanation_text)
            st.audio(explanation_audio, format="audio/mp3")
