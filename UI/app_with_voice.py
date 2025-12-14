import streamlit as st
import os
from dotenv import load_dotenv
import requests
import speech_recognition as sr
from gtts import gTTS
from tempfile import NamedTemporaryFile

# Environment & Page Config
load_dotenv()

STT_MODEL = os.getenv("M2V_STT_MODEL", "distil-whisper/distil-large-v3")
STT_DEVICE = os.getenv("M2V_STT_DEVICE", "cpu")
LLM_ENDPOINT = os.getenv("M2V_LLM_ENDPOINT", "http://localhost:8000/generate")

st.set_page_config(
    page_title="Math2Visual",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #181c24 0%, #232946 100%);
    color: #f4f4f4;
}
</style>
""", unsafe_allow_html=True)

# Voice Utils 
def listen_for_speech():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
        text = recognizer.recognize_google(audio)
        return text.lower()
    except sr.WaitTimeoutError:
        return "Timeout: No speech detected"
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Could not request results"
    except Exception as e:
        return f"Error: {str(e)}"

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        with NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_path = fp.name
            tts.save(temp_path)
        return temp_path
    except Exception as e:
        st.error(f"TTS Error: {str(e)}")
        return None

def cleanup_audio_file(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

# Helper Functions
def load_stt_model():
    from transformers import pipeline
    try:
        pipe = pipeline(
            task="automatic-speech-recognition",
            model=STT_MODEL,
            device=0 if STT_DEVICE.isdigit() else -1
        )
        st.success(" Whisper STT model loaded")
        return pipe
    except Exception as e:
        st.error(f" Failed to load Whisper STT: {e}")
        return None

def send_to_llm(prompt: str):
    try:
        resp = requests.post(
            LLM_ENDPOINT,
            json={"prompt": prompt},
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json().get("code", "")
        else:
            st.warning(f"LLM error: {resp.status_code}")
            return ""
    except Exception as e:
        st.warning(f"LLM request failed: {e}")
        return ""

# Session State
if "voice_prompt" not in st.session_state:
    st.session_state.voice_prompt = ""

if "tts_audio" not in st.session_state:
    st.session_state.tts_audio = None

# Sidebar
with st.sidebar:
    st.header("Math2Visual Settings")
    st.selectbox(
        "Select AI Model",
        ["CodeLlama-7B (Fine-tuned)", "CodeLlama-13B", "Mistral-7B", "Phi-2"]
    )
    st.select_slider(
        "Video Quality",
        ["Low", "Medium", "High", "4K Ultra HD"],
        value="Medium"
    )
    st.slider("Max Duration (seconds)", 5, 300, 60, step=10)
    st.selectbox(
        "Animation Style",
        ["Smooth", "Fast", "Detailed", "Minimalist"]
    )

# Main UI
st.title("Math2Visual ")
st.subheader("Speech / Text → Math → Manim Animation Code")

user_prompt = st.text_area(
    "Enter your prompt",
    value=st.session_state.voice_prompt,
    placeholder="Explain the Pythagorean theorem with animation..."
)

col1, col2 = st.columns([1, 3])

with col1:
    if st.button(" Speak"):
        with st.spinner("Listening..."):
            spoken_text = listen_for_speech()
            st.session_state.voice_prompt = spoken_text
            st.success("Voice captured")

with col2:
    if st.session_state.voice_prompt:
        st.info(f"Recognized Speech:\n{st.session_state.voice_prompt}")

# Generate Code
if st.button("Generate Code & Video"):
    if not user_prompt.strip():
        st.warning("Please provide a prompt.")
    else:
        stt_model = load_stt_model()  # Optional Whisper load
        final_prompt = f"Generate Manim code for: {user_prompt}"

        with st.spinner("Generating Manim code..."):
            code = send_to_llm(final_prompt)

        if code:
            st.code(code, language="python")
            st.success("Manim code generated")

            if st.button("Read Status"):
                audio_path = text_to_speech(
                    "Manim code generation completed successfully."
                )
                if audio_path:
                    st.audio(audio_path)
                    st.session_state.tts_audio = audio_path
        else:
            st.warning("Code generation failed")

# Cleanup
if st.session_state.tts_audio:
    cleanup_audio_file(st.session_state.tts_audio)
    st.session_state.tts_audio = None
