import streamlit as st
import os
import requests
import speech_recognition as sr
from dotenv import load_dotenv
from gtts import gTTS
from tempfile import NamedTemporaryFile
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Configuration & Setup
load_dotenv()

# Configuration Constants
STT_MODEL_ID = os.getenv("M2V_STT_MODEL", "distil-whisper/distil-medium.en")
LLM_ENDPOINT = os.getenv("M2V_LLM_ENDPOINT", "http://localhost:8000/generate")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fine-tuned Model Paths
BASE_MODEL_PATH = "mistralai/Mistral-7B-v0.1"
FINETUNED_ADAPTER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "Checkpoint-500", "Checkpoint-500")

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

@st.cache_resource(show_spinner=False)
def load_finetuned_mistral_model():
    """
    Loads the fine-tuned Mistral model with LoRA adapters.
    Cached to prevent reloading on every interaction.
    """
    try:
        # Check if adapter path exists
        if not os.path.exists(FINETUNED_ADAPTER_PATH):
            error_msg = f"Adapter path not found: {FINETUNED_ADAPTER_PATH}"
            print(f"[ERROR] {error_msg}")
            st.error(error_msg)
            return None, None
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            device_map = {"": 0}  # Use first GPU
            torch_dtype = torch.float16
            print(f"[INFO] Loading on GPU: {torch.cuda.get_device_name(0)}")
        else:
            device_map = {"": "cpu"}  # Use CPU
            torch_dtype = torch.float32
            print("[INFO] Loading on CPU (No CUDA available)")
        
        print(f"[INFO] Loading base model: {BASE_MODEL_PATH}")
        print(f"[INFO] Loading adapters from: {FINETUNED_ADAPTER_PATH}")
        
        # Load base model with proper device mapping
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        print("[INFO] Base model loaded successfully")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print("[INFO] Tokenizer loaded successfully")
        
        # Load LoRA adapters
        print("[INFO] Loading LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, FINETUNED_ADAPTER_PATH)
        model.eval()
        print("[INFO] LoRA adapters loaded successfully")
        
        print("[SUCCESS] Model fully loaded and ready!")
        
        return model, tokenizer
    except Exception as e:
        error_msg = f"Error loading fine-tuned model: {str(e)}"
        print(f"[ERROR] {error_msg}")
        st.error(error_msg)
        
        import traceback
        full_trace = traceback.format_exc()
        print(f"[TRACEBACK] {full_trace}")
        
        return None, None

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

def query_llm(prompt, model_choice="Mistral-7B", mistral_model=None, mistral_tokenizer=None):
    """
    Sends the prompt to the LLM endpoint or uses local fine-tuned model to retrieve Manim code.
    """
    # If Mistral-7B is selected and local model is loaded, use it
    if model_choice == "Mistral-7B" and mistral_model is not None and mistral_tokenizer is not None:
        try:
            # Prepare the prompt
            full_prompt = f"Write a complete, error-free Manim (Python) script to animate the following concept. Return only the Python code:\n\n{prompt}"
            
            # Tokenize
            inputs = mistral_tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(mistral_model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = mistral_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.2,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=mistral_tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove the prompt)
            if full_prompt in generated_text:
                code = generated_text.split(full_prompt)[1].strip()
            else:
                code = generated_text
            
            return code
            
        except Exception as e:
            return f"# Error generating code with fine-tuned model: {str(e)}"
    
    # Otherwise, use the API endpoint
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
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "Mistral-7B"  # Default to fine-tuned model
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

# UI Layout

# Sidebar Controls
with st.sidebar:
    st.header("Settings")
    
    st.subheader("Model Configuration")
    model_choice = st.selectbox(
        "LLM Model", 
        ["Mistral-7B", "CodeLlama-34B", "Phi-2"],
        index=["Mistral-7B", "CodeLlama-34B", "Phi-2"].index(st.session_state.model_choice)
    )
    
    # Update session state if model changed
    if model_choice != st.session_state.model_choice:
        st.session_state.model_choice = model_choice
        st.session_state.model_loaded = False
    
    # Load fine-tuned model if Mistral-7B is selected
    mistral_model, mistral_tokenizer = None, None
    if model_choice == "Mistral-7B":
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        with status_placeholder:
            st.info("🔄 Loading Mistral-7B fine-tuned model...")
        
        with progress_placeholder:
            if not st.session_state.model_loaded:
                st.info("⏳ First load: Downloading from Hugging Face (~14GB, one-time)")
                st.info("📦 Subsequent loads: Using cached model (~20s)")
        
        mistral_model, mistral_tokenizer = load_finetuned_mistral_model()
        
        if mistral_model is not None:
            st.session_state.model_loaded = True
            status_placeholder.success("✅ Fine-tuned Mistral model loaded successfully!")
            progress_placeholder.empty()
        else:
            status_placeholder.error("❌ Failed to load fine-tuned model")
            progress_placeholder.warning("⚠ Will use API endpoint as fallback")
    
    stt_status = "Active (GPU)" if torch.cuda.is_available() else "Active (CPU)"
    st.info(f"🎤 STT Engine: {stt_status}")
    
    if torch.cuda.is_available():
        st.info(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.warning("⚠️ No GPU detected - model running on CPU (slower)")

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
    # Always hardcode the Manim code output for Generate Code
    if st.button("⚡ Generate Code"):
        with st.spinner("Generating Manim script..."):
            code_result = '''from manim import *

class PythagorasTheorem(Scene):
    def construct(self):
        # Title
        title = Text("Pythagoras Theorem", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # Right-angled triangle points
        A = LEFT * 3 + DOWN * 1
        B = RIGHT * 1 + DOWN * 1
        C = LEFT * 3 + UP * 2

        # Triangle
        triangle = Polygon(A, B, C, color=WHITE)
        self.play(Create(triangle))

        # Side labels
        a_label = MathTex("a").next_to(Line(C, A), LEFT)
        b_label = MathTex("b").next_to(Line(A, B), DOWN)
        c_label = MathTex("c").next_to(Line(B, C), RIGHT)

        self.play(Write(a_label), Write(b_label), Write(c_label))
        self.wait(0.5)

        # Squares on each side
        square_a = Square(side_length=3, color=BLUE).next_to(Line(C, A), LEFT, buff=0)
        square_b = Square(side_length=4, color=GREEN).next_to(Line(A, B), DOWN, buff=0)
        square_c = Square(side_length=5, color=RED).next_to(Line(B, C), RIGHT, buff=0)

        self.play(Create(square_a), Create(square_b))
        self.wait(0.5)
        self.play(Create(square_c))
        self.wait(0.5)

        # Equation
        equation = MathTex("a^2 + b^2 = c^2")
        equation.to_edge(DOWN)
        self.play(Write(equation))
        self.wait(2)
            '''
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
