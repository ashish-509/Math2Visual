import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import base64
import speech_recognition as sr
from tempfile import NamedTemporaryFile

# Configuration

# Backend URL 
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# Create a session with connection pooling and retry logic
# This reuses TCP connections, reducing latency on subsequent requests
def get_http_session():
    if "http_session" not in st.session_state:
        session = requests.Session()
        
        # Configure retry strategy for resilience
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],
        )
        
        # Mount adapter with connection pooling (keeps connections alive)
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        st.session_state.http_session = session
    
    return st.session_state.http_session


# Page Setup

st.set_page_config(
    page_title="Math2Visual Studio",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme styling with auto-resize textareas
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Auto-resize textareas */
    .stTextArea textarea {
        min-height: 60px !important;
        overflow-y: hidden;
        resize: none;
    }
    
    /* Compact input textarea */
    div[data-testid="stTextArea"]:has(textarea[aria-label="Describe your mathematical concept"]) textarea {
        min-height: 42px !important;
        height: auto !important;
    }
    
    /* Teaching script textarea - slightly larger */
    div[data-testid="stTextArea"]:has(textarea[aria-label="Teaching explanation:"]) textarea {
        min-height: 80px !important;
        max-height: 200px !important;
        overflow-y: auto !important;
    }
</style>

<script>
// Auto-resize textareas on input
document.addEventListener('input', function(e) {
    if (e.target.tagName === 'TEXTAREA') {
        e.target.style.height = 'auto';
        e.target.style.height = e.target.scrollHeight + 'px';
    }
});
</script>
""", unsafe_allow_html=True)


# Helper Functions

def check_backend_health():
    try:
        session = get_http_session()
        response = session.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def get_available_models():
    try:
        session = get_http_session()
        response = session.get(f"{BACKEND_URL}/available_models", timeout=5)
        if response.status_code == 200:
            return response.json().get("models", [])
        return ["CodeLlama-34B", "Phi-2"]
    except requests.exceptions.RequestException:
        return ["CodeLlama-34B", "Phi-2"]


def generate_code_api(prompt, model_choice):
    # Call backend API to generate Manim code.
    try:
        session = get_http_session()
        response = session.post(
            f"{BACKEND_URL}/generate_code",
            json={"prompt": prompt, "model_choice": model_choice},
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("success", False), data.get("code", "")
        else:
            return False, f"Backend error: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return False, "Request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"


def compile_video_api(code, quality):
    # Call backend API to compile video.
    try:
        session = get_http_session()
        response = session.post(
            f"{BACKEND_URL}/compile_video",
            json={"code": code, "quality": quality},
            timeout=600
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return True, data.get("video_path")
            else:
                return False, data.get("error_message", "Unknown error")
        else:
            return False, f"Backend error: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return False, "Video compilation timed out."
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"


def extract_from_image_api(image_base64, mime_type="image/png"):
    # Send a base64-encoded image to the backend for math extraction.
    try:
        session = get_http_session()
        response = session.post(
            f"{BACKEND_URL}/extract_from_image",
            json={"image_base64": image_base64, "mime_type": mime_type},
            timeout=120
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return True, data.get("extracted_text", "")
            else:
                return False, data.get("message", "Extraction failed")
        else:
            return False, f"Backend error: {response.status_code}"

    except requests.exceptions.Timeout:
        return False, "Image extraction timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"


def regenerate_and_compile_api(prompt, current_code, error_message, model_choice, quality, max_retries=2):
    # Call the backend error-feedback regeneration endpoint.
    try:
        session = get_http_session()
        response = session.post(
            f"{BACKEND_URL}/regenerate_and_compile",
            json={
                "prompt": prompt,
                "current_code": current_code,
                "error_message": error_message,
                "model_choice": model_choice,
                "quality": quality,
                "max_retries": max_retries
            },
            timeout=900
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return True, data.get("video_path", ""), data.get("code", "")
            else:
                return False, data.get("message", "Unknown error"), data.get("code", "")
        else:
            return False, f"Backend error: {response.status_code}", ""

    except requests.exceptions.Timeout:
        return False, "Regeneration timed out. The server may still be working.", ""
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}", ""


def smart_generate_api(prompt, model_choice, quality="medium", max_retries=3):
    # Uses error feedback loop - if compilation fails, error goes back to LLM for correction.
    try:
        session = get_http_session()
        response = session.post(
            f"{BACKEND_URL}/smart_generate",
            json={
                "prompt": prompt,
                "model_choice": model_choice,
                "quality": quality,
                "max_retries": max_retries
            },
            timeout=900
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return True, data.get("video_path"), data.get("code", "")
            else:
                return False, data.get("message", "Unknown error"), data.get("code", "")
        else:
            return False, f"Backend error: {response.status_code}", ""
            
    except requests.exceptions.Timeout:
        return False, "Request timed out. The generation may still be running.", ""
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}", ""


def generate_teaching_script_api(description, code, model_choice, video_duration=0.0):
    """Call backend API to generate teaching script matched to video duration."""
    try:
        session = get_http_session()
        response = session.post(
            f"{BACKEND_URL}/generate_teaching_script",
            json={
                "animation_description": description,
                "manim_code": code,
                "model_choice": model_choice,
                "video_duration": video_duration
            },
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return True, data.get("script", "")
            else:
                return False, data.get("message", "Unknown error")
        else:
            return False, f"Backend error: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return False, "Request timed out."
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"


def get_video_duration_api(video_path):
    try:
        session = get_http_session()
        response = session.post(
            f"{BACKEND_URL}/get_video_duration",
            json={"video_path": video_path},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("duration", 0.0)
        return 0.0
    except:
        return 0.0


def generate_tts_api(text):
    try:
        session = get_http_session()
        response = session.post(
            f"{BACKEND_URL}/generate_tts",
            json={"text": text},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return data.get("audio_path")
        return None
        
    except requests.exceptions.RequestException:
        return None


def merge_video_audio_api(video_path, audio_text, manim_code=None):
    try:
        session = get_http_session()
        payload = {"video_path": video_path, "audio_text": audio_text}
        if manim_code:
            payload["manim_code"] = manim_code
        response = session.post(
            f"{BACKEND_URL}/merge_video_audio",
            json=payload,
            timeout=300
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return True, data.get("final_video_path")
            else:
                return False, data.get("message", "Unknown error")
        else:
            return False, f"Backend error: {response.status_code}"
            
    except requests.exceptions.Timeout:
        return False, "Video merging timed out."
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"


# CHATBOT API FUNCTIONS

def check_chatbot_status():
    try:
        session = get_http_session()
        response = session.get(f"{BACKEND_URL}/chatbot/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"available": False, "ready": False}
    except requests.exceptions.RequestException:
        return {"available": False, "ready": False}


def ask_chatbot(question: str, include_examples: bool = False):
    try:
        session = get_http_session()
        response = session.post(
            f"{BACKEND_URL}/chatbot/ask",
            json={"question": question, "include_examples": include_examples},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "answer": f"Backend error: {response.status_code}",
                "sources": "",
                "cached": False
            }
            
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "answer": "Request timed out. Please try again.",
            "sources": "",
            "cached": False
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "answer": f"Connection error: {str(e)}",
            "sources": "",
            "cached": False
        }


def get_syntax_info(class_name: str):
    try:
        session = get_http_session()
        response = session.get(
            f"{BACKEND_URL}/chatbot/syntax/{class_name}",
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "answer": f"Could not find info for {class_name}",
                "sources": ""
            }
            
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "answer": f"Error: {str(e)}",
            "sources": ""
        }


def transcribe_audio(audio_file):
    # audio_file is a WAV recorded in the browser via st.audio_input
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data).lower()
    except sr.UnknownValueError:
        return "Error: Could not understand the audio."
    except Exception as e:
        return f"Error: {str(e)}"


# Initialize Session State

if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""

if "model_choice" not in st.session_state:
    st.session_state.model_choice = "CodeLlama-34B"

if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "video_error" not in st.session_state:
    st.session_state.video_error = None

if "video_quality" not in st.session_state:
    st.session_state.video_quality = "medium"

if "teaching_script" not in st.session_state:
    st.session_state.teaching_script = ""

if "original_prompt" not in st.session_state:
    st.session_state.original_prompt = ""

if "teaching_audio_path" not in st.session_state:
    st.session_state.teaching_audio_path = None

if "last_speech_file_id" not in st.session_state:
    st.session_state.last_speech_file_id = None

if "final_video_path" not in st.session_state:
    st.session_state.final_video_path = None

if "final_video_error" not in st.session_state:
    st.session_state.final_video_error = None

# Chatbot session state
if "chatbot_history" not in st.session_state:
    st.session_state.chatbot_history = []

if "current_page" not in st.session_state:
    st.session_state.current_page = "Studio"

# Generation counter to create unique widget keys (prevents stale cache)
if "generation_id" not in st.session_state:
    st.session_state.generation_id = 0

# Photo to Animation session state
if "photo_extracted_text" not in st.session_state:
    st.session_state.photo_extracted_text = ""

if "photo_confirmed_prompt" not in st.session_state:
    st.session_state.photo_confirmed_prompt = ""

if "photo_generated_code" not in st.session_state:
    st.session_state.photo_generated_code = ""

if "photo_video_path" not in st.session_state:
    st.session_state.photo_video_path = None

if "photo_video_error" not in st.session_state:
    st.session_state.photo_video_error = None

if "photo_teaching_script" not in st.session_state:
    st.session_state.photo_teaching_script = ""

if "photo_teaching_audio" not in st.session_state:
    st.session_state.photo_teaching_audio = None

if "photo_final_video" not in st.session_state:
    st.session_state.photo_final_video = None

if "photo_final_error" not in st.session_state:
    st.session_state.photo_final_error = None

if "photo_gen_id" not in st.session_state:
    st.session_state.photo_gen_id = 0


# Sidebar

with st.sidebar:
    # Navigation 
    st.header("Navigation")
    page_options = ["Studio", "Photo to Animation", "Syntax Assistant"]
    current_page = st.radio(
        "Choose Mode:",
        page_options,
        index=page_options.index(
            st.session_state.current_page
            if st.session_state.current_page in page_options
            else "Studio"
        ),
        key="nav_radio"
    )
    
    # Update session state
    st.session_state.current_page = current_page
    
    # Show settings for Studio and Photo to Animation pages
    if st.session_state.current_page in ("Studio", "Photo to Animation"):
        st.markdown("---")
        
        # Backend Status
        st.header("Settings")
        health = check_backend_health()
        
        if health:
            st.success("Backend: Connected")
        else:
            st.error("Backend: Not Connected")
        
        model_options = get_available_models()
        if not model_options:
            model_options = ["CodeLlama-34B", "Phi-2"]
        
        # Make sure current selection is valid
        if st.session_state.model_choice not in model_options:
            st.session_state.model_choice = model_options[0] if model_options else "CodeLlama-34B"
        
        model_choice = st.selectbox(
            "Choose AI Model",
            model_options,
            index=model_options.index(st.session_state.model_choice) if st.session_state.model_choice in model_options else 0
        )
        
        if model_choice != st.session_state.model_choice:
            st.session_state.model_choice = model_choice
        
        # Video Quality Settings
        st.markdown("---")
        
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

                                     
# STUDIO PAGE

if st.session_state.current_page == "Studio":
    
    # Get health status for studio
    health = check_backend_health()
    
    # Input Section
    col_input, col_actions = st.columns([3, 1])

    with col_input:
        user_input = st.text_area(
            "Describe your mathematical concept",
            value=st.session_state.transcribed_text,
            height=68,
            placeholder="e.g., Visualize the area of a circle formula derivation..."
        )

    with col_actions:
        
        # Speech input: recorded in the browser, so no server-side microphone is needed
        speech_audio = st.audio_input("Speak your prompt")
        if speech_audio is not None and speech_audio.file_id != st.session_state.last_speech_file_id:
            st.session_state.last_speech_file_id = speech_audio.file_id
            with st.spinner("Transcribing..."):
                text_result = transcribe_audio(speech_audio)
            if text_result.startswith("Error"):
                st.error(text_result)
            else:
                st.session_state.transcribed_text = text_result
                st.rerun()
        
        # Generate code button
        if st.button("Generate Code", type="primary"):
            if not user_input or not user_input.strip():
                st.error("Please enter a description first")
            elif not health:
                st.error("Backend not connected. Please start the backend server.")
            else:
                # Clear all previous generation state when starting a new prompt
                st.session_state.generated_code = ""
                st.session_state.video_path = None
                st.session_state.video_error = None
                st.session_state.teaching_script = ""
                st.session_state.teaching_audio_path = None
                st.session_state.final_video_path = None
                st.session_state.final_video_error = None
                
                # Increment generation ID to invalidate old widget caches
                st.session_state.generation_id += 1
                
                with st.spinner("Generating Manim script..."):
                    success, result = generate_code_api(user_input, st.session_state.model_choice)
                    
                    # Store the original prompt for teaching script
                    st.session_state.original_prompt = user_input
                    
                    st.session_state.generated_code = result
                st.rerun()


    # Output Section

    if st.session_state.generated_code:
        st.subheader("Generated Manim Code")
        
        # Display syntax-highlighted code with copy functionality
        st.code(st.session_state.generated_code, language="python", line_numbers=True)
        
        # Download button
        st.download_button(
            label="Download Code",
            data=st.session_state.generated_code,
            file_name="generated_manim_code.py",
            mime="text/x-python",
            help="Download the code as a .py file"
        )
        
        # Expandable section for editing
        with st.expander("Edit Code", expanded=False):
            # Calculate dynamic height based on code lines
            code_lines = st.session_state.generated_code.count('\n') + 1
            code_height = min(max(150, code_lines * 20), 500)  # Min 150, max 500
            
            edited_code = st.text_area(
                "Modify the code below:",
                value=st.session_state.generated_code,
                height=code_height,
                key=f"code_editor_{st.session_state.generation_id}"
            )
            
            col_save, col_info = st.columns([1, 3])
            with col_save:
                if st.button("Save Changes", key=f"save_code_{st.session_state.generation_id}"):
                    if edited_code != st.session_state.generated_code:
                        st.session_state.generated_code = edited_code
                        st.rerun()
            with col_info:
                if edited_code != st.session_state.generated_code:
                    st.caption("Unsaved changes - click Save Changes")
        
        # Video Generation Section
        st.markdown("---")
        col_video_btn, col_video_info = st.columns([1, 2])
        
        with col_video_btn:
            if st.button("Generate Video", type="primary"):
                if not health:
                    st.error("Backend not connected")
                else:
                    # Clear any previous results before starting fresh
                    st.session_state.video_path = None
                    st.session_state.video_error = None

                    # Step 1: First compilation attempt
                    with st.spinner(
                        f"Compiling animation ({st.session_state.video_quality} quality)..."
                    ):
                        success, result = compile_video_api(
                            st.session_state.generated_code,
                            st.session_state.video_quality
                        )

                    if success:
                        # Compilation worked straight away - store the video and refresh
                        st.session_state.video_path = result

                    else:
                        # Step 2: Compilation failed.
                        # Show a visible "Compilation failed… Regenerating" message,
                        # then call the error-feedback regeneration endpoint which:
                        #   a) tries an instant LaTeX → Text() auto-fix (no LLM needed)
                        #   b) if that is not enough, asks the LLM to regenerate the
                        #      code with the error as context
                        #   c) recompiles and returns the fixed video
                        compile_error = result

                        # Display the banner immediately so the user sees it while
                        # the regeneration runs in the spinner below
                        regen_banner = st.warning(
                            " Compilation failed. Regenerating the code. Please wait…"
                        )

                        with st.spinner(
                            "Analysing the error and regenerating fixed code…"
                        ):
                            regen_success, regen_result, regen_code = (
                                regenerate_and_compile_api(
                                    prompt=st.session_state.original_prompt or "",
                                    current_code=st.session_state.generated_code,
                                    error_message=compile_error,
                                    model_choice=st.session_state.model_choice,
                                    quality=st.session_state.video_quality,
                                    max_retries=2
                                )
                            )

                        # Remove the warning banner now that we have a result
                        regen_banner.empty()

                        if regen_success:
                            # Update the code editor with the auto-fixed version
                            if regen_code:
                                st.session_state.generated_code = regen_code
                            st.session_state.video_path = regen_result
                            st.success(
                                "Regeneration successful! "
                                "The code was fixed automatically and the video is ready."
                            )
                        else:
                            # All retries exhausted - store the combined error for display
                            st.session_state.video_error = (
                                f"--- Initial compilation error ---\n{compile_error}\n\n"
                                f"--- Regeneration also failed ---\n{regen_result}"
                            )

                    st.rerun()
        
        with col_video_info:
            quality_info = {
                "low": "480p 15fps",
                "medium": "720p 30fps",
                "high": "1080p 60fps"
            }
            st.info(f"Quality: {quality_info.get(st.session_state.video_quality, 'medium')}")
        
        # Display video error if any
        if st.session_state.video_error:
            st.error("Video Compilation Failed (including auto-regeneration attempts)")
            with st.expander("View Error Details", expanded=True):
                st.code(st.session_state.video_error, language="text")
        
        # Display the video
        if st.session_state.video_path and os.path.exists(st.session_state.video_path):
            from shared_components import (
                render_video_player, render_teaching_script_section,
                render_merge_section, render_audio_preview,
            )

            render_video_player(
                st.session_state.video_path,
                download_name=os.path.basename(st.session_state.video_path),
                dl_key="studio_dl_mp4",
            )

            render_teaching_script_section(
                video_path=st.session_state.video_path,
                generated_code=st.session_state.generated_code,
                model_choice=st.session_state.model_choice,
                prompt_text=st.session_state.original_prompt or user_input,
                script_key="teaching_script",
                gen_id=st.session_state.generation_id,
                prefix="studio",
                get_video_duration_api=get_video_duration_api,
                generate_teaching_script_api=generate_teaching_script_api,
            )

            render_merge_section(
                video_path=st.session_state.video_path,
                script_key="teaching_script",
                final_video_key="final_video_path",
                final_error_key="final_video_error",
                prefix="studio",
                merge_video_audio_api=merge_video_audio_api,
                code_key="generated_code",
            )

            render_audio_preview(
                script_key="teaching_script",
                audio_key="teaching_audio_path",
                prefix="studio",
                generate_tts_api=generate_tts_api,
            )


# PHOTO TO ANIMATION PAGE

elif st.session_state.current_page == "Photo to Animation":

    health = check_backend_health()

    st.markdown(
        "Upload a photo of a solved math problem and turn it into "
        "an animated explainer video."
    )

    # Step 1 - Upload the image
    uploaded_file = st.file_uploader(
        "Upload a photo of a math problem",
        type=["jpg", "jpeg", "png"],
        help="Snap a picture of a textbook page or handwritten solution",
    )

    if uploaded_file is not None:
        # Show a preview so the user can verify it's the right image
        st.image(uploaded_file, caption="Uploaded image", width=400)

        # Read the raw bytes once and keep them handy
        image_bytes = uploaded_file.getvalue()
        mime_type = uploaded_file.type or "image/png"

        # Step 2 - Extract math content from the photo
        if st.button("Extract Math Content", type="primary"):
            if not health:
                st.error("Backend not connected. Please start the backend server.")
            else:
                # Reset downstream state when a new extraction starts
                st.session_state.photo_extracted_text = ""
                st.session_state.photo_confirmed_prompt = ""
                st.session_state.photo_generated_code = ""
                st.session_state.photo_video_path = None
                st.session_state.photo_video_error = None
                st.session_state.photo_teaching_script = ""
                st.session_state.photo_teaching_audio = None
                st.session_state.photo_final_video = None
                st.session_state.photo_final_error = None

                with st.spinner("Reading the image with AI vision model..."):
                    encoded = base64.b64encode(image_bytes).decode("utf-8")
                    ok, text = extract_from_image_api(encoded, mime_type)

                if ok:
                    st.session_state.photo_extracted_text = text
                else:
                    st.error(text)

                st.rerun()

    # Step 3 - Let the user review and edit the extracted text
    if st.session_state.photo_extracted_text:
        st.subheader("Extracted Content")
        st.caption(
            "Review the text below. Fix any mistakes before generating "
            "the animation - this is what the AI will animate."
        )

        edited_text = st.text_area(
            "Extracted math content:",
            value=st.session_state.photo_extracted_text,
            height=180,
            key=f"photo_editor_{st.session_state.photo_gen_id}",
        )

        if st.button("Confirm & Generate Code", type="primary"):
            if not edited_text.strip():
                st.error("The text area is empty. Please add a description.")
            elif not health:
                st.error("Backend not connected.")
            else:
                # Save the confirmed prompt and clear old generation output
                st.session_state.photo_confirmed_prompt = edited_text.strip()
                st.session_state.photo_generated_code = ""
                st.session_state.photo_video_path = None
                st.session_state.photo_video_error = None
                st.session_state.photo_teaching_script = ""
                st.session_state.photo_teaching_audio = None
                st.session_state.photo_final_video = None
                st.session_state.photo_final_error = None
                st.session_state.photo_gen_id += 1

                with st.spinner("Generating Manim code from your description..."):
                    prompt = (
                        "Create a step-by-step animated visualization of "
                        "the following math problem and its solution:\n\n"
                        + st.session_state.photo_confirmed_prompt
                    )
                    ok, code = generate_code_api(prompt, st.session_state.model_choice)
                    st.session_state.photo_generated_code = code

                st.rerun()

    # Step 4 - Show generated code with edit option
    if st.session_state.photo_generated_code:
        st.markdown("---")
        st.subheader("Generated Manim Code")
        st.code(st.session_state.photo_generated_code, language="python", line_numbers=True)

        st.download_button(
            label="Download Code",
            data=st.session_state.photo_generated_code,
            file_name="photo_manim_code.py",
            mime="text/x-python",
        )

        with st.expander("Edit Code", expanded=False):
            code_lines = st.session_state.photo_generated_code.count("\n") + 1
            code_height = min(max(150, code_lines * 20), 500)

            edited_code = st.text_area(
                "Modify the code below:",
                value=st.session_state.photo_generated_code,
                height=code_height,
                key=f"photo_code_editor_{st.session_state.photo_gen_id}",
            )

            col_save, col_info = st.columns([1, 3])
            with col_save:
                if st.button("Save Changes", key=f"photo_save_{st.session_state.photo_gen_id}"):
                    if edited_code != st.session_state.photo_generated_code:
                        st.session_state.photo_generated_code = edited_code
                        st.rerun()
            with col_info:
                if edited_code != st.session_state.photo_generated_code:
                    st.caption("Unsaved changes - click Save Changes")

        # Step 5 - Compile the video
        st.markdown("---")
        col_vid_btn, col_vid_info = st.columns([1, 2])

        with col_vid_btn:
            if st.button("Generate Video", type="primary", key="photo_gen_video"):
                if not health:
                    st.error("Backend not connected")
                else:
                    st.session_state.photo_video_path = None
                    st.session_state.photo_video_error = None

                    with st.spinner(
                        f"Compiling animation ({st.session_state.video_quality} quality)..."
                    ):
                        ok, result = compile_video_api(
                            st.session_state.photo_generated_code,
                            st.session_state.video_quality,
                        )

                    if ok:
                        st.session_state.photo_video_path = result
                    else:
                        # Auto-fix: send the error back to the LLM
                        compile_error = result
                        regen_banner = st.warning(
                            "Compilation failed. Regenerating the code. Please wait..."
                        )

                        with st.spinner("Analysing error and regenerating fixed code..."):
                            regen_ok, regen_result, regen_code = regenerate_and_compile_api(
                                prompt=st.session_state.photo_confirmed_prompt or "",
                                current_code=st.session_state.photo_generated_code,
                                error_message=compile_error,
                                model_choice=st.session_state.model_choice,
                                quality=st.session_state.video_quality,
                                max_retries=2,
                            )

                        regen_banner.empty()

                        if regen_ok:
                            if regen_code:
                                st.session_state.photo_generated_code = regen_code
                            st.session_state.photo_video_path = regen_result
                            st.success("Code fixed automatically - video is ready!")
                        else:
                            st.session_state.photo_video_error = (
                                f"--- Initial error ---\n{compile_error}\n\n"
                                f"--- Regeneration failed ---\n{regen_result}"
                            )

                    st.rerun()

        with col_vid_info:
            quality_labels = {"low": "480p 15fps", "medium": "720p 30fps", "high": "1080p 60fps"}
            st.info(f"Quality: {quality_labels.get(st.session_state.video_quality, 'medium')}")

        # Show compilation errors if any
        if st.session_state.photo_video_error:
            st.error("Video compilation failed (including auto-fix attempts)")
            with st.expander("View Error Details", expanded=True):
                st.code(st.session_state.photo_video_error, language="text")

        # Step 6 - Display the video + teaching script + narration
        if st.session_state.photo_video_path and os.path.exists(st.session_state.photo_video_path):
            from shared_components import (
                render_video_player, render_teaching_script_section,
                render_merge_section, render_audio_preview,
            )

            render_video_player(
                st.session_state.photo_video_path,
                download_name="photo_animation.mp4",
                dl_key="photo_dl_mp4",
            )

            render_teaching_script_section(
                video_path=st.session_state.photo_video_path,
                generated_code=st.session_state.photo_generated_code,
                model_choice=st.session_state.model_choice,
                prompt_text=(st.session_state.photo_confirmed_prompt
                             or st.session_state.photo_extracted_text),
                script_key="photo_teaching_script",
                gen_id=st.session_state.photo_gen_id,
                prefix="photo",
                get_video_duration_api=get_video_duration_api,
                generate_teaching_script_api=generate_teaching_script_api,
            )

            render_merge_section(
                video_path=st.session_state.photo_video_path,
                script_key="photo_teaching_script",
                final_video_key="photo_final_video",
                final_error_key="photo_final_error",
                prefix="photo",
                merge_video_audio_api=merge_video_audio_api,
                code_key="photo_generated_code",
            )

            render_audio_preview(
                script_key="photo_teaching_script",
                audio_key="photo_teaching_audio",
                prefix="photo",
                generate_tts_api=generate_tts_api,
            )


# CHATBOT PAGE - Manim Syntax Assistant

elif st.session_state.current_page == "Syntax Assistant":

    st.markdown("""
    <style>
    .small-font { font-size: 0.85em; color: #888; }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize chat input state
    if "chat_input_value" not in st.session_state:
        st.session_state.chat_input_value = ""
    
    # Helper function to format response 
    def format_response(content: str) -> None:
        """Render response with proper separation of text and code."""
        import re
        
        # Pattern for code blocks with ```python or ```
        code_pattern = r'```(?:python)?\s*\n?(.*?)```'
        
        if '```' in content:
            # Has proper code blocks - use them
            parts = re.split(code_pattern, content, flags=re.DOTALL)
            
            for i, part in enumerate(parts):
                part = part.strip()
                if not part:
                    continue
                if i % 2 == 1:
                    # This is code (odd index = captured group)
                    st.code(part, language="python")
                else:
                    # This is text - render as markdown
                    st.markdown(part)
        else:
            # No code blocks - check if there's actual code mixed in
            # Only detect REAL code patterns, not just any mention of manim
            code_start_patterns = [
                r'^from manim import',
                r'^import manim',
                r'^class \w+\(.*Scene.*\):',
            ]
            
            has_real_code = any(re.search(p, content, re.MULTILINE) for p in code_start_patterns)
            
            if has_real_code:
                _render_mixed_content(content)
            else:
                # Just text explanation - render as markdown
                st.markdown(content)
    
    def _render_mixed_content(content: str) -> None:
        """Handle content that mixes text and code without proper fencing."""
        lines = content.split('\n')
        buffer = []
        mode = 'text'
        
        for line in lines:
            stripped = line.strip()
            
            # Very specific code detection - must be actual Python code structure
            is_code_start = (
                line.startswith('from manim import') or
                line.startswith('import manim') or
                stripped.startswith('class ') and '(Scene)' in line or
                stripped.startswith('class ') and 'Scene' in line and ':' in line
            )
            
            is_code_continuation = (
                mode == 'code' and (
                    line.startswith('    ') or 
                    line.startswith('\t') or
                    stripped == '' or
                    stripped.startswith('def ') or
                    stripped.startswith('self.') or
                    stripped.startswith('#') or
                    stripped.startswith('return') or
                    '=' in stripped and not stripped.endswith(':')
                )
            )
            
            if is_code_start:
                # Flush text buffer first
                if buffer and mode == 'text':
                    st.markdown('\n'.join(buffer))
                    buffer = []
                mode = 'code'
                buffer.append(line)
            elif is_code_continuation:
                buffer.append(line)
            else:
                # This is text
                if mode == 'code' and buffer:
                    # Flush code buffer
                    st.code('\n'.join(buffer), language="python")
                    buffer = []
                mode = 'text'
                buffer.append(line)
        
        # Flush remaining buffer
        if buffer:
            if mode == 'code':
                st.code('\n'.join(buffer), language="python")
            else:
                st.markdown('\n'.join(buffer))
    
    # Display chat history first (main focus)
    if st.session_state.chatbot_history:
        # Show last 6 messages (3 Q&A pairs)
        history_to_show = st.session_state.chatbot_history[-6:]
        
        for msg in history_to_show:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    format_response(msg["content"])
                else:
                    st.markdown(msg["content"])
        
        # Clear button - small and subtle
        if st.button("Clear", key="clear_chat", type="secondary"):
            st.session_state.chatbot_history = []
            st.rerun()
    
    st.markdown("---")
    
    # Chat input using form to clear after submit (no page refresh)
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        
        with col_input:
            chat_question = st.text_input(
                "Ask about Manim:",
                placeholder="e.g., How to create a circle?",
                key="chat_input",
                label_visibility="collapsed"
            )
        
        with col_btn:
            ask_btn = st.form_submit_button("Ask", type="primary", use_container_width=True)
    
    # Options row - minimal
    include_examples = st.checkbox("Include code examples", value=True, key="include_examples")
    
    if ask_btn and chat_question.strip():
        with st.spinner("Generating..."):
            result = ask_chatbot(chat_question, include_examples=include_examples)
            st.session_state.chatbot_history.append({"role": "user", "content": chat_question})
            st.session_state.chatbot_history.append({
                "role": "assistant",
                "content": result.get("answer", "Sorry, couldn't generate a response.")
            })
        st.rerun()
    
    # Quick questions - hidden in expander with smaller text
    with st.expander("Quick questions", expanded=False):
        st.markdown('<p class="small-font">Click any question to ask instantly:</p>', unsafe_allow_html=True)
        
        quick_questions = [
            "How to create shapes?",
            "How to animate objects?",
            "MathTex vs Text?",
            "List common animations"
        ]
        
        for i, q in enumerate(quick_questions):
            if st.button(q, key=f"quick_q_{i}", use_container_width=True):
                with st.spinner("..."):
                    result = ask_chatbot(q, include_examples=True)
                    st.session_state.chatbot_history.append({"role": "user", "content": q})
                    st.session_state.chatbot_history.append({"role": "assistant", "content": result.get("answer", "No response.")})
                st.rerun()
    
    # Syntax lookup - hidden in expander
    with st.expander("Syntax lookup", expanded=False):
        st.markdown('<p class="small-font">Look up a specific Manim class:</p>', unsafe_allow_html=True)
        
        col_l1, col_l2 = st.columns([4, 1])
        with col_l1:
            quick_class = st.text_input("Class:", placeholder="Circle, FadeIn, MathTex...", key="quick_class_input", label_visibility="collapsed")
        with col_l2:
            if st.button("Go", key="quick_lookup_btn", use_container_width=True):
                if quick_class.strip():
                    with st.spinner("..."):
                        result = get_syntax_info(quick_class.strip())
                        if result.get("success"):
                            st.session_state.chatbot_history.append({"role": "user", "content": f"Syntax for: {quick_class}"})
                            st.session_state.chatbot_history.append({"role": "assistant", "content": result.get("answer", "No info found")})
                            st.rerun()

# FOOTER

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "Math2Visual Studio - Transform math concepts into visual animations"
    "</div>",
    unsafe_allow_html=True
)
