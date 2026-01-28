import streamlit as st
import requests
import os
import speech_recognition as sr
from tempfile import NamedTemporaryFile

# Configuration

# Backend URL 
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# Page Setup

st.set_page_config(
    page_title="Math2Visual Studio",
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Helper Functions

def check_backend_health():
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


def get_available_models():
    try:
        response = requests.get(f"{BACKEND_URL}/available_models", timeout=5)
        if response.status_code == 200:
            return response.json().get("models", [])
        return ["CodeLlama-34B", "Phi-2"]
    except requests.exceptions.RequestException:
        return ["CodeLlama-34B", "Phi-2"]


def generate_code_api(prompt, model_choice):
    # Call backend API to generate Manim code.
    try:
        response = requests.post(
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
        response = requests.post(
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


def generate_teaching_script_api(description, code, model_choice, video_duration=0.0):
    """Call backend API to generate teaching script matched to video duration."""
    try:
        response = requests.post(
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
        response = requests.post(
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
        response = requests.post(
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


def merge_video_audio_api(video_path, audio_text):
    try:
        response = requests.post(
            f"{BACKEND_URL}/merge_video_audio",
            json={"video_path": video_path, "audio_text": audio_text},
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
        response = requests.get(f"{BACKEND_URL}/chatbot/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"available": False, "ready": False}
    except requests.exceptions.RequestException:
        return {"available": False, "ready": False}


def ask_chatbot(question: str, include_examples: bool = False):
  
    try:
        response = requests.post(
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
        response = requests.get(
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


def capture_audio_input():
    recognizer = sr.Recognizer()
    
    # Settings for better recognition
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 2.0
    
    try:
        with sr.Microphone() as source:
            st.info("Adjusting for background noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            
            st.info("Listening for 10 seconds... (Speak now)")
            audio_data = recognizer.listen(source, timeout=10, phrase_time_limit=7)
            
            st.info("Processing your speech...")
            
            # Use Google Speech Recognition
            text = recognizer.recognize_google(audio_data)
            return text.lower()
            
    except sr.WaitTimeoutError:
        return "Error: No speech detected within 10 seconds."
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

if "teaching_audio_path" not in st.session_state:
    st.session_state.teaching_audio_path = None

if "speech_enabled" not in st.session_state:
    st.session_state.speech_enabled = False

if "final_video_path" not in st.session_state:
    st.session_state.final_video_path = None

if "final_video_error" not in st.session_state:
    st.session_state.final_video_error = None

# Chatbot session state
if "chatbot_history" not in st.session_state:
    st.session_state.chatbot_history = []

if "current_page" not in st.session_state:
    st.session_state.current_page = "Studio"


# Sidebar

with st.sidebar:
    # Navigation 
    st.header(" Navigation")
    current_page = st.radio(
        "Choose Mode:",
        [" Studio", " Syntax Assistant"],
        index=0 if st.session_state.current_page == "Studio" else 1,
        key="nav_radio"
    )
    
    # Update session state
    st.session_state.current_page = "Studio" if "Studio" in current_page else "Chatbot"
    
    st.markdown("---")
    
    # Backend Status
    st.header(" Settings")
    health = check_backend_health()
    
    if health:
        st.success("Backend: Connected")
    else:
        st.error("Backend: Not Connected")
        st.warning("Start backend with: python backend/app.py")
    
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
    
    # Chatbot status (only show when on chatbot page)
    if st.session_state.current_page == "Chatbot":
        st.markdown("---")
        chatbot_status = check_chatbot_status()
        if chatbot_status.get("ready"):
            st.success(" Chatbot Ready")
        elif chatbot_status.get("available"):
            st.warning(" Chatbot Loading...")
        else:
            st.info(" Chatbot Offline")



# Main Content

st.title("Math2Visual Studio")

                                     
# STUDIO PAGE

if st.session_state.current_page == "Studio":
    
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
        
        # Speech recognition button
        if st.button("Enable Speech Recognition"):
            st.session_state.speech_enabled = True
            st.rerun()
        
        # Handle speech recognition
        if st.session_state.speech_enabled:
            text_result = capture_audio_input()
            if not text_result.startswith("Error"):
                st.session_state.transcribed_text = text_result
                st.session_state.speech_enabled = False
                st.rerun()
            else:
                st.error(text_result)
                st.session_state.speech_enabled = False
        
        # Generate code button
        if st.button("Generate Code", type="primary"):
            if not user_input or not user_input.strip():
                st.error("Please enter a description first")
            elif not health:
                st.error("Backend not connected. Please start the backend server.")
            else:
                with st.spinner("Generating Manim script..."):
                    success, result = generate_code_api(user_input, model_choice)
                    st.session_state.generated_code = result
                st.rerun()


    # Output Section

    if st.session_state.generated_code:
        st.subheader("Generated Manim Code")
        
        # Editable code display
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
            if st.button("Generate Video", type="primary"):
                if not health:
                    st.error("Backend not connected")
                else:
                    with st.spinner(f"Compiling animation ({st.session_state.video_quality} quality)..."):
                        st.session_state.video_path = None
                        st.session_state.video_error = None
                        
                        success, result = compile_video_api(
                            st.session_state.generated_code,
                            st.session_state.video_quality
                        )
                        
                        if success:
                            st.session_state.video_path = result
                        else:
                            st.session_state.video_error = result
                    
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
            st.error("Video Compilation Failed")
            with st.expander("View Error Details", expanded=True):
                st.code(st.session_state.video_error, language="text")
        
        # Display the video
        if st.session_state.video_path and os.path.exists(st.session_state.video_path):
            st.markdown("---")
            st.subheader("Generated Animation")
            
            st.video(st.session_state.video_path)
            
            # Download button
            with open(st.session_state.video_path, "rb") as video_file:
                video_bytes = video_file.read()
                st.download_button(
                    label="Download Video",
                    data=video_bytes,
                    file_name=os.path.basename(st.session_state.video_path),
                    mime="video/mp4"
                )
            
            # Teaching Script Section
            st.markdown("---")
            st.subheader("Teaching Script")
            
            col_teach_btn, col_teach_info = st.columns([1, 2])
            
            with col_teach_btn:
                if st.button("Generate Teaching Script"):
                    with st.spinner("Creating explanation matched to video..."):
                        # Get video duration for script timing
                        video_duration = get_video_duration_api(st.session_state.video_path)
                        
                        success, result = generate_teaching_script_api(
                            user_input,
                            st.session_state.generated_code,
                            st.session_state.model_choice,
                            video_duration
                        )
                        
                        if success:
                            st.session_state.teaching_script = result
                        else:
                            st.error(result)
                    
                    st.rerun()
            
            with col_teach_info:
                # Show video duration info
                video_duration = get_video_duration_api(st.session_state.video_path)
                if video_duration > 0:
                    st.info(f"Video duration: {video_duration:.1f}s - Script will be generated to match")
                else:
                    st.info("The AI will explain the animation in simple words")
            
            # Display teaching script
            if st.session_state.teaching_script:
                st.markdown("---")
                st.subheader("Teaching Explanation")
                
                edited_script = st.text_area(
                    "Teaching explanation:",
                    value=st.session_state.teaching_script,
                    height=300,
                    key="teaching_script_editor"
                )
                
                if edited_script != st.session_state.teaching_script:
                    st.session_state.teaching_script = edited_script
                    st.session_state.teaching_audio_path = None
                    st.session_state.final_video_path = None
                
                # Create Final Synchronized Video Section
                st.markdown("---")
                st.subheader("Create Final Video with Narration")
                
                col_merge_btn, col_merge_info = st.columns([1, 2])
                
                with col_merge_btn:
                    if st.button("Create Synchronized Video", type="primary"):
                        with st.spinner("Merging animation with narration..."):
                            st.session_state.final_video_path = None
                            st.session_state.final_video_error = None
                            
                            success, result = merge_video_audio_api(
                                st.session_state.video_path,
                                st.session_state.teaching_script
                            )
                            
                            if success:
                                st.session_state.final_video_path = result
                            else:
                                st.session_state.final_video_error = result
                        
                        st.rerun()
                
                with col_merge_info:
                    st.info("Video and audio will be perfectly synchronized")
                
                # Display final video error if any
                if st.session_state.final_video_error:
                    st.error("Video Merging Failed")
                    with st.expander("View Error Details", expanded=True):
                        st.code(st.session_state.final_video_error, language="text")
                
                # Display the final synchronized video
                if st.session_state.final_video_path and os.path.exists(st.session_state.final_video_path):
                    st.markdown("---")
                    st.subheader("Final Video with Audio Narration")
                    st.success("Video and audio are now synchronized!")
                    
                    st.video(st.session_state.final_video_path)
                    
                    # Download button for final video
                    with open(st.session_state.final_video_path, "rb") as video_file:
                        video_bytes = video_file.read()
                        st.download_button(
                            label="Download Final Video",
                            data=video_bytes,
                            file_name=os.path.basename(st.session_state.final_video_path),
                            mime="video/mp4"
                        )
                
                # Audio-only playback section (optional)
                st.markdown("---")
                st.subheader("Audio Only (Preview)")
                
                col_audio1, col_audio2 = st.columns([1, 1])
                
                with col_audio1:
                    if st.button("Generate Audio Preview"):
                        with st.spinner("Converting to speech..."):
                            st.session_state.teaching_audio_path = generate_tts_api(
                                st.session_state.teaching_script
                            )
                        st.rerun()
                
                with col_audio2:
                    if st.session_state.teaching_audio_path:
                        st.success("Audio ready!")
                
                # Play audio
                if st.session_state.teaching_audio_path and os.path.exists(st.session_state.teaching_audio_path):
                    st.audio(st.session_state.teaching_audio_path, format="audio/mp3")
                    
                    with open(st.session_state.teaching_audio_path, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                        st.download_button(
                            label="Download Audio Only",
                            data=audio_bytes,
                            file_name="teaching_script_audio.mp3",
                            mime="audio/mp3"
                        )


# CHATBOT PAGE - Manim Syntax Assistant

elif st.session_state.current_page == "Chatbot":
    
    # Minimal styling
    st.markdown("""
    <style>
    .small-font { font-size: 0.85em; color: #888; }
    </style>
    """, unsafe_allow_html=True)
    
    # Display chat history first (main focus)
    if st.session_state.chatbot_history:
        # Show last 6 messages (3 Q&A pairs)
        history_to_show = st.session_state.chatbot_history[-6:]
        
        for msg in history_to_show:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Clear button - small and subtle
        if st.button("Clear", key="clear_chat", type="secondary"):
            st.session_state.chatbot_history = []
            st.rerun()
    
    st.markdown("---")
    
    # Chat input - compact layout with button on the right
    col_input, col_btn = st.columns([5, 1])
    
    with col_input:
        chat_question = st.text_input(
            "Ask about Manim:",
            placeholder="e.g., How to create a circle?",
            key="chat_input",
            label_visibility="collapsed"
        )
    
    with col_btn:
        ask_btn = st.button("Ask", key="ask_chatbot_btn", type="primary", use_container_width=True)
    
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
