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


# ANIMATION FEATURES API FUNCTIONS

def get_template_categories():
    try:
        response = requests.get(f"{BACKEND_URL}/templates/categories", timeout=10)
        if response.status_code == 200:
            return response.json().get("categories", [])
        return []
    except:
        return []


def get_templates_in_category(category):
    try:
        response = requests.get(f"{BACKEND_URL}/templates/{category}", timeout=10)
        if response.status_code == 200:
            return response.json().get("templates", [])
        return []
    except:
        return []


def get_template_code(category, template_id, theme=None):
    try:
        url = f"{BACKEND_URL}/templates/{category}/{template_id}"
        if theme:
            url += f"?theme={theme}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_color_themes():
    try:
        response = requests.get(f"{BACKEND_URL}/themes", timeout=10)
        if response.status_code == 200:
            return response.json().get("themes", [])
        return []
    except:
        return []


def apply_theme_to_code(code, theme_id):
    try:
        response = requests.post(
            f"{BACKEND_URL}/themes/apply",
            json={"code": code, "theme_id": theme_id},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return data.get("code", code)
        return code
    except:
        return code


def convert_video_format(video_path, target_format, quality="medium"):
    try:
        response = requests.post(
            f"{BACKEND_URL}/video/convert",
            json={"video_path": video_path, "target_format": target_format, "quality": quality},
            timeout=300
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return True, data.get("output_path")
            return False, data.get("error", "Conversion failed")
        return False, "Backend error"
    except Exception as e:
        return False, str(e)


def change_video_speed(video_path, speed_factor):
    try:
        response = requests.post(
            f"{BACKEND_URL}/video/speed",
            json={"video_path": video_path, "speed_factor": speed_factor},
            timeout=300
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return True, data.get("output_path")
            return False, data.get("error", "Speed change failed")
        return False, "Backend error"
    except Exception as e:
        return False, str(e)


def get_supported_formats():
    try:
        response = requests.get(f"{BACKEND_URL}/video/formats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"formats": ["mp4"], "ffmpeg_available": False}
    except:
        return {"formats": ["mp4"], "ffmpeg_available": False}


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

if "original_prompt" not in st.session_state:
    st.session_state.original_prompt = ""

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

# Animation features session state
if "selected_theme" not in st.session_state:
    st.session_state.selected_theme = "default"

if "selected_template_category" not in st.session_state:
    st.session_state.selected_template_category = None

if "video_speed" not in st.session_state:
    st.session_state.video_speed = 1.0

if "export_format" not in st.session_state:
    st.session_state.export_format = "mp4"

# Generation counter to create unique widget keys (prevents stale cache)
if "generation_id" not in st.session_state:
    st.session_state.generation_id = 0


# Sidebar

with st.sidebar:
    # Navigation 
    st.header("Navigation")
    current_page = st.radio(
        "Choose Mode:",
        ["Studio", "Templates", "Syntax Assistant"],
        index=["Studio", "Templates", "Syntax Assistant"].index(
            st.session_state.current_page if st.session_state.current_page in ["Studio", "Templates", "Syntax Assistant"] else "Studio"
        ),
        key="nav_radio"
    )
    
    # Update session state
    st.session_state.current_page = current_page
    
    # Only show settings for Studio page
    if st.session_state.current_page == "Studio":
        st.markdown("---")
        
        # Backend Status
        st.header("Settings")
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
        
        # Color Theme Selection
        st.markdown("---")
        st.subheader("Color Theme")
        
        themes = get_color_themes()
        if themes:
            theme_names = ["default"] + [t["id"] for t in themes if t["id"] != "default"]
            theme_display = {t["id"]: t["name"] for t in themes}
            
            selected_theme = st.selectbox(
                "Animation Colors",
                theme_names,
                format_func=lambda x: theme_display.get(x, x.title()),
                index=theme_names.index(st.session_state.selected_theme) if st.session_state.selected_theme in theme_names else 0
            )
            
            if selected_theme != st.session_state.selected_theme:
                st.session_state.selected_theme = selected_theme
            
            # Show color preview
            selected_theme_data = next((t for t in themes if t["id"] == selected_theme), None)
            if selected_theme_data and selected_theme_data.get("preview_colors"):
                cols = st.columns(len(selected_theme_data["preview_colors"]))
                for i, color in enumerate(selected_theme_data["preview_colors"]):
                    with cols[i]:
                        st.markdown(
                            f'<div style="background-color:{color};width:30px;height:30px;border-radius:4px;"></div>',
                            unsafe_allow_html=True
                        )
        
        # Video Speed Control
        st.markdown("---")
        st.subheader("Playback Speed")
        
        speed = st.slider(
            "Video Speed",
            min_value=0.25,
            max_value=2.0,
            value=st.session_state.video_speed,
            step=0.25,
            help="Adjust playback speed (0.25x to 2x)"
        )
        
        if speed != st.session_state.video_speed:
            st.session_state.video_speed = speed
        
        # Export Format
        st.markdown("---")
        st.subheader("Export Format")
        
        format_info = get_supported_formats()
        available_formats = format_info.get("formats", ["mp4"])
        
        export_format = st.selectbox(
            "Export As",
            available_formats,
            index=available_formats.index(st.session_state.export_format) if st.session_state.export_format in available_formats else 0
        )
        
        if export_format != st.session_state.export_format:
            st.session_state.export_format = export_format
        
        if not format_info.get("ffmpeg_available", False):
            st.warning("FFmpeg not found. Only MP4 export available.")


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
                    
                    # Apply color theme if not default
                    if success and st.session_state.selected_theme != "default":
                        result = apply_theme_to_code(result, st.session_state.selected_theme)
                    
                    st.session_state.generated_code = result
                st.rerun()


    # Output Section

    if st.session_state.generated_code:
        st.subheader("Generated Manim Code")
        
        # Display syntax-highlighted code with copy functionality
        st.code(st.session_state.generated_code, language="python", line_numbers=True)
        
        # Copy/Download buttons row
        col_copy, col_theme_apply, col_edit_toggle = st.columns([1, 1, 2])
        
        with col_copy:
            # Download as .py file (also works as copy - user can open and copy)
            st.download_button(
                label="Download Code",
                data=st.session_state.generated_code,
                file_name="generated_manim_code.py",
                mime="text/x-python",
                help="Download the code as a .py file"
            )
        
        with col_theme_apply:
            if st.button("Apply Theme", help="Apply the selected color theme to the code"):
                themed_code = apply_theme_to_code(
                    st.session_state.generated_code,
                    st.session_state.selected_theme
                )
                st.session_state.generated_code = themed_code
                st.rerun()
        
        # Expandable section for editing
        with st.expander("Edit Code", expanded=False):
            edited_code = st.text_area(
                "Modify the code below:",
                value=st.session_state.generated_code,
                height=400,
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
            
            # Video options row
            col_download, col_speed, col_format = st.columns(3)
            
            with col_download:
                with open(st.session_state.video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                    st.download_button(
                        label="Download MP4",
                        data=video_bytes,
                        file_name=os.path.basename(st.session_state.video_path),
                        mime="video/mp4"
                    )
            
            with col_speed:
                if st.session_state.video_speed != 1.0:
                    if st.button(f"Apply {st.session_state.video_speed}x Speed"):
                        with st.spinner("Adjusting video speed..."):
                            success, result = change_video_speed(
                                st.session_state.video_path,
                                st.session_state.video_speed
                            )
                            if success:
                                st.success("Speed adjusted!")
                                with open(result, "rb") as f:
                                    st.download_button(
                                        label="Download Speed-Adjusted",
                                        data=f.read(),
                                        file_name=f"video_{st.session_state.video_speed}x.mp4",
                                        mime="video/mp4",
                                        key="speed_download"
                                    )
                            else:
                                st.error(f"Failed: {result}")
            
            with col_format:
                if st.session_state.export_format != "mp4":
                    if st.button(f"Convert to {st.session_state.export_format.upper()}"):
                        with st.spinner(f"Converting to {st.session_state.export_format}..."):
                            success, result = convert_video_format(
                                st.session_state.video_path,
                                st.session_state.export_format,
                                st.session_state.video_quality
                            )
                            if success:
                                st.success("Converted!")
                                if st.session_state.export_format == "gif":
                                    with open(result, "rb") as f:
                                        st.download_button(
                                            label="Download GIF",
                                            data=f.read(),
                                            file_name="animation.gif",
                                            mime="image/gif",
                                            key="gif_download"
                                        )
                                elif st.session_state.export_format == "webm":
                                    with open(result, "rb") as f:
                                        st.download_button(
                                            label="Download WebM",
                                            data=f.read(),
                                            file_name="animation.webm",
                                            mime="video/webm",
                                            key="webm_download"
                                        )
                                else:
                                    st.info(f"Saved to: {result}")
                            else:
                                st.error(f"Failed: {result}")
            
            # Teaching Script Section
            st.markdown("---")
            st.subheader("Teaching Script")
            
            col_teach_btn, col_teach_info = st.columns([1, 2])
            
            with col_teach_btn:
                if st.button("Generate Teaching Script"):
                    # Use stored original prompt, fallback to current input, or extract from code
                    prompt_for_script = st.session_state.original_prompt or user_input
                    
                    # If still no prompt, try to extract topic from the code
                    if not prompt_for_script or not prompt_for_script.strip():
                        # Extract class name from code as a fallback description
                        import re
                        code = st.session_state.generated_code
                        class_match = re.search(r'class\s+(\w+)', code)
                        if class_match:
                            class_name = class_match.group(1)
                            # Convert CamelCase to readable text
                            readable_name = re.sub(r'([A-Z])', r' \1', class_name).strip()
                            prompt_for_script = f"Explain the concept shown in this {readable_name} animation"
                    
                    if not prompt_for_script or not prompt_for_script.strip():
                        st.error("No description available. Please enter a description in the text area above.")
                    else:
                        with st.spinner("Creating explanation matched to video..."):
                            # Get video duration for script timing
                            video_duration = get_video_duration_api(st.session_state.video_path)
                            
                            success, result = generate_teaching_script_api(
                                prompt_for_script,
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
                    key=f"teaching_script_editor_{st.session_state.generation_id}"
                )
                
                col_save_script, col_script_info = st.columns([1, 3])
                with col_save_script:
                    if st.button("Save Script Changes", key=f"save_script_{st.session_state.generation_id}"):
                        if edited_script != st.session_state.teaching_script:
                            st.session_state.teaching_script = edited_script
                            st.session_state.teaching_audio_path = None
                            st.session_state.final_video_path = None
                            st.rerun()
                with col_script_info:
                    if edited_script != st.session_state.teaching_script:
                        st.caption("Unsaved changes - click Save Script Changes")
                
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


# TEMPLATES PAGE - Animation Template Library

elif st.session_state.current_page == "Templates":
    
    st.subheader("Animation Templates Library")
    st.write("Choose from pre-built animation templates for common math concepts.")
    
    # Get template categories
    categories = get_template_categories()
    
    if not categories:
        st.warning("Could not load templates. Make sure the backend is running.")
    else:
        # Category selection
        col_cat, col_theme = st.columns([2, 1])
        
        with col_cat:
            selected_category = st.selectbox(
                "Select Category",
                categories,
                format_func=lambda x: x.replace("_", " ").title()
            )
        
        with col_theme:
            themes = get_color_themes()
            theme_options = [t["id"] for t in themes] if themes else ["default"]
            theme_display = {t["id"]: t["name"] for t in themes} if themes else {"default": "Default"}
            
            template_theme = st.selectbox(
                "Color Theme",
                theme_options,
                format_func=lambda x: theme_display.get(x, x.title())
            )
        
        st.markdown("---")
        
        # Get templates in selected category
        if selected_category:
            templates = get_templates_in_category(selected_category)
            
            if templates:
                # Display templates in a grid
                cols_per_row = 2
                for i in range(0, len(templates), cols_per_row):
                    row_templates = templates[i:i+cols_per_row]
                    cols = st.columns(cols_per_row)
                    
                    for j, template in enumerate(row_templates):
                        with cols[j]:
                            with st.container():
                                st.markdown(f"**{template['name']}**")
                                st.caption(template['description'])
                                
                                if st.button("Use Template", key=f"template_{template['id']}"):
                                    # Load template code
                                    template_data = get_template_code(
                                        selected_category,
                                        template['id'],
                                        theme=template_theme
                                    )
                                    
                                    if template_data and template_data.get("code"):
                                        st.session_state.generated_code = template_data["code"]
                                        # Store template info as the prompt for teaching script
                                        st.session_state.original_prompt = f"{template['name']}: {template['description']}"
                                        st.session_state.current_page = "Studio"
                                        st.rerun()
                                    else:
                                        st.error("Could not load template code")
                                
                                st.markdown("---")
            else:
                st.info(f"No templates found in {selected_category}")
        
        # 3D Templates note
        if selected_category == "3d_scenes":
            st.info("3D templates use ThreeDScene for 3D visualizations with camera rotation.")


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
