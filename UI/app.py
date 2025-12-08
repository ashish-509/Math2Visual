import streamlit as st
import traceback
import logging
from src.stt.stt_core import load_stt_from_env
from src.pipeline.math2visual_pipeline import Math2VisualPipeline
from src.llm.client import LLMClient
from src.tts.tts_core import load_tts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Math2Visual", page_icon="🎬", layout="wide")

# Model init
if "model_status" not in st.session_state:
    st.session_state.model_status = {
        "stt_loaded": False,
        "stt_error": None,
        "llm_ready": False,
        "llm_error": None,
        "tts_ready": False,
        "tts_error": None
    }

# Load STT
if "stt" not in st.session_state:
    try:
        stt_instance, ok, err = load_stt_from_env()
        st.session_state.stt = stt_instance
        st.session_state.model_status["stt_loaded"] = ok
        st.session_state.model_status["stt_error"] = err
    except Exception as e:
        st.session_state.stt = None
        st.session_state.model_status["stt_loaded"] = False
        st.session_state.model_status["stt_error"] = str(e)

# Load LLM client
if "llm_client" not in st.session_state:
    try:
        client = LLMClient()
        st.session_state.llm_client = client
        st.session_state.model_status["llm_ready"] = True
    except Exception as e:
        st.session_state.llm_client = None
        st.session_state.model_status["llm_ready"] = False
        st.session_state.model_status["llm_error"] = str(e)

# Load TTS
if "tts" not in st.session_state:
    try:
        tts_instance, ok, err = load_tts()
        st.session_state.tts = tts_instance
        st.session_state.model_status["tts_ready"] = ok
        st.session_state.model_status["tts_error"] = err
    except Exception as e:
        st.session_state.tts = None
        st.session_state.model_status["tts_ready"] = False
        st.session_state.model_status["tts_error"] = str(e)

# Pipeline
if "pipeline" not in st.session_state:
    try:
        pipeline_inst = Math2VisualPipeline(llm_client=st.session_state.llm_client)
        st.session_state.pipeline = pipeline_inst
    except Exception as e:
        st.session_state.pipeline = None
        st.error("Failed to initialize pipeline: " + str(e))
        logger.exception("pipeline init failed")

# UI layout
with st.sidebar:
    st.title("Math2Visual")
    st.markdown("**Model Status**")
    st.write(st.session_state.model_status)
    st.markdown("---")
    st.markdown("Settings")
    st.selectbox("Model choice (placeholder)", ["CodeLlama-7B-Finetuned", "CodeLlama-13B"])
    st.slider("Max Duration (s)", 5, 300, 60)

st.title("🎬 Math2Visual — Speech → Math → Animation")
st.write("Speak or type a math prompt. The system will transcribe, parse math and generate Manim code.")

# Landing / Input area
col1, col2 = st.columns([2,3])

with col1:
    st.header("Input")
    # If STT available, show record button; else show message and text box
    if st.session_state.model_status["stt_loaded"]:
        if st.button("Record (3s)"):
            try:
                with st.spinner("Recording and transcribing..."):
                    text = st.session_state.stt.transcribe_from_mic(duration=3.0)
                    st.session_state.last_transcript = text
                    st.success("Transcription done")
            except Exception as e:
                st.session_state.last_transcript = ""
                st.error("STT error: " + str(e))
                st.session_state.model_status["stt_loaded"] = False
                st.session_state.model_status["stt_error"] = str(e)
    else:
        st.warning("STT model not loaded. Please type your prompt below.")

    user_prompt = st.text_area("Prompt (or paste transcription)", value=st.session_state.get("last_transcript", ""), height=160)

    if st.button("Generate Manim Code"):
        if not user_prompt.strip():
            st.warning("Please provide a prompt (via speech or typing).")
        else:
            try:
                with st.spinner("Generating code..."):
                    code, ok, err = st.session_state.pipeline.generate_code(user_prompt)
                    st.session_state.generated_code = code
                    if ok:
                        st.success("Code generated (LLM).")
                    else:
                        st.warning("Used fallback code (LLM failed). See logs for details.")
                        st.session_state.pipeline_error = err
            except Exception as e:
                st.error("Generation error: " + str(e))
                st.session_state.generated_code = ""
                logger.exception("generation failed")

with col2:
    st.header("Output")
    if "generated_code" in st.session_state and st.session_state.generated_code:
        st.subheader("Generated Manim Code")
        st.code(st.session_state.generated_code, language="python")
        st.markdown("---")
        st.subheader("Actions")
        st.download_button("Download code", st.session_state.generated_code, file_name="manim_scene.py")
        if st.button("Render locally (instructions)"):
            st.info("This will not run in the cloud. Run: `manim -pql manim_scene.py SceneClassName`")
    else:
        st.info("No code generated yet. Use the left panel to speak or type a prompt and press Generate Manim Code.")
