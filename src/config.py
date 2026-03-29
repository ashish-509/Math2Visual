"""Centralized configuration for the Math2Visual backend.

All magic numbers, model names, timeouts, and path defaults live here
so they can be overridden via environment variables in one place.
"""

import os

# Paths 
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DOCS_PATH = os.path.join(PROJECT_ROOT, "crawler", "markdown_output.md")

# Model names (Groq) 
DEFAULT_MODEL = "CodeLlama-34B"
GROQ_MODEL_CODELLAMA = os.getenv("GROQ_MODEL_CODELLAMA", "llama-3.3-70b-versatile")
GROQ_MODEL_PHI2 = os.getenv("GROQ_MODEL_PHI2", "llama-3.1-8b-instant")
MODEL_CHOICES = ["CodeLlama-34B", "Phi-2", "Mistral-7B (Finetuned)"]

# Video quality presets 
QUALITY_FLAGS = {
    "low": "-ql",      # 480p 15fps
    "medium": "-qm",   # 720p 30fps
    "high": "-qh",     # 1080p 60fps
}
QUALITY_LABELS = {
    "low": "480p 15fps",
    "medium": "720p 30fps",
    "high": "1080p 60fps",
}
DEFAULT_QUALITY = "medium"

# Timeouts (seconds) 
MANIM_RENDER_TIMEOUT = int(os.getenv("MANIM_RENDER_TIMEOUT", "300"))
MANIM_DRYRUN_TIMEOUT = int(os.getenv("MANIM_DRYRUN_TIMEOUT", "30"))
FFMPEG_TIMEOUT = int(os.getenv("FFMPEG_TIMEOUT", "300"))
FFPROBE_TIMEOUT = 30
TTS_TIMEOUT = 60

# LLM generation defaults 
CODE_MAX_TOKENS = int(os.getenv("CODE_MAX_TOKENS", "2048"))
CODE_TEMPERATURE = float(os.getenv("CODE_TEMPERATURE", "0.7"))
NARRATION_MAX_TOKENS = 1024
NARRATION_TEMPERATURE = 0.8
SPEAKING_RATE_WPS = 2.5  # words per second for script length estimation

# Concurrency 
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
MAX_RETRIES_COMPILE = 3
MAX_RETRIES_REGEN = 3

# Render cache 
RENDER_CACHE_MAX = int(os.getenv("RENDER_CACHE_MAX", "200"))

# TTS 
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")
TTS_LANG = "en"

# Backend server 
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
BACKEND_URL = os.getenv("BACKEND_URL", f"http://localhost:{BACKEND_PORT}")

# Audio encoding 
AUDIO_BITRATE = "192k"
AUDIO_CODEC = "aac"

# speed adjustment bounds for video-audio sync
SPEED_FACTOR_MIN = 0.5
SPEED_FACTOR_MAX = 2.0
