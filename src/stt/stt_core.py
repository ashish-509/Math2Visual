# Simple Speech-to-Text (STT) core for Math2Visual
# Uses Distil-Whisper (transformers) if available, else falls back to a dummy STT.

import os
import logging
import numpy as np
import sounddevice as sd
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class STTError(Exception):
    pass

class BaseSTT:
    def transcribe_buffer(self, audio: np.ndarray, sampling_rate: int) -> str:
        # Subclasses should implement this!
        raise NotImplementedError()

    def transcribe_from_mic(self, duration=3.0, sampling_rate=16000) -> str:
        """
        Record audio from the microphone and transcribe it.
        """
        audio = sd.rec(int(duration * sampling_rate), samplerate=sampling_rate, channels=1, dtype="float32")
        sd.wait()
        audio = np.squeeze(audio)
        return self.transcribe_buffer(audio, sampling_rate)

# Try to import transformers pipeline
try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

class DistilWhisperSTT(BaseSTT):
    def __init__(self, model_id="distil-whisper/distil-large-v3", device: Optional[int] = None):
        if not HAS_TRANSFORMERS:
            raise STTError("transformers not available")
        self.model_id = model_id
        self.device = device
        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=device if device is not None else -1,
                chunk_length_s=30
            )
            logger.info(f"Loaded STT pipeline {model_id}")
        except Exception as e:
            logger.error(f"Failed to load STT model: {e}")
            raise STTError(str(e))

    def transcribe_buffer(self, audio: np.ndarray, sampling_rate: int) -> str:
        try:
            out = self.pipe(audio, sampling_rate=sampling_rate)
            return out.get("text", "").strip()
        except Exception as e:
            logger.error(f"STT inference failed: {e}")
            raise STTError(str(e))

class FallbackSTT(BaseSTT):
    def __init__(self):
        logger.warning("Using fallback STT: returns empty string, manual input needed.")

    def transcribe_buffer(self, audio: np.ndarray, sampling_rate: int) -> str:
        # No real transcription, just a placeholder
        return ""

# Helper: Try to load STT model, else fallback

def load_stt_from_env():
    """
    Try to load preferred STT model. If fails, return FallbackSTT.
    Uses environment variables:
      M2V_STT_MODEL - model id string (optional)
      M2V_STT_DEVICE - 'cpu' or gpu index (optional)
    """
    model_id = os.environ.get("M2V_STT_MODEL", "distil-whisper/distil-large-v3")
    dev = os.environ.get("M2V_STT_DEVICE", None)
    device = None
    if dev is not None and dev.lower() != "cpu":
        try:
            device = int(dev)
        except Exception:
            device = None
    try:
        stt = DistilWhisperSTT(model_id=model_id, device=device)
        return stt, True, None
    except Exception as e:
        logger.error(f"Could not load DistilWhisper: {e}")
        return FallbackSTT(), False, str(e)
