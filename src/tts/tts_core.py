import os
import logging
logger = logging.getLogger(__name__)

class TTSFallback:
    def __init__(self):
        logger.info("TTS fallback initialized. It does not synthesize audio.")

    def synthesize(self, text: str, out_path: str) -> bool:
        # Placeholder: write script to text file for now.
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        return True

def load_tts():
    # Attempt to load heavier TTS; if fails, return fallback
    try:
        # TODO: load real TTS
        raise RuntimeError("No real TTS installed (placeholder)")
    except Exception as e:
        return TTSFallback(), False, str(e)
