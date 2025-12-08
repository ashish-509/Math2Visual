"""
STT pipeline using Distil-Whisper (transformers) + math parser + LLM prompt builder.

How it works:
- Captures audio from microphone in short chunks (CHUNK_SECONDS).
- A worker thread transcribes each chunk using the ASR pipeline.
- The transcribed natural-speech text is passed to a rule-based math parser to produce LaTeX-like output.
- The final 'clean prompt' is printed and forwarded to `send_to_llm()`
"""

import queue
import threading
import time
import sys
import math
import re
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# Transformers ASR pipeline
from transformers import pipeline

# CONFIG
MODEL_ID = "distil-whisper/distil-large-v3"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 3       # smaller -> lower latency but more requests
CHANNELS = 1
MAX_QUEUE_SIZE = 8

# ASR pipeline options (adjust device_map if GPU is available)
ASR_DEVICE = "cuda" if (os.getenv("CUDA_VISIBLE_DEVICES") is not None or False) else "cpu"

# SETUP: ASR pipeline
print("Loading ASR model... (this can take 10-60s)")
asr_pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_ID,
    device=0 if ASR_DEVICE == "cuda" else -1,
    chunk_length_s=30  # helpful for long audio segments
)
print("ASR model loaded on", ASR_DEVICE)

# AUDIO CAPTURE 
audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
stop_flag = threading.Event()

def audio_callback(indata, frames, time_info, status):
    """Callback runs in audio thread. Buffers frames into chunk buffer."""
    if status:
        print("Audio status:", status, file=sys.stderr)
    # Convert to float32 mono numpy array
    audio_callback.buf.append(indata.copy())

# bucket to hold small frames until chunk is full
audio_callback.buf = []

def producer_stream():
    """Continuously capture audio in CHUNK_SECONDS chunks and push to queue."""
    blocksize = int(SAMPLE_RATE * 0.1)  # callback will provide 100ms frames
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        blocksize=blocksize, callback=audio_callback):
        print(f">>> Listening (chunk {CHUNK_SECONDS}s). Press Ctrl+C to stop.")
        try:
            while not stop_flag.is_set():
                # accumulate until CHUNK_SECONDS reached
                target_frames = int(SAMPLE_RATE * CHUNK_SECONDS)
                collected = []
                collected_frames = 0
                while collected_frames < target_frames and not stop_flag.is_set():
                    if audio_callback.buf:
                        data = audio_callback.buf.pop(0)
                        collected.append(data)
                        collected_frames += data.shape[0]
                    else:
                        time.sleep(0.01)
                if collected_frames == 0:
                    continue
                chunk = np.concatenate(collected, axis=0)
                # Flatten to mono 1D
                if chunk.ndim > 1:
                    chunk = chunk.mean(axis=1)
                # Put into queue (non-blocking drop if full)
                try:
                    audio_queue.put_nowait(chunk.astype(np.float32))
                except queue.Full:
                    # if queue full, drop oldest and push new
                    try:
                        _ = audio_queue.get_nowait()
                        audio_queue.put_nowait(chunk.astype(np.float32))
                    except queue.Empty:
                        pass
        except KeyboardInterrupt:
            stop_flag.set()
            print("Stopping microphone capture...")

# MATH PARSER (rule-based)
# This parser converts spoken math English into LaTeX-like expressions.

REPLACEMENTS_BASIC = [
    (r"\bplus\b", "+"),
    (r"\bminus\b", "-"),
    (r"\btimes\b", r"\\times "),
    (r"\bmultiplied by\b", r"\\times "),
    (r"\bdivided by\b", "/"),
    (r"\bover\b", "/"),
    (r"\btheta\b", r"\\theta "),
    (r"\balpha\b", r"\\alpha "),
    (r"\bbeta\b", r"\\beta "),
    (r"\bsqrt of\b", r"sqrt "),
    (r"\bsquare root of\b", r"sqrt "),
    (r"\bequals\b", "="),
    (r"\bis\b", "="),
    (r"\bto the power of\b", "^"),
    (r"\bp o w e r\b", "^"),  # spoken letter spacing
]

# regex patterns for structured phrases
PATTERNS = [
    # exponent like "x squared" -> x^2
    (re.compile(r"([a-zA-Z0-9\\]+)\s+(squared|square)"), r"\1^2"),
    (re.compile(r"([a-zA-Z0-9\\]+)\s+(cubed|cube)"), r"\1^3"),
    (re.compile(r"([a-zA-Z0-9\\]+)\s+to the (\d+)(st|nd|rd|th)?"), r"\1^\2"),
    # power expressed as "x power y" or "x to the power y"
    (re.compile(r"([a-zA-Z0-9\\]+)\s+(?:to the )?power(?: of)?\s+([a-zA-Z0-9\\]+)"), r"\1^\2"),
    # fractions "a over b"
    (re.compile(r"([0-9a-zA-Z\\\{\}\^\_]+)\s+over\s+([0-9a-zA-Z\\\{\}\^\_]+)"), r"\\frac{\1}{\2}"),
    # integral "integral from a to b of <expr> dx"
    (re.compile(r"integral\s+from\s+([a-zA-Z0-9\-]+)\s+to\s+([a-zA-Z0-9\-]+)\s+of\s+(.+?)\s+d([a-zA-Z])$"), r"\\int_{\1}^{\2} \3 \, d\4"),
    # limit "limit as x tends to 0 of sin x over x"
    (re.compile(r"limit\s+as\s+([a-zA-Z])\s+(?:tends|goes)\s+to\s+([a-zA-Z0-9\-\+]+)\s+of\s+(.+)"), r"\\lim_{\1 \\to \2} \3"),
    # sqrt "sqrt <expr>"
    (re.compile(r"\bsqrt\s+of\s+(.+)"), r"\\sqrt{\1}"),
]

def clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def basic_replacements(s: str) -> str:
    out = s.lower()
    for pat, rep in REPLACEMENTS_BASIC:
        out = re.sub(pat, rep, out)
    return clean_whitespace(out)

def apply_patterns(s: str) -> str:
    out = s
    # attempt iterative pattern application
    for pattern, rep in PATTERNS:
        out_new = pattern.sub(rep, out)
        if out_new != out:
            out = out_new
    return clean_whitespace(out)

def math_speech_to_latex(spoken: str) -> str:
    # Convert spoken math text (from ASR) into a LaTeX-ish expression.
    if not spoken or len(spoken.strip()) == 0:
        return ""
    s = spoken.strip()
    s = basic_replacements(s)
    s = apply_patterns(s)

    # numeric words -> digits (simple map; extend as needed)
    num_map = {
        "zero":"0", "one":"1","two":"2","three":"3","four":"4","five":"5",
        "six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
        "negative 1":"-1", "negative one":"-1"
    }
    for w, d in num_map.items():
        s = re.sub(rf"\b{w}\b", d, s)

    # tidy some common constructs
    s = re.sub(r"\b(x|y|z) (\^|\^) (\d+)", r"\1^\3", s)  # safe-guard
    s = re.sub(r"\s+([,+\-*=\/\)\}])", r"\1", s)
    s = re.sub(r"([^\s])\s+([^\s])", r"\1 \2", s)  # one-space between tokens

    return s

# LLM PROMPT BUILDER
def build_llm_prompt(original_text: str, latex_expr: Optional[str]):
    """
    Create a clear instruction prompt for code-generating model.
    Replace or extend to match the exact instruction style used when finetuning Code-LLaMA.
    """
    prompt = "Task: Generate a Manim (Python) script that visualizes the following mathematical request.\n\n"
    prompt += f"User speech (raw): \"{original_text}\"\n\n"
    if latex_expr:
        prompt += f"Interpreted math (LaTeX-like): `{latex_expr}`\n\n"
    prompt += (
        "Requirements:\n"
        "- Produce a self-contained Manim Community compatible Python script.\n"
        "- Include comments explaining each scene and parameters.\n"
        "- Ensure the script renders a clear animation illustrating the math.\n"
        "- Use simple, well-commented code suitable for educational use.\n\n"
        "Output only the Python code (no extra explanation)."
    )
    return prompt

# ASR Consumer 
def send_to_llm(prompt: str):
    """
    Stub: Replace this with call to the finetuned Code-LLaMA + RAG service.
    send prompt to your local LLaMA server, or an API endpoint, and return the code.
    """
    print("\n>>> [LLM PROMPT READY] (length: %d chars)\n%s\n" % (len(prompt), prompt[:800]))

    return None

def worker_transcribe_and_parse():
    """Continuously take audio chunks from queue, run ASR, parse math, and build final prompts."""
    while not stop_flag.is_set():
        try:
            chunk = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        # Convert float32 numpy to int16 WAV bytes (required by some pipelines; transformers accepts numpy)
        # Sample rate is SAMPLE_RATE
        # For transformers' pipeline, supplying numpy array with 'sampling_rate' works.
        try:
            asr_result = asr_pipe(chunk, sampling_rate=SAMPLE_RATE)
        except Exception as e:
            print("ASR error:", e)
            continue

        text = asr_result.get("text", "").strip()
        if not text:
            continue

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"\n[{timestamp}] ASR -> {text}")

        # Math parsing
        latex = math_speech_to_latex(text)
        if latex:
            print("Parsed LaTeX-ish:", latex)
        else:
            print("No math parsed (plain text).")

        # Build final prompt for code generator
        prompt = build_llm_prompt(text, latex)
        # Send to LLM (stub)
        send_to_llm(prompt)

# MAIN-
def main():
    producer = threading.Thread(target=producer_stream, name="AudioProducer", daemon=True)
    consumer = threading.Thread(target=worker_transcribe_and_parse, name="ASRWorker", daemon=True)
    consumer.start()
    producer.start()

    try:
        while producer.is_alive() and consumer.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        stop_flag.set()
        print("Exiting...")

if __name__ == "__main__":
    main()
