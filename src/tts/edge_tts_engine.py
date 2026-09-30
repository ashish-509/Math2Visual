"""Edge-TTS wrapper with optional SSML pause insertion.

Falls back to gTTS when edge-tts is unavailable.
"""

import asyncio
import logging
import os
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

# natural-sounding English voice
DEFAULT_VOICE = "en-US-AriaNeural"


def _build_ssml(text: str, pauses: Optional[List[Tuple[int, int]]] = None,
                voice: str = DEFAULT_VOICE) -> str:
    """Wrap plain text in SSML, inserting <break> tags at pause positions.

    Args:
        text: plain narration text
        pauses: list of (char_offset, pause_ms) tuples
        voice: edge-tts voice identifier
    """
    escaped = (text.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;"))

    if not pauses:
        body = escaped
    else:
        # insert breaks from end to start so offsets stay valid
        chars = list(escaped)
        for offset, ms in sorted(pauses, reverse=True):
            pos = min(offset, len(chars))
            chars.insert(pos, f' <break time="{ms}ms"/> ')
        body = "".join(chars)

    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
        f'xml:lang="en-US">'
        f'<voice name="{voice}">{body}</voice></speak>'
    )


def compute_pause_points(narration: str, scene_events=None) -> List[Tuple[int, int]]:
    """Heuristic: insert short pauses at sentence boundaries aligned to scene gaps.

    If scene_events is provided (list of SceneEvent), pauses are weighted
    toward timestamps where self.wait() or long gaps appear in the animation.
    """
    if not narration:
        return []

    # find sentence-ending positions
    sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s', narration)]

    if not sentence_ends:
        return []

    # if no scene events, add a small pause at every other sentence boundary
    if not scene_events:
        return [(pos, 350) for i, pos in enumerate(sentence_ends) if i % 2 == 1]

    # with scene events: figure out which fraction of narration each event maps to
    pauses_list: List[Tuple[int, int]] = []
    wait_fracs = []
    total_dur = sum(ev.duration for ev in scene_events) or 1.0
    elapsed = 0.0
    for ev in scene_events:
        if ev.label == "pause" and ev.duration >= 0.5:
            frac = elapsed / total_dur
            pause_ms = min(int(ev.duration * 400), 1200)  # cap at 1.2s
            wait_fracs.append((frac, pause_ms))
        elapsed += ev.duration

    # map each fraction to the nearest sentence boundary
    text_len = len(narration) or 1
    used = set()
    for frac, ms in wait_fracs:
        target_pos = int(frac * text_len)
        best = min(sentence_ends, key=lambda p: abs(p - target_pos), default=None)
        if best is not None and best not in used:
            pauses_list.append((best, ms))
            used.add(best)

    return pauses_list


async def _generate_edge_tts(text: str, output_path: str,
                             voice: str = DEFAULT_VOICE,
                             pauses: Optional[List[Tuple[int, int]]] = None) -> str:
    """Generate speech using edge-tts. Returns output path."""
    if pauses:
        ssml = _build_ssml(text, pauses, voice)
        comm = edge_tts.Communicate(ssml, voice)
    else:
        comm = edge_tts.Communicate(text, voice)
    await comm.save(output_path)
    return output_path


def generate_speech(text: str, output_path: str,
                    voice: str = DEFAULT_VOICE,
                    scene_events=None) -> Optional[str]:
    """High-level API: generate TTS audio, return file path or None.

    Uses edge-tts with SSML pauses when available, otherwise falls back
    to gTTS.
    """
    if not text or not text.strip():
        return None

    if EDGE_TTS_AVAILABLE:
        try:
            pauses = None  # edge-tts >= 7 has no SSML support: markup would be read aloud
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    _generate_edge_tts(text, output_path, voice, pauses)
                )
            finally:
                loop.close()
            logger.info("edge-tts: saved %s", output_path)
            return output_path
        except Exception as e:
            logger.warning("edge-tts failed, falling back to gTTS: %s", e)

    # fallback: gTTS
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(output_path)
        logger.info("gTTS fallback: saved %s", output_path)
        return output_path
    except Exception as e:
        logger.error("TTS generation failed entirely: %s", e)
        return None
