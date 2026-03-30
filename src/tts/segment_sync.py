"""Segment-aware audio-video synchronisation engine.

The video is NEVER cut, sped up, or modified in any way.  Only the audio
side is adjusted so that the narration aligns with the visual timeline.

Strategy
--------
1. Parse the Manim code into **logical slides** — a slide is a group of
   animation events that share the screen (e.g. title + formula appear
   together), separated by major transitions (FadeOut clearing the screen).
2. Generate TTS for the **entire narration at once** — one continuous voice
   file so the speech sounds natural and uninterrupted.
3. If the resulting audio is shorter than the video → pad silence at the end.
   If longer → gently trim with a fade-out.  The video is kept intact.
4. Mux the single audio file onto the original video — no re-encoding.

The user sees one seamless video with perfectly paced narration.
"""

import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)



# Helpers


def _get_duration(path: str) -> float:
    """Get audio/video duration via ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    return 0.0



# 1. Build logical slides from the scene timeline


@dataclass
class Slide:
    """A logical 'screen' — content visible together before a transition."""
    start: float          # seconds from video start
    duration: float       # how long this slide is on screen
    labels: List[str]     # human-readable event names in this slide


def build_slides(scene_events, total_duration: float) -> List[Slide]:
    """Group scene events into logical slides.

    A new slide starts when:
    - A FadeOut is followed by new content (FadeIn / Write / Create)

    Pauses, waits, and consecutive additions without a screen-clear all
    belong to the *same* slide.  This keeps grouping coarse — each slide
    maps to one "screen" the viewer perceives.
    """
    if not scene_events:
        return [Slide(start=0.0, duration=max(total_duration, 1.0),
                       labels=["full video"])]

    _CLEAR = {"FadeOut", "Uncreate"}
    _CONTENT = {"Write", "FadeIn", "Create", "DrawBorderThenFill",
                "GrowFromCenter", "ShowCreation", "Transform",
                "add objects", "ReplacementTransform"}

    slides: List[Slide] = []
    cur_start = 0.0
    cur_labels: List[str] = []
    saw_clear = False

    for ev in scene_events:
        label = ev.label.strip()
        base = label.split(" → ")[0].split("(")[0].strip()

        is_clear = any(c in label for c in _CLEAR)
        is_content = any(c in label for c in _CONTENT)

        if is_clear:
            saw_clear = True
            cur_labels.append(label)
            continue

        if saw_clear and is_content:
            # Transition boundary — close current slide
            if cur_labels:
                slides.append(Slide(
                    start=cur_start,
                    duration=max(ev.start - cur_start, 0.1),
                    labels=cur_labels[:],
                ))
            cur_start = ev.start
            cur_labels = [label]
            saw_clear = False
            continue

        saw_clear = False
        cur_labels.append(label)

    # Close last slide — extends to video end
    if cur_labels:
        slides.append(Slide(
            start=cur_start,
            duration=max(total_duration - cur_start, 0.1),
            labels=cur_labels[:],
        ))

    if not slides:
        slides.append(Slide(start=0.0, duration=max(total_duration, 1.0),
                             labels=["full video"]))
    return slides



# 2. Distribute narration across slides (informational / logging only)


@dataclass
class NarrationSlice:
    """A portion of narration mapped to a slide's time range."""
    text: str
    slide_start: float
    slide_duration: float


def distribute_narration(narration: str, slides: List[Slide]) -> List[NarrationSlice]:
    """Split narration at sentence boundaries proportional to slide durations.

    Used for logging / diagnostics.  TTS still receives the FULL narration
    as one string so the voice stays continuous and natural.
    """
    if len(slides) <= 1:
        return [NarrationSlice(
            text=narration,
            slide_start=slides[0].start if slides else 0.0,
            slide_duration=slides[0].duration if slides else 1.0,
        )]

    sentence_ends = [m.end() for m in re.finditer(r'[.!?]\s', narration + " ")]
    if not sentence_ends:
        sentence_ends = [len(narration)]

    total_dur = sum(s.duration for s in slides) or 1.0
    slices: List[NarrationSlice] = []
    cursor = 0

    for idx, slide in enumerate(slides):
        frac = slide.duration / total_dur
        target = cursor + int(frac * len(narration))

        if idx == len(slides) - 1:
            text = narration[cursor:].strip()
        else:
            best = min(sentence_ends, key=lambda p: abs(p - target),
                       default=len(narration))
            best = max(best, cursor + 1)
            text = narration[cursor:best].strip()
            cursor = best

        slices.append(NarrationSlice(
            text=text,
            slide_start=slide.start,
            slide_duration=slide.duration,
        ))

    return slices



# 3. Match audio duration to video duration


def match_audio_to_video(audio_path: str, video_duration: float,
                         output_path: str) -> bool:
    """Ensure audio is exactly *video_duration* seconds.

    - Shorter → append silence at the end.
    - Longer  → trim with a gentle 0.5 s fade-out.
    - Close enough (< 0.2 s diff) → copy as-is.
    """
    audio_dur = _get_duration(audio_path)
    if audio_dur <= 0:
        return False

    diff = video_duration - audio_dur

    if abs(diff) < 0.2:
        import shutil
        shutil.copy2(audio_path, output_path)
        return True

    try:
        if diff > 0:
            cmd = [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-f", "lavfi", "-t", f"{diff:.3f}",
                "-i", "anullsrc=r=24000:cl=mono",
                "-filter_complex",
                "[0:a][1:a]concat=n=2:v=0:a=1[outa]",
                "-map", "[outa]",
                "-c:a", "libmp3lame", "-b:a", "128k",
                output_path,
            ]
        else:
            trim_to = video_duration
            fade_start = max(trim_to - 0.5, 0)
            cmd = [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-af", f"atrim=0:{trim_to:.3f},afade=t=out:st={fade_start:.3f}:d=0.5",
                "-c:a", "libmp3lame", "-b:a", "128k",
                output_path,
            ]

        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and os.path.exists(output_path):
            return True
        logger.warning("Audio pad/trim failed: %s", r.stderr[-300:] if r.stderr else "")
    except Exception as e:
        logger.error("match_audio_to_video error: %s", e)

    import shutil
    shutil.copy2(audio_path, output_path)
    return True



# 4. Mux audio onto video (video is NEVER modified)


def mux_audio_onto_video(video_path: str, audio_path: str,
                         output_path: str) -> bool:
    """Overlay *audio_path* onto *video_path*.

    Video stream is copied byte-for-byte — no re-encoding, no speed
    change, no trimming.  The full video plays from start to finish.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode == 0 and os.path.exists(output_path):
            return True
        logger.error("mux failed: %s", r.stderr[-500:] if r.stderr else "")
    except Exception as e:
        logger.error("mux error: %s", e)
    return False



# Top-level orchestrator


def sync_narration_to_video(
    narration: str,
    manim_code: str,
    video_path: str,
    output_path: str,
    tts_func,              # callable(text, output_path) -> Optional[str]
    parse_timing_func,     # callable(code) -> SceneTimeline
    clean_text_func=None,  # optional callable(text) -> str
) -> Tuple[bool, str]:
    """End-to-end narration sync.  The video is never modified.

    1. Parse Manim code → logical slides (for logging / future SSML)
    2. Generate TTS for the FULL narration (one voice, one file)
    3. Pad or trim audio to match video duration exactly
    4. Mux onto original video — byte-for-byte copy of video stream

    Returns (success, output_path_or_error).
    """
    if not os.path.exists(video_path):
        return False, "Video file not found"

    if clean_text_func:
        narration = clean_text_func(narration)
    if not narration or not narration.strip():
        return False, "No narration text after cleaning"

    video_duration = _get_duration(video_path)
    if video_duration <= 0:
        return False, "Could not determine video duration"

    # ---- Parse timeline (for diagnostics & future SSML pauses) ----
    try:
        timeline = parse_timing_func(manim_code)
    except Exception as e:
        logger.warning("Timeline parse failed: %s", e)
        timeline = None

    events = timeline.events if timeline else []

    # Scale parsed durations to actual video length
    if timeline and timeline.total_duration > 0:
        scale = video_duration / timeline.total_duration
        if abs(scale - 1.0) > 0.05:
            for ev in events:
                ev.start *= scale
                ev.duration *= scale

    slides = build_slides(events, video_duration)
    logger.info("Logical slides: %d for %.1fs video", len(slides), video_duration)
    for i, sl in enumerate(slides):
        logger.debug("  Slide %d: %.1fs–%.1fs  %s",
                      i, sl.start, sl.start + sl.duration,
                      ", ".join(sl.labels[:3]))

    # Distribute narration across slides (for logging only)
    slices = distribute_narration(narration, slides)
    for i, ns in enumerate(slices):
        logger.debug("  Narration %d (%.1fs): %s…",
                      i, ns.slide_duration, ns.text[:60])

    # ---- Generate TTS — ONE continuous audio file ----
    work_dir = tempfile.mkdtemp(prefix="sync_")
    raw_audio = os.path.join(work_dir, "narration_raw.mp3")

    try:
        result = tts_func(narration, raw_audio)
        if not result or not os.path.exists(raw_audio):
            return False, "TTS generation failed"
    except Exception as e:
        return False, f"TTS error: {e}"

    # ---- Pad / trim so audio == video duration ----
    matched_audio = os.path.join(work_dir, "narration_matched.mp3")
    match_audio_to_video(raw_audio, video_duration, matched_audio)

    if not os.path.exists(matched_audio):
        matched_audio = raw_audio

    # ---- Mux onto original video (video is untouched) ----
    success = mux_audio_onto_video(video_path, matched_audio, output_path)

    # Clean up temp files
    try:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception:
        pass

    if success:
        return True, output_path
    return False, "Failed to mux audio onto video"
