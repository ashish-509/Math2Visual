"""Slide-aware audio-video synchronisation.

The video is never modified.  The narration is split into one chunk per
*slide* (a screen of content between FadeOut transitions), each chunk is
synthesised separately, fitted into exactly that slide's time window
(padded with silence, or gently sped up — never cut off — if it runs long)
and the chunks are concatenated, so every sentence starts when the content
it explains appears.  The same slide plan is used to write the script, so
script and audio agree segment by segment.
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

MIN_SLIDE_SECONDS = 2.0   # shorter slides are merged into their neighbour
MAX_TEMPO = 1.35          # max speed-up before we trim with a fade instead
_SAMPLE_RATE = 24000
_SEG_RE = re.compile(r"\[\s*(\d+)\s*\]")


def _get_duration(path: str) -> float:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=noprint_wrappers=1:nokey=1", path],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    return 0.0


def _run(cmd: List[str], timeout: int = 60) -> bool:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return True
        logger.warning("ffmpeg failed: %s", (r.stderr or "")[-300:])
    except Exception as e:
        logger.warning("ffmpeg error: %s", e)
    return False


# 1. Slides

@dataclass
class Slide:
    start: float
    duration: float
    labels: List[str]
    texts: List[str] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)


def _absorb(slide: Slide, ev) -> None:
    slide.labels.append(ev.label)
    slide.texts.extend(t for t in getattr(ev, "texts", []) if t not in slide.texts)
    slide.targets.extend(t for t in getattr(ev, "targets", []) if t not in slide.targets)


def build_slides(scene_events, total_duration: float) -> List[Slide]:
    """Group timeline events into slides. A new slide starts when new content follows a
    FadeOut/clear (or is combined with one in the same play); pauses and additive content
    stay in the current slide."""
    if not scene_events:
        return [Slide(0.0, max(total_duration, 1.0), ["full video"])]

    _CLEAR = ("FadeOut", "Uncreate", "clear screen")
    _CONTENT = ("Write", "FadeIn", "Create", "DrawBorderThenFill", "GrowFromCenter", "GrowArrow",
                "ShowCreation", "Transform", "add objects", "AddTextLetterByLetter")

    slides: List[Slide] = []
    cur = Slide(0.0, 0.0, [])
    saw_clear = False

    def _close_and_open(ev):
        nonlocal cur
        if cur.labels:
            cur.duration = max(ev.start - cur.start, 0.1)
            slides.append(cur)
        cur = Slide(ev.start, 0.0, [])
        _absorb(cur, ev)

    for ev in scene_events:
        label = ev.label.strip()
        is_clear = any(c in label for c in _CLEAR)
        is_content = any(c in label for c in _CONTENT)
        if is_clear and is_content:      # e.g. play(FadeOut(old), FadeIn(new)) — one-step transition
            _close_and_open(ev)
            saw_clear = False
        elif is_clear:
            saw_clear = True
            cur.labels.append(label)
        elif is_content and saw_clear:
            _close_and_open(ev)
            saw_clear = False
        else:
            _absorb(cur, ev)

    if cur.labels or not slides:
        cur.duration = max(total_duration - cur.start, 0.1)
        slides.append(cur)
    return slides


def _merge_short_slides(slides: List[Slide], min_dur: float = MIN_SLIDE_SECONDS) -> List[Slide]:
    merged: List[Slide] = []
    for sl in slides:
        if merged and (sl.duration < min_dur or merged[-1].duration < min_dur):
            prev = merged[-1]
            prev.duration += sl.duration
            prev.labels += sl.labels
            prev.texts += [t for t in sl.texts if t not in prev.texts]
            prev.targets += [t for t in sl.targets if t not in prev.targets]
        else:
            merged.append(Slide(sl.start, sl.duration, list(sl.labels), list(sl.texts), list(sl.targets)))
    return merged


def build_slide_plan(manim_code: str, video_duration: float = 0.0,
                     parse_timing_func: Optional[Callable] = None) -> List[Slide]:
    """Slides for *manim_code*, scaled to the real *video_duration* when known.
    Shared by the script writer and the sync engine so both see identical segments."""
    if parse_timing_func is None:
        from src.codegen.scene_timer import parse_scene_timing as parse_timing_func
    try:
        timeline = parse_timing_func(manim_code or "")
    except Exception as e:
        logger.warning("Timeline parse failed: %s", e)
        timeline = None

    events = list(timeline.events) if timeline else []
    parsed_total = timeline.total_duration if timeline else 0.0
    total = video_duration if video_duration > 0 else (parsed_total or 30.0)

    if events and parsed_total > 0:
        scale = total / parsed_total
        if abs(scale - 1.0) > 0.02:
            for ev in events:
                ev.start *= scale
                ev.duration *= scale

    slides = _merge_short_slides(build_slides(events, total))
    logger.info("Slide plan: %d slides for %.1fs video (parsed %.1fs)", len(slides), total, parsed_total)
    return slides


# 2. Narration segments

def split_segments(narration: str) -> Optional[List[str]]:
    """Split '[1] ... [2] ...' narration into per-segment texts (None if unmarked)."""
    if not narration:
        return None
    parts = _SEG_RE.split(narration)
    if len(parts) < 3:
        return None
    segments = {}
    for i in range(1, len(parts) - 1, 2):
        idx = int(parts[i])
        segments[idx] = (segments.get(idx, "") + " " + parts[i + 1]).strip()
    ordered = [segments[k] for k in sorted(segments)]
    lead = parts[0].strip()
    if lead and ordered:
        ordered[0] = (lead + " " + ordered[0]).strip()
    return ordered


def distribute_narration(narration: str, slides: List[Slide]) -> List[str]:
    """Fallback for unmarked text: split at sentence ends, proportional to slide durations."""
    if len(slides) <= 1:
        return [narration.strip()]
    sentence_ends = [m.end() for m in re.finditer(r"[.!?]\s", narration + " ")] or [len(narration)]
    total = sum(s.duration for s in slides) or 1.0
    out, cursor = [], 0
    for idx, slide in enumerate(slides):
        if idx == len(slides) - 1:
            out.append(narration[cursor:].strip())
            break
        target = cursor + int(slide.duration / total * len(narration))
        candidates = [p for p in sentence_ends if p > cursor] or [len(narration)]
        best = min(candidates, key=lambda p: abs(p - target))
        out.append(narration[cursor:best].strip())
        cursor = best
    return out


def segments_for_slides(narration: str, slides: List[Slide],
                        clean_func: Optional[Callable[[str], str]] = None) -> List[str]:
    """One narration chunk per slide. Uses [n] markers when they match the plan (extra trailing
    chunks fold into the last slide); otherwise splits the marker-free text proportionally."""
    clean = clean_func or (lambda s: s)
    segs = split_segments(narration)
    if segs and len(segs) > len(slides):
        segs = segs[:len(slides) - 1] + [" ".join(segs[len(slides) - 1:])]
    if segs and len(segs) == len(slides):
        return [clean(s) for s in segs]
    flat = clean(" ".join(segs) if segs else _SEG_RE.sub(" ", narration))
    return distribute_narration(flat, slides)


# 3. Audio fitting

def _silence(duration: float, out_path: str) -> bool:
    return _run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r={_SAMPLE_RATE}:cl=mono",
                 "-t", f"{max(duration, 0.05):.3f}", "-c:a", "pcm_s16le", out_path])


def fit_audio_to_slot(audio_path: str, slot: float, out_path: str) -> Tuple[bool, float]:
    """Make *audio_path* exactly *slot* seconds: pad with silence, or speed up (<= MAX_TEMPO)
    and only as a last resort trim with a fade. Returns (ok, tempo applied)."""
    audio_dur = _get_duration(audio_path)
    if audio_dur <= 0:
        return _silence(slot, out_path), 1.0
    tempo = audio_dur / slot
    applied = min(tempo, MAX_TEMPO) if tempo > 1.02 else 1.0
    # drop the TTS engine's leading silence so the first word lands on the slide boundary
    filters = ["silenceremove=start_periods=1:start_silence=0.05:start_threshold=-45dB"]
    filters += [f"atempo={applied:.4f}"] if applied > 1.0 else []
    filters += ["apad", f"atrim=end={slot:.3f}"]
    if tempo > MAX_TEMPO:
        filters.append(f"afade=t=out:st={max(slot - 0.4, 0):.3f}:d=0.4")
    filters.append("asetpts=N/SR/TB")
    ok = _run(["ffmpeg", "-y", "-i", audio_path, "-af", ",".join(filters),
               "-ar", str(_SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", out_path])
    return ok, applied


def _concat(paths: List[str], total: float, out_path: str) -> bool:
    list_file = out_path + ".txt"
    with open(list_file, "w") as f:
        for p in paths:
            f.write(f"file '{p}'\n")
    return _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
                 "-af", f"apad,atrim=end={total:.3f}", "-c:a", "pcm_s16le", out_path], timeout=120)


def mux_audio_onto_video(video_path: str, audio_path: str, output_path: str) -> bool:
    """Copy the video stream untouched and attach the narration track.
    MP3 rather than AAC: Chromium/Firefox builds without proprietary codecs (VS Code, Fedora) play AAC silently."""
    return _run(["ffmpeg", "-y", "-i", video_path, "-i", audio_path,
                 "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                 "-c:a", "libmp3lame", "-ar", "44100", "-ac", "2", "-b:a", "128k",
                 "-movflags", "+faststart", output_path], timeout=120)


# 4. Orchestrator

def sync_narration_to_video(
    narration: str,
    manim_code: str,
    video_path: str,
    output_path: str,
    tts_func,              # callable(text, output_path) -> Optional[str]
    parse_timing_func,     # callable(code) -> SceneTimeline
    clean_text_func=None,  # optional callable(text) -> str, applied per segment
) -> Tuple[bool, str]:
    if not os.path.exists(video_path):
        return False, "Video file not found"
    video_duration = _get_duration(video_path)
    if video_duration <= 0:
        return False, "Could not determine video duration"

    slides = build_slide_plan(manim_code, video_duration, parse_timing_func)
    segments = segments_for_slides(narration or "", slides, clean_text_func)
    if not any(s.strip() for s in segments):
        return False, "No narration text after cleaning"

    work_dir = tempfile.mkdtemp(prefix="sync_")

    def _render(i: int) -> Optional[str]:
        slide, text = slides[i], segments[i]
        fitted = os.path.join(work_dir, f"seg_{i:02d}.wav")
        if not text.strip():
            return fitted if _silence(slide.duration, fitted) else None
        raw = os.path.join(work_dir, f"seg_{i:02d}.mp3")
        try:
            if not tts_func(text, raw) or not os.path.exists(raw):
                raise RuntimeError("TTS returned nothing")
        except Exception as e:
            logger.warning("TTS failed for slide %d (%s); using silence", i + 1, e)
            return fitted if _silence(slide.duration, fitted) else None
        ok, tempo = fit_audio_to_slot(raw, slide.duration, fitted)
        logger.info("Slide %d: slot %.1fs, speech %.1fs, tempo %.2f, %d words",
                    i + 1, slide.duration, _get_duration(raw), tempo, len(text.split()))
        return fitted if ok else None

    try:
        with ThreadPoolExecutor(max_workers=min(4, len(slides))) as pool:
            fitted = list(pool.map(_render, range(len(slides))))
        if any(f is None for f in fitted):
            return False, "Audio fitting failed"
        track = os.path.join(work_dir, "narration.wav")
        if not _concat(fitted, video_duration, track):
            return False, "Audio concat failed"
        if not mux_audio_onto_video(video_path, track, output_path):
            return False, "Failed to mux audio onto video"
        return True, output_path
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
