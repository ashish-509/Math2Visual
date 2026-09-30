"""End-to-end check: code -> video -> per-slide script -> synced audio, then measure alignment."""
import re
import subprocess
import sys
import time

import requests

sys.path.insert(0, ".")
from src.tts.segment_sync import build_slide_plan  # noqa: E402

B = "http://localhost:8000"
prompt = sys.argv[1] if len(sys.argv) > 1 else \
    "Show a circle transforming into a square, then display the formula for the area of a circle"

t = time.time()
r = requests.post(f"{B}/generate_code", json={"prompt": prompt, "model_choice": "CodeLlama-34B"}, timeout=180).json()
code = r["code"]
print(f"[1] code       {time.time()-t:5.1f}s ok={r['success']} {len(code)} chars, starts: {code[:18]!r}")

t = time.time()
v = requests.post(f"{B}/compile_video", json={"code": code, "quality": "low"}, timeout=400).json()
print(f"[2] video      {time.time()-t:5.1f}s ok={v['success']} {v.get('video_path')} {v.get('message', '')[:120]}")
vp = v["video_path"]
dur = requests.post(f"{B}/get_video_duration", json={"video_path": vp}).json()["duration"]

t = time.time()
s = requests.post(f"{B}/generate_teaching_script", json={
    "animation_description": prompt, "manim_code": code,
    "model_choice": "CodeLlama-34B", "video_duration": dur}, timeout=180).json()
print(f"[3] script     {time.time()-t:5.1f}s ok={s['success']} {s.get('message', '')}")
print("    " + s["script"].replace("\n", "\n    "))
bad = re.findall(r"[<>^*=+/\\{}$#`_]|speak|xmlns|break time", s["script"])
print(f"    symbols/markup in script: {bad or 'none'}")

t = time.time()
m = requests.post(f"{B}/merge_video_audio", json={"video_path": vp, "audio_text": s["script"], "manim_code": code}, timeout=400).json()
print(f"[4] merge      {time.time()-t:5.1f}s ok={m['success']} {m.get('final_video_path')} {m.get('message', '')}")
fp = m["final_video_path"]

# ---- sync measurement ----
plan = build_slide_plan(code, dur)
probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format=duration",
                        "-of", "csv=p=0", fp], capture_output=True, text=True).stdout.split()
print(f"\n    video {dur:.2f}s | final file streams/duration: {probe}")
det = subprocess.run(["ffmpeg", "-i", fp, "-af", "silencedetect=noise=-35dB:d=0.3", "-f", "null", "-"],
                     capture_output=True, text=True).stderr
starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", det)]
ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", det)]
onsets = ([0.0] if not starts or starts[0] > 0.15 else []) + ends
print(f"    speech onsets (s): {[round(o, 2) for o in onsets]}")
worst = 0.0
for i, sl in enumerate(plan, 1):
    near = min(onsets, key=lambda o: abs(o - sl.start))
    worst = max(worst, abs(near - sl.start))
    print(f"    slide {i}: window {sl.start:5.1f}-{sl.start+sl.duration:5.1f}s  speech starts {near:5.2f}s  "
          f"offset {near-sl.start:+.2f}s  on-screen: {sl.texts[:2]}")
print(f"\n    WORST OFFSET {worst:.2f}s  ->  {'IN SYNC' if worst <= 0.5 else 'OUT OF SYNC'}")
