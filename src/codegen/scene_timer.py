"""Extract a rough timeline from Manim scene code.

Parses self.play(), self.wait(), and self.add() calls to estimate when
each visual event occurs. Used to inject timing hints into the teaching-
script prompt so the narration aligns with the animation.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import List


# default durations (seconds) used by Manim community edition
_DEFAULT_PLAY_DURATION = 1.0
_DEFAULT_WAIT_DURATION = 1.0


@dataclass
class SceneEvent:
    start: float
    duration: float
    label: str


@dataclass
class SceneTimeline:
    events: List[SceneEvent] = field(default_factory=list)
    total_duration: float = 0.0

    def as_prompt_block(self) -> str:
        """Format timeline as a human-readable string for LLM injection."""
        if not self.events:
            return ""
        lines = [f"Scene timeline (total ≈ {self.total_duration:.1f}s):"]
        for ev in self.events:
            lines.append(f"  {ev.start:5.1f}s – {ev.label} ({ev.duration:.1f}s)")
        return "\n".join(lines)


def _animation_label(node: ast.Call) -> str:
    """Best-effort human-readable label for a Manim animation call."""
    parts = []
    for arg in node.args:
        src = ast.dump(arg)
        # pull out animation class name if present (e.g. FadeIn, Create)
        m = re.search(r"func=Name\(id='(\w+)'\)", src)
        if m:
            parts.append(m.group(1))
            continue
        m = re.search(r"attr='(\w+)'", src)
        if m:
            parts.append(m.group(1))
            continue
        m = re.search(r"id='(\w+)'", src)
        if m:
            parts.append(m.group(1))
    return " → ".join(parts) if parts else "animation"


def _extract_run_time(node: ast.Call) -> float | None:
    """Return explicit run_time= kwarg if present."""
    for kw in node.keywords:
        if kw.arg == "run_time":
            if isinstance(kw.value, (ast.Constant,)):
                try:
                    return float(kw.value.value)
                except (TypeError, ValueError):
                    pass
    return None


def parse_scene_timing(code: str) -> SceneTimeline:
    """Parse Manim code and return a SceneTimeline."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return SceneTimeline()

    events: List[SceneEvent] = []
    cursor = 0.0  # current time in seconds

    # walk the construct method body
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in ast.walk(node):
            if not isinstance(item, ast.FunctionDef):
                continue
            if item.name != "construct":
                continue
            for stmt in ast.walk(item):
                if not isinstance(stmt, ast.Expr):
                    continue
                call = stmt.value
                if not isinstance(call, ast.Call):
                    continue
                func = call.func
                # match self.play(...) / self.wait(...) / self.add(...)
                if not (isinstance(func, ast.Attribute) and
                        isinstance(func.value, ast.Name) and
                        func.value.id == "self"):
                    continue

                method = func.attr

                if method == "play":
                    dur = _extract_run_time(call) or _DEFAULT_PLAY_DURATION
                    label = _animation_label(call)
                    events.append(SceneEvent(start=cursor, duration=dur, label=label))
                    cursor += dur

                elif method == "wait":
                    dur = _DEFAULT_WAIT_DURATION
                    if call.args:
                        arg0 = call.args[0]
                        if isinstance(arg0, ast.Constant):
                            try:
                                dur = float(arg0.value)
                            except (TypeError, ValueError):
                                pass
                    events.append(SceneEvent(start=cursor, duration=dur, label="pause"))
                    cursor += dur

                elif method == "add":
                    events.append(SceneEvent(start=cursor, duration=0.0, label="add objects"))

    return SceneTimeline(events=events, total_duration=round(cursor, 2))
