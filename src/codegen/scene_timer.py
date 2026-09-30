"""Extract a timeline from Manim scene code.

Parses self.play(), self.wait(), self.add(), self.clear() and self.remove()
calls in source order to estimate when each visual event occurs, which
variables it animates and which on-screen strings it shows.  Drives the
narration slide plan and the audio sync.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# default durations (seconds) used by Manim community edition
_DEFAULT_PLAY_DURATION = 1.0
_DEFAULT_WAIT_DURATION = 1.0

_TEXT_CLASSES = {"Text", "MathTex", "Tex", "Title", "MarkupText", "Paragraph", "BulletedList"}
# Manim defaults: Write/Unwrite run 2 s when the target has >= 15 glyphs; DrawBorderThenFill always 2 s
_WRITE_LIKE = {"Write", "Unwrite"}
_TWO_SECOND = {"DrawBorderThenFill"}


@dataclass
class SceneEvent:
    start: float
    duration: float
    label: str
    texts: List[str] = field(default_factory=list)    # on-screen strings this event shows
    targets: List[str] = field(default_factory=list)  # variable names it animates


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
    """Human-readable label for a Manim animation call: the animation class (or .animate method)
    of each positional argument, e.g. 'FadeOut → FadeIn'. Handles *[...] comprehensions."""
    parts = []
    for arg in node.args:
        name = None
        for sub in ast.walk(arg):  # breadth-first, so the outermost call wins
            if isinstance(sub, ast.Call):
                name = sub.func.id if isinstance(sub.func, ast.Name) else getattr(sub.func, "attr", None)
                if name:
                    break
        if name is None:
            for sub in ast.walk(arg):
                if isinstance(sub, (ast.Attribute, ast.Name)):
                    name = getattr(sub, "attr", None) or getattr(sub, "id", None)
                    break
        parts.append(name or "animation")
    return " → ".join(parts) if parts else "animation"


def _extract_run_time(node: ast.Call) -> float | None:
    """Return explicit run_time= kwarg if present."""
    for kw in node.keywords:
        if kw.arg == "run_time" and isinstance(kw.value, ast.Constant):
            try:
                return float(kw.value.value)
            except (TypeError, ValueError):
                pass
    return None


def _extract_string_arg(node: ast.expr) -> str | None:
    """Return a plain-text representation of the first string argument."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):  # f-string
        return "".join(str(v.value) if isinstance(v, ast.Constant) else "{...}" for v in node.values)
    return None


def _iter_stmts(stmts):
    """Yield statements in source order, descending into compound statements."""
    for s in stmts:
        yield s
        for attr in ("body", "orelse", "finalbody"):
            inner = getattr(s, attr, None)
            if isinstance(inner, list):
                yield from _iter_stmts(inner)
        for handler in getattr(s, "handlers", None) or []:
            yield from _iter_stmts(handler.body)


def _inline_texts(node: ast.AST) -> List[str]:
    """Strings passed to Text()-like constructors anywhere inside *node*."""
    out: List[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and sub.args:
            name = sub.func.id if isinstance(sub.func, ast.Name) else getattr(sub.func, "attr", None)
            if name in _TEXT_CLASSES:
                s = _extract_string_arg(sub.args[0])
                if s:
                    out.append(s)
    return out


def _referenced(node: ast.AST, assigned: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    """(on-screen texts, assigned variable names) referenced by an expression."""
    texts = _inline_texts(node)
    names: List[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in assigned and sub.id not in names:
            names.append(sub.id)
            texts.extend(assigned[sub.id])
    return list(dict.fromkeys(texts)), names


def _default_play_duration(call: ast.Call, assigned: Dict[str, List[str]]) -> float:
    dur = _DEFAULT_PLAY_DURATION
    for sub in ast.walk(call):
        if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)):
            continue
        if sub.func.id in _TWO_SECOND:
            dur = 2.0
        elif sub.func.id in _WRITE_LIKE:
            texts, _ = _referenced(sub, assigned)
            if sum(len(t.replace(" ", "")) for t in texts) >= 15:
                dur = 2.0
    return dur


def parse_scene_timing(code: str) -> SceneTimeline:
    """Parse Manim code and return a SceneTimeline."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return SceneTimeline()

    events: List[SceneEvent] = []
    cursor = 0.0
    assigned: Dict[str, List[str]] = {}  # variable name -> texts it displays

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "construct":
            continue
        for stmt in _iter_stmts(node.body):
            if isinstance(stmt, ast.Assign):
                texts, _ = _referenced(stmt.value, assigned)
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name):
                        assigned[tgt.id] = texts
                continue
            if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                continue
            call = stmt.value
            func = call.func
            if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                    and func.value.id == "self"):
                continue

            texts, targets = _referenced(call, assigned)
            if func.attr == "play":
                dur = _extract_run_time(call) or _default_play_duration(call, assigned)
                events.append(SceneEvent(cursor, dur, _animation_label(call), texts, targets))
                cursor += dur
            elif func.attr == "wait":
                dur = _DEFAULT_WAIT_DURATION
                if call.args and isinstance(call.args[0], ast.Constant):
                    try:
                        dur = float(call.args[0].value)
                    except (TypeError, ValueError):
                        pass
                events.append(SceneEvent(cursor, dur, "pause"))
                cursor += dur
            elif func.attr == "add":
                events.append(SceneEvent(cursor, 0.0, "add objects", texts, targets))
            elif func.attr in ("clear", "remove"):
                events.append(SceneEvent(cursor, 0.0, "clear screen", [], targets))

    return SceneTimeline(events=events, total_duration=round(cursor, 2))


# ---------------------------------------------------------------------------
# Visual-content extraction — pulls out all text / formula objects so
# the narration prompt knows exactly what appears on screen.
# ---------------------------------------------------------------------------

@dataclass
class VisualElement:
    kind: str          # "Text", "MathTex", "Title", "Tex", "Paragraph", etc.
    content: str       # the string shown on screen
    order: int         # appearance order

    def __str__(self) -> str:
        return f"[{self.kind}] {self.content}"


def extract_visual_content(code: str) -> List[VisualElement]:
    """Parse Manim code and return all visible text/formula objects in order."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    elements: List[VisualElement] = []
    order = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name not in _TEXT_CLASSES:
            continue

        parts = [s for s in (_extract_string_arg(a) for a in node.args) if s]
        if parts:
            order += 1
            elements.append(VisualElement(kind=name, content=" ".join(parts), order=order))

    return elements


def build_content_summary(code: str) -> str:
    """Return a human-readable summary of all visual content in *code*."""
    elements = extract_visual_content(code)
    if not elements:
        return ""

    lines = ["Visual content appearing on screen (in order):"]
    for el in elements:
        lines.append(f"  {el.order}. {el}")
    return "\n".join(lines)
