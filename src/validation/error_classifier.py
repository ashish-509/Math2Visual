"""
Classifies Manim compilation errors and returns targeted fix instructions.

Instead of dumping a raw traceback at the LLM and hoping it figures out what
to fix, this module categorizes the error and produces specific, actionable
guidance. This dramatically improves the LLM's fix-on-retry success rate.
"""

import re
from enum import Enum
from typing import Tuple


class ErrorType(Enum):
    LATEX = "latex"
    IMPORT = "import"
    ATTRIBUTE = "attribute"
    TYPE = "type"
    VALUE = "value"
    NAME = "name"
    SCENE_LOGIC = "scene_logic"
    SYNTAX = "syntax"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


# Each entry: (ErrorType, list of regex patterns to match in error text)
_PATTERNS = [
    (ErrorType.LATEX, [
        r"LaTeX Error",
        r"Package inputenc Error",
        r"Undefined control sequence",
        r"Missing \$ inserted",
        r"latex.*not found",
        r"FileNotFoundError.*latex",
        r"LaTeX is not installed",
    ]),
    (ErrorType.IMPORT, [
        r"ModuleNotFoundError",
        r"ImportError",
        r"cannot import name",
        r"No module named",
    ]),
    (ErrorType.ATTRIBUTE, [
        r"AttributeError",
        r"has no attribute",
        r"object has no method",
    ]),
    (ErrorType.TYPE, [
        r"TypeError",
        r"expected .* got",
        r"takes \d+ positional argument",
        r"unexpected keyword argument",
    ]),
    (ErrorType.VALUE, [
        r"ValueError",
        r"could not convert",
        r"invalid literal",
        r"math domain error",
    ]),
    (ErrorType.NAME, [
        r"NameError",
        r"name '.*' is not defined",
    ]),
    (ErrorType.SCENE_LOGIC, [
        r"cannot be added to",
        r"not a valid VMobject",
        r"Mobject.*not found",
        r"Animation.*error",
    ]),
    (ErrorType.SYNTAX, [
        r"SyntaxError",
        r"IndentationError",
        r"unexpected EOF",
        r"invalid syntax",
    ]),
    (ErrorType.TIMEOUT, [
        r"timed? ?out",
        r"TimeoutExpired",
    ]),
]

# Targeted fix instructions per error type
_FIX_STRATEGIES = {
    ErrorType.LATEX: (
        "DIAGNOSIS: LaTeX compilation failure.\n"
        "FIX INSTRUCTIONS:\n"
        "- If LaTeX is not installed, replace ALL MathTex() and Tex() with Text().\n"
        "- Write math as plain text: Text('x^2 + 2x + 1').scale(0.4)\n"
        "- If LaTeX IS available, check for: unbalanced braces, missing backslashes,\n"
        "  invalid commands. Use raw strings: MathTex(r'\\frac{a}{b}').\n"
        "- Common mistake: single backslash in non-raw string. Use r'...' prefix."
    ),
    ErrorType.IMPORT: (
        "DIAGNOSIS: Import / module error.\n"
        "FIX INSTRUCTIONS:\n"
        "- Use ONLY 'from manim import *' and do NOT import from manim submodules.\n"
        "- Standard library modules (math, random, numpy) are fine.\n"
        "- Do NOT use third-party packages like scipy, matplotlib, sympy in Manim code.\n"
        "- If using 'import numpy as np', that is fine since numpy ships with Manim."
    ),
    ErrorType.ATTRIBUTE: (
        "DIAGNOSIS: Called a method or attribute that doesn't exist on the object.\n"
        "FIX INSTRUCTIONS:\n"
        "- Check the Manim class API. Common mistakes:\n"
        "  * .set_color() not .color() — use the setter method\n"
        "  * .move_to() not .position() — use move_to or shift\n"
        "  * .get_center() not .center — it's a method, not a property\n"
        "  * Axes has .plot() not .get_graph() (Community Edition)\n"
        "  * Use Create() not ShowCreation() (deprecated)\n"
        "- Make sure you're using Manim Community Edition API, not 3b1b's version."
    ),
    ErrorType.TYPE: (
        "DIAGNOSIS: Wrong argument type or count.\n"
        "FIX INSTRUCTIONS:\n"
        "- Coordinates must be numpy arrays or lists: [x, y, z] not (x, y)\n"
        "- Colors must be Manim constants: RED, BLUE, GREEN — not strings like 'red'\n"
        "- Numbers must be numeric, not strings: scale(0.5) not scale('0.5')\n"
        "- self.play() takes Animation objects: self.play(Create(circle)) not self.play(circle)\n"
        "- Check argument counts — maybe you passed too many or too few args."
    ),
    ErrorType.VALUE: (
        "DIAGNOSIS: Invalid value passed to a function.\n"
        "FIX INSTRUCTIONS:\n"
        "- Check x_range/y_range: must be [min, max, step] with min < max and step > 0\n"
        "- Lambda functions must return valid numbers — avoid division by zero\n"
        "- Scale values must be positive: .scale(0.5) not .scale(-1)\n"
        "- Angle values: use radians (PI, TAU) or convert with degrees from math module."
    ),
    ErrorType.NAME: (
        "DIAGNOSIS: A variable or name is used before being defined.\n"
        "FIX INSTRUCTIONS:\n"
        "- Make sure all variables are defined before use.\n"
        "- Check for typos in variable names.\n"
        "- Manim constants are: UP, DOWN, LEFT, RIGHT, ORIGIN, PI, TAU.\n"
        "- Colors: RED, BLUE, GREEN, YELLOW, WHITE, etc.\n"
        "- If you reference an object in self.play(), make sure it was created earlier."
    ),
    ErrorType.SCENE_LOGIC: (
        "DIAGNOSIS: Manim scene construction error.\n"
        "FIX INSTRUCTIONS:\n"
        "- All Mobjects must be created before being animated.\n"
        "- self.play() needs Animation objects: Create(), FadeIn(), Write(), Transform()\n"
        "- VGroup can only contain VMobjects — don't mix with non-visual objects.\n"
        "- Don't call self.play() outside of construct().\n"
        "- Make sure to self.add() or self.play() objects before transforming them."
    ),
    ErrorType.SYNTAX: (
        "DIAGNOSIS: Python syntax error.\n"
        "FIX INSTRUCTIONS:\n"
        "- Check for: missing colons after class/def/if/for, unmatched brackets,\n"
        "  wrong indentation, unclosed strings.\n"
        "- Every class definition needs a colon: class MyScene(Scene):\n"
        "- Every def needs a colon: def construct(self):\n"
        "- Balance ALL parentheses, brackets, and braces."
    ),
    ErrorType.TIMEOUT: (
        "DIAGNOSIS: Code took too long to execute — likely an infinite loop.\n"
        "FIX INSTRUCTIONS:\n"
        "- Check for infinite while loops — add a break condition.\n"
        "- Reduce the number of objects if creating thousands of Mobjects.\n"
        "- Simplify complex lambda functions that might cause numerical issues.\n"
        "- Limit self.play() to at most ~20 calls in a single scene."
    ),
    ErrorType.UNKNOWN: (
        "DIAGNOSIS: Unknown error type.\n"
        "FIX INSTRUCTIONS:\n"
        "- Read the error message carefully and fix the specific issue.\n"
        "- Make sure the code starts with 'from manim import *'\n"
        "- Make sure there's exactly one class inheriting from Scene with a construct() method.\n"
        "- Keep the code simple — fewer moving parts means fewer bugs."
    ),
}


class ErrorClassifier:
    """Classifies Manim errors and produces targeted fix instructions."""

    @staticmethod
    def classify(error_text: str) -> ErrorType:
        """Determine the error type from a traceback / error message."""
        for error_type, patterns in _PATTERNS:
            if any(re.search(p, error_text, re.IGNORECASE) for p in patterns):
                return error_type
        return ErrorType.UNKNOWN

    @staticmethod
    def get_fix_instructions(error_type: ErrorType) -> str:
        """Get targeted, actionable fix instructions for the given error type."""
        return _FIX_STRATEGIES.get(error_type, _FIX_STRATEGIES[ErrorType.UNKNOWN])

    @staticmethod
    def classify_and_instruct(error_text: str) -> Tuple[ErrorType, str]:
        """Convenience: classify + get instructions in one call."""
        error_type = ErrorClassifier.classify(error_text)
        instructions = ErrorClassifier.get_fix_instructions(error_type)
        return error_type, instructions
