"""
Lightweight pre-render validation for generated Manim code.

Runs a quick dry-run with Manim to catch runtime errors (AttributeError,
TypeError, missing methods, etc.) BEFORE the expensive full render.

This saves minutes of wasted render time on code that was never going to work.
"""

import os
import re
import ast
import textwrap
import tempfile
import subprocess
import logging

logger = logging.getLogger(__name__)


class PreRenderValidator:
    """
    Validates Manim code by running a quick dry-run.
    Catches runtime errors that AST parsing alone can't detect.
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def validate(self, code: str) -> dict:
        """
        Run all validation checks on the code.

        Returns dict with:
            - passed: bool
            - stage: which stage failed (if any)
            - error: error message (if any)
            - error_category: rough category for the error classifier
        """
        # Stage 1: quick structural checks (no subprocess needed)
        result = self._check_structure(code)
        if not result["passed"]:
            return result

        # Stage 2: dry-run with Manim (catches runtime errors)
        result = self._dry_run(code)
        return result

    def _check_structure(self, code: str) -> dict:
        """Check that the code has the basic structure Manim needs."""
        ok = {"passed": True, "stage": "structure", "error": "", "error_category": ""}

        # Must parse as valid Python
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {
                "passed": False,
                "stage": "structure",
                "error": f"Line {e.lineno}: {e.msg}",
                "error_category": "syntax",
            }

        # Must have at least one Scene subclass
        scene_classes = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                base_name = ""
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if "Scene" in base_name:
                    scene_classes.append(node)
                    break

        if not scene_classes:
            return {
                "passed": False,
                "stage": "structure",
                "error": "No class inheriting from Scene found",
                "error_category": "manim_api",
            }

        # Each Scene class must have construct()
        for cls in scene_classes:
            has_construct = any(
                isinstance(item, ast.FunctionDef) and item.name == "construct"
                for item in cls.body
            )
            if not has_construct:
                return {
                    "passed": False,
                    "stage": "structure",
                    "error": f"Class '{cls.name}' is missing a construct() method",
                    "error_category": "manim_api",
                }

        return ok

    def _dry_run(self, code: str) -> dict:
        """Run the code with `manim render --dry_run` to catch runtime errors."""
        ok = {"passed": True, "stage": "dry_run", "error": "", "error_category": ""}

        # Extract class name
        match = re.search(r"class\s+(\w+)\s*\(", code)
        if not match:
            return {
                "passed": False,
                "stage": "dry_run",
                "error": "Could not find class name",
                "error_category": "manim_api",
            }

        class_name = match.group(1)
        tmp_path = None

        try:
            # Write code to a temp file
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, prefix="validate_"
            ) as tmp:
                tmp.write(code)
                tmp_path = tmp.name

            result = subprocess.run(
                ["manim", "render", "--dry_run", "-ql", tmp_path, class_name],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            if result.returncode != 0:
                stderr = result.stderr
                # Trim to last 600 chars to keep it readable
                if len(stderr) > 600:
                    stderr = "..." + stderr[-600:]

                # Guess the error category from stderr
                category = self._guess_category(stderr)

                return {
                    "passed": False,
                    "stage": "dry_run",
                    "error": stderr.strip(),
                    "error_category": category,
                }

            return ok

        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "stage": "dry_run",
                "error": f"Dry run timed out after {self.timeout}s (possible infinite loop)",
                "error_category": "runtime",
            }
        except FileNotFoundError:
            # Manim CLI not installed — skip dry run, let the full render catch it
            logger.warning("manim CLI not found, skipping dry-run validation")
            return ok
        except Exception as e:
            logger.warning(f"Dry run failed unexpectedly: {e}")
            # Don't block the pipeline on validator bugs
            return ok
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def _guess_category(self, stderr: str) -> str:
        """Quick category guess from stderr for the error classifier."""
        text = stderr.lower()
        if "attributeerror" in text:
            return "attribute"
        if "typeerror" in text:
            return "type"
        if "valueerror" in text:
            return "value"
        if "importerror" in text or "modulenotfounderror" in text:
            return "import"
        if "latex" in text or "mathtex" in text:
            return "latex"
        if "nameerror" in text:
            return "name"
        return "runtime"


#  Post-processing: fix text overlap and enforce step-by-step patterns

def fix_text_overlap(code: str) -> str:
    """Post-process generated Manim code to reduce text overlapping.

    Applies several heuristic fixes:
    1. Caps text scale values that are too large
    2. Ensures large-scale cleanup FadeOuts appear between heavy text sections
    3. Reduces font_size values that are above safe thresholds
    """
    if not code or not code.strip():
        return code

    # --- 1. Cap .scale() values on Text objects ---
    # Match Text(...).scale(X) where X > 0.45
    def _cap_text_scale(m):
        prefix = m.group(1)
        val = float(m.group(2))
        if val > 0.45:
            val = 0.4
        return f"{prefix}.scale({val})"

    code = re.sub(
        r'(Text\s*\([^)]*\))\s*\.scale\s*\(\s*([0-9]*\.?[0-9]+)\s*\)',
        _cap_text_scale,
        code,
    )

    # --- 2. Cap font_size in Text() calls ---
    def _cap_font_size(m):
        prefix = m.group(1)
        val = int(m.group(2))
        if val > 32:
            val = 28
        return f"{prefix}font_size={val}"

    code = re.sub(
        r'(Text\s*\([^)]*?)font_size\s*=\s*(\d+)',
        _cap_font_size,
        code,
    )

    # --- 3. Detect stacking patterns and insert FadeOuts ---
    # Look for patterns where multiple Write() calls happen without any FadeOut in between
    lines = code.split('\n')
    fixed_lines = []
    consecutive_writes = 0
    write_vars = []  # track variable names being written
    indent = "        "  # default indent (2 levels)

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect indentation of the current line
        if stripped:
            current_indent = line[:len(line) - len(line.lstrip())]
            if current_indent:
                indent = current_indent

        # Count consecutive Write/FadeIn calls for text
        is_write = bool(re.search(r'self\.play\s*\(\s*(Write|FadeIn)\s*\(', stripped))
        is_fadeout = bool(re.search(r'FadeOut', stripped))
        is_clear_all = bool(re.search(r'FadeOut\s*\(\s*mob\s*\)', stripped))

        if is_fadeout or is_clear_all:
            consecutive_writes = 0
            write_vars = []

        if is_write:
            consecutive_writes += 1
            # Try to extract the variable name being written
            var_match = re.search(r'(Write|FadeIn)\s*\(\s*(\w+)', stripped)
            if var_match:
                write_vars.append(var_match.group(2))
        elif not stripped.startswith('self.wait') and not stripped.startswith('#') and stripped != '':
            # Non-write, non-wait line — could be creating new objects
            pass

        # If we've accumulated 5+ writes without a fadeout, insert a bulk fadeout
        if consecutive_writes >= 5 and is_write:
            # Insert a FadeOut of accumulated objects before this write
            if write_vars and len(write_vars) >= 4:
                # Fade out the older objects (keep the last 2)
                to_fade = write_vars[:-2]
                fade_line = f"{indent}self.play({', '.join(f'FadeOut({v})' for v in to_fade)})"
                fixed_lines.append(fade_line)
                write_vars = write_vars[-2:]
                consecutive_writes = 2

        fixed_lines.append(line)

    code = '\n'.join(fixed_lines)

    # --- 4. Fix .next_to(prev, DOWN) stacking patterns without fadeout ---
    # This pattern causes text to pile up vertically:
    #   step.next_to(prev, DOWN, ...)
    #   self.play(Write(step))
    #   prev = step
    # Convert to slide-based approach where possible
    if re.search(r'\.next_to\s*\(\s*prev\s*,\s*DOWN', code):
        logger.info("Detected stacking pattern (next_to prev DOWN) — consider slide-based approach")
        # We don't auto-rewrite this complex pattern, but we ensure scales are small
        code = re.sub(
            r'(\.scale\s*\(\s*)0\.[4-9]\d*(\s*\))',
            r'\g<1>0.3\2',
            code,
        )

    return code


def fix_step_by_step_overlap(code: str) -> str:
    """Specifically target step-by-step solving patterns that cause overlap.

    Detects the common pattern where steps are placed below each other
    (using .next_to(prev, DOWN)) and rewrites it to a slide-based approach.
    """
    if not code or not code.strip():
        return code

    # Pattern: for loop that stacks steps using next_to(prev, DOWN)
    stacking_pattern = re.compile(
        r'(\s+)prev\s*=\s*None\s*\n'
        r'\s+for\s+\w+,\s*(\w+)\s+in\s+enumerate\((\w+)\):\s*\n'
        r'\s+(\w+)\s*=\s*Text\((\w+)\)\.scale\([^)]+\)\s*\n'
        r'\s+if\s+prev\s+is\s+None:\s*\n'
        r'\s+\4\.move_to\([^)]+\)\s*\n'
        r'\s+else:\s*\n'
        r'\s+\4\.next_to\(prev,\s*DOWN[^)]*\)\s*\n'
        r'\s+self\.play\(Write\(\4\)\)\s*\n'
        r'\s+self\.wait\([^)]+\)\s*\n'
        r'\s+prev\s*=\s*\4',
        re.MULTILINE,
    )

    match = stacking_pattern.search(code)
    if match:
        indent = match.group(1)
        loop_var = match.group(2)
        list_var = match.group(3)
        step_var = match.group(4)
        text_expr = match.group(5)

        replacement = textwrap.dedent(f"""\
{indent}for i, {loop_var} in enumerate({list_var}):
{indent}    header = Text(f"Step {{i+1}}").scale(0.28).to_edge(UP, buff=0.3).set_color(YELLOW)
{indent}    {step_var} = Text({text_expr}).scale(0.3).move_to(ORIGIN)
{indent}    self.play(Write(header), Write({step_var}))
{indent}    self.wait(1.5)
{indent}    self.play(FadeOut(header), FadeOut({step_var}))""")

        code = stacking_pattern.sub(replacement, code)
        logger.info("Rewrote stacking step pattern to slide-based pattern")

    return code
