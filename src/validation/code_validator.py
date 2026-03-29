"""
Lightweight pre-render validation for generated Manim code.

Runs a quick dry-run with Manim to catch runtime errors (AttributeError,
TypeError, missing methods, etc.) BEFORE the expensive full render.

This saves minutes of wasted render time on code that was never going to work.
"""

import os
import re
import ast
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
