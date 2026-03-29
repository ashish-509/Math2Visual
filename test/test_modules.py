"""Unit tests for P0–P2 modules.

Run:  python -m pytest test/ -v
"""

import ast
import json
import os
import sys
import tempfile
import textwrap

import pytest

# ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# P0: PreRenderValidator 

from src.validation.code_validator import PreRenderValidator


class TestPreRenderValidator:
    validator = PreRenderValidator(timeout=10)

    def test_valid_simple_scene(self):
        code = textwrap.dedent("""\
            from manim import *
            class Demo(Scene):
                def construct(self):
                    self.play(Create(Circle()))
        """)
        result = self.validator._check_structure(code)
        assert result["passed"]

    def test_missing_scene_class(self):
        code = textwrap.dedent("""\
            from manim import *
            def hello():
                pass
        """)
        result = self.validator._check_structure(code)
        assert not result["passed"]
        assert "Scene" in result["error"]

    def test_missing_construct(self):
        code = textwrap.dedent("""\
            from manim import *
            class Demo(Scene):
                def run(self):
                    pass
        """)
        result = self.validator._check_structure(code)
        assert not result["passed"]
        assert "construct" in result["error"]

    def test_syntax_error(self):
        code = "def foo(:"
        result = self.validator._check_structure(code)
        assert not result["passed"]
        assert result["stage"] == "structure"


# P0: ErrorClassifier 

from src.validation.error_classifier import ErrorClassifier, ErrorType


class TestErrorClassifier:
    def test_latex_error(self):
        err = "FileNotFoundError: latex is not installed"
        etype, hint = ErrorClassifier.classify_and_instruct(err)
        assert etype == ErrorType.LATEX

    def test_import_error(self):
        err = "ImportError: cannot import name 'FooBar' from 'manim'"
        etype, hint = ErrorClassifier.classify_and_instruct(err)
        assert etype == ErrorType.IMPORT
        assert len(hint) > 0

    def test_attribute_error(self):
        err = "AttributeError: 'Circle' object has no attribute 'set_colour'"
        etype, _ = ErrorClassifier.classify_and_instruct(err)
        assert etype == ErrorType.ATTRIBUTE

    def test_unknown_error(self):
        etype, _ = ErrorClassifier.classify_and_instruct("something strange happened")
        assert etype == ErrorType.UNKNOWN


# P1: RenderCache

from src.rendering import RenderCache


class TestRenderCache:
    def test_get_miss(self, tmp_path):
        cache = RenderCache(cache_dir=str(tmp_path))
        assert cache.get("code_here", "medium") is None

    def test_put_and_get(self, tmp_path):
        cache = RenderCache(cache_dir=str(tmp_path))
        # create a fake video file
        fake_video = tmp_path / "vid.mp4"
        fake_video.write_bytes(b"\x00" * 100)

        cache.put("my_code", "low", str(fake_video))
        hit = cache.get("my_code", "low")
        assert hit is not None
        assert os.path.exists(hit)

    def test_clear(self, tmp_path):
        cache = RenderCache(cache_dir=str(tmp_path))
        fake = tmp_path / "v.mp4"
        fake.write_bytes(b"\x00")
        cache.put("c", "m", str(fake))
        cache.clear()
        assert cache.get("c", "m") is None

    def test_stats(self, tmp_path):
        cache = RenderCache(cache_dir=str(tmp_path))
        s = cache.stats()
        assert "total" in s


# P1: Template matching 

from src.codegen import match_template, get_template_hint, list_templates


class TestTemplates:
    def test_match_function_plot(self):
        name, tmpl = match_template("plot a quadratic function graph")
        assert name is not None
        assert name == "function_plot"

    def test_match_derivative(self):
        name, tmpl = match_template("compute the derivative tangent line")
        assert name is not None
        assert name == "derivative_visualization"

    def test_no_match_garbage(self):
        name, tmpl = match_template("asdfghjklzxcvbnm")
        assert name is None

    def test_hint_string(self):
        hint = get_template_hint("plot a sine wave function")
        assert hint is not None
        assert "skeleton" in hint.lower() or "template" in hint.lower() or len(hint) > 20

    def test_list_all(self):
        all_t = list_templates()
        assert len(all_t) >= 10


# P1: Query enhancer

from src.rag.query_enhancer import enhance_query


class TestQueryEnhancer:
    def test_circle_boost(self):
        enhanced = enhance_query("draw a circle")
        assert "Circle" in enhanced

    def test_graph_boost(self):
        enhanced = enhance_query("plot a function graph")
        assert "Axes" in enhanced or "plot" in enhanced

    def test_no_boost_generic(self):
        enhanced = enhance_query("hello world")
        assert enhanced == "hello world"


# P2: SceneTimer 

from src.codegen.scene_timer import parse_scene_timing


class TestSceneTimer:
    SAMPLE = textwrap.dedent("""\
        from manim import *
        class Demo(Scene):
            def construct(self):
                self.play(Write(Text("Hi")))
                self.wait(2)
                self.play(FadeIn(Circle()), run_time=1.5)
    """)

    def test_total_duration(self):
        tl = parse_scene_timing(self.SAMPLE)
        assert tl.total_duration == pytest.approx(4.5, abs=0.01)

    def test_event_count(self):
        tl = parse_scene_timing(self.SAMPLE)
        assert len(tl.events) == 3

    def test_prompt_block(self):
        tl = parse_scene_timing(self.SAMPLE)
        block = tl.as_prompt_block()
        assert "timeline" in block.lower()

    def test_empty_code(self):
        tl = parse_scene_timing("")
        assert tl.total_duration == 0.0

    def test_syntax_error_code(self):
        tl = parse_scene_timing("def foo(:")
        assert tl.total_duration == 0.0


# P2: EdgeTTS engine 

from src.tts.edge_tts_engine import compute_pause_points, _build_ssml


class TestEdgeTTS:
    def test_pause_points_basic(self):
        text = "First sentence. Second sentence. Third one. Fourth one."
        pts = compute_pause_points(text)
        # should insert pauses at some sentence boundaries
        assert isinstance(pts, list)

    def test_ssml_no_pauses(self):
        ssml = _build_ssml("Hello world", pauses=None)
        assert "<speak" in ssml
        assert "Hello world" in ssml

    def test_ssml_with_pauses(self):
        ssml = _build_ssml("Hello world. Goodbye.", pauses=[(12, 500)])
        assert "<break" in ssml

    def test_escaping(self):
        ssml = _build_ssml("a < b & c > d")
        assert "&lt;" in ssml
        assert "&amp;" in ssml
        assert "&gt;" in ssml


# P3: Config 
from src.config import (
    PROJECT_ROOT, QUALITY_FLAGS, QUALITY_LABELS,
    DEFAULT_QUALITY, MANIM_RENDER_TIMEOUT, SPEAKING_RATE_WPS,
)


class TestConfig:
    def test_project_root_exists(self):
        assert os.path.isdir(PROJECT_ROOT)

    def test_quality_flags(self):
        assert set(QUALITY_FLAGS.keys()) == {"low", "medium", "high"}

    def test_quality_labels(self):
        for k in QUALITY_FLAGS:
            assert k in QUALITY_LABELS

    def test_defaults_sensible(self):
        assert DEFAULT_QUALITY == "medium"
        assert MANIM_RENDER_TIMEOUT >= 60
        assert SPEAKING_RATE_WPS > 0


# P3: Logging utils 

from src.logging_utils import new_request_id, log_timing
import logging


class TestLoggingUtils:
    def test_request_id_unique(self):
        ids = {new_request_id() for _ in range(100)}
        assert len(ids) == 100

    def test_request_id_length(self):
        assert len(new_request_id()) == 10

    def test_log_timing(self, caplog):
        logger = logging.getLogger("test_timer")
        with caplog.at_level(logging.INFO):
            with log_timing(logger, "test_op"):
                pass
        assert any("test_op" in r.message for r in caplog.records)
