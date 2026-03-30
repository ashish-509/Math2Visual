import re
import logging

logger = logging.getLogger(__name__)


# each template: name, category, keywords, skeleton code
TEMPLATES = {

    "function_plot": {
        "category": "calculus",
        "keywords": ["plot", "graph", "function", "curve", "parabola", "sine", "cosine", "polynomial"],
        "description": "Plot a mathematical function on axes",
        "skeleton": '''from manim import *
import numpy as np

class {class_name}(Scene):
    def construct(self):
        axes = Axes(
            x_range={x_range}, y_range={y_range},
            x_length=6, y_length=4,
            axis_config={{"include_numbers": True}}
        ).scale(0.7)
        labels = axes.get_axis_labels(x_label=Text("{x_label}").scale(0.3), y_label=Text("{y_label}").scale(0.3))

        graph = axes.plot({function}, color={color})
        graph_label = Text("{formula_label}").scale(0.25).next_to(graph, UR, buff=0.1)

        self.play(Create(axes), Write(labels))
        self.play(Create(graph), Write(graph_label), run_time=2)
        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
''',
    },

    "step_by_step_equation": {
        "category": "algebra",
        "keywords": ["solve", "equation", "step", "simplify", "factor", "quadratic", "linear"],
        "description": "Solve an equation step by step",
        "skeleton": '''from manim import *

class {class_name}(Scene):
    def construct(self):
        # --- Problem Statement Slide ---
        title = Text("{title}").scale(0.4).to_edge(UP, buff=0.3)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))

        steps = [{steps_list}]

        # --- Show each step as its own slide ---
        for i, step_text in enumerate(steps):
            header = Text(f"Step {{i+1}}").scale(0.28).to_edge(UP, buff=0.3).set_color(YELLOW)
            step = Text(step_text).scale(0.3).move_to(ORIGIN)
            self.play(Write(header), Write(step))
            self.wait(1.5)
            # Clear screen before next step
            self.play(FadeOut(header), FadeOut(step))

        # --- Final Answer Slide ---
        result = Text("{result}").scale(0.4).set_color(GREEN).move_to(ORIGIN)
        self.play(Write(result))
        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
''',
    },

    "geometric_shapes": {
        "category": "geometry",
        "keywords": ["circle", "triangle", "square", "rectangle", "polygon", "shape", "area", "perimeter"],
        "description": "Show and label geometric shapes",
        "skeleton": '''from manim import *

class {class_name}(Scene):
    def construct(self):
        title = Text("{title}").scale(0.4).to_edge(UP, buff=0.4)
        self.play(Write(title))
        self.wait(0.5)
        self.play(FadeOut(title))

        shape = {shape_code}
        shape.set_color({color})
        label = Text("{label}").scale(0.3).next_to(shape, DOWN, buff=0.3)

        self.play(Create(shape))
        self.play(Write(label))
        self.wait(1)

        {extra_animations}

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
''',
    },

    "coordinate_geometry": {
        "category": "geometry",
        "keywords": ["coordinate", "point", "distance", "midpoint", "slope", "line segment"],
        "description": "Plot points and lines on a coordinate plane",
        "skeleton": '''from manim import *
import numpy as np

class {class_name}(Scene):
    def construct(self):
        plane = NumberPlane(
            x_range={x_range}, y_range={y_range},
            x_length=6, y_length=5,
            background_line_style={{"stroke_opacity": 0.3}}
        ).scale(0.7)
        self.play(Create(plane))

        {points_and_lines}

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
''',
    },

    "derivative_visualization": {
        "category": "calculus",
        "keywords": ["derivative", "tangent", "slope", "differentiation", "rate of change"],
        "description": "Show a function and its tangent line / derivative",
        "skeleton": '''from manim import *
import numpy as np

class {class_name}(Scene):
    def construct(self):
        axes = Axes(
            x_range={x_range}, y_range={y_range},
            x_length=6, y_length=4,
            axis_config={{"include_numbers": True}}
        ).scale(0.7)
        self.play(Create(axes))

        func = axes.plot({function}, color=BLUE)
        func_label = Text("{func_label}").scale(0.25).next_to(func, UR, buff=0.1)
        self.play(Create(func), Write(func_label))
        self.wait(1)

        # tangent at a point
        x_val = {x_val}
        dot = Dot(axes.c2p(x_val, {function}(x_val)), color=YELLOW)
        tangent = axes.get_secant_slope_group(x=x_val, graph=func, dx=0.01, secant_line_color=RED, secant_line_length=3)
        self.play(Create(dot), Create(tangent))

        slope_text = Text("Slope at x={x_val_str}").scale(0.25).to_edge(DOWN)
        self.play(Write(slope_text))

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
''',
    },

    "integral_area": {
        "category": "calculus",
        "keywords": ["integral", "area", "under curve", "definite integral", "integration", "shade"],
        "description": "Show the area under a curve",
        "skeleton": '''from manim import *
import numpy as np

class {class_name}(Scene):
    def construct(self):
        axes = Axes(
            x_range={x_range}, y_range={y_range},
            x_length=6, y_length=4,
            axis_config={{"include_numbers": True}}
        ).scale(0.7)
        self.play(Create(axes))

        func = axes.plot({function}, color=BLUE)
        self.play(Create(func))
        self.wait(0.5)

        area = axes.get_area(func, x_range=[{area_start}, {area_end}], color=GREEN, opacity=0.4)
        area_label = Text("{area_label}").scale(0.25).next_to(area, DOWN, buff=0.3)
        self.play(FadeIn(area), Write(area_label))

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
''',
    },

    "matrix_display": {
        "category": "linear_algebra",
        "keywords": ["matrix", "determinant", "vector", "transformation", "eigen", "linear algebra"],
        "description": "Display and animate matrices",
        "skeleton": '''from manim import *

class {class_name}(Scene):
    def construct(self):
        title = Text("{title}").scale(0.4).to_edge(UP, buff=0.4)
        self.play(Write(title))

        {matrix_code}

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
''',
    },

    "number_line_visual": {
        "category": "algebra",
        "keywords": ["number line", "inequality", "interval", "range", "absolute value"],
        "description": "Visualize values on a number line",
        "skeleton": '''from manim import *

class {class_name}(Scene):
    def construct(self):
        title = Text("{title}").scale(0.4).to_edge(UP, buff=0.4)
        self.play(Write(title))
        self.wait(0.5)

        nline = NumberLine(x_range={x_range}, length=8, include_numbers=True).scale(0.8)
        self.play(Create(nline))

        {markers}

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
''',
    },

    "trig_unit_circle": {
        "category": "trigonometry",
        "keywords": ["unit circle", "sin", "cos", "trig", "trigonometry", "angle", "radian"],
        "description": "Show the unit circle with sin/cos",
        "skeleton": '''from manim import *
import numpy as np

class {class_name}(Scene):
    def construct(self):
        circle = Circle(radius=2, color=WHITE)
        axes_h = Line(LEFT * 2.5, RIGHT * 2.5, stroke_width=1, color=GREY)
        axes_v = Line(DOWN * 2.5, UP * 2.5, stroke_width=1, color=GREY)
        self.play(Create(circle), Create(axes_h), Create(axes_v))

        angle = {angle}
        end_point = circle.point_at_angle(angle)
        radius_line = Line(ORIGIN, end_point, color=YELLOW)
        dot = Dot(end_point, color=YELLOW)

        cos_line = Line(ORIGIN, [end_point[0], 0, 0], color=BLUE)
        sin_line = Line([end_point[0], 0, 0], end_point, color=RED)

        cos_label = Text("cos").scale(0.22).next_to(cos_line, DOWN, buff=0.1)
        sin_label = Text("sin").scale(0.22).next_to(sin_line, RIGHT, buff=0.1)

        self.play(Create(radius_line), Create(dot))
        self.play(Create(cos_line), Create(sin_line))
        self.play(Write(cos_label), Write(sin_label))

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
''',
    },

    "comparison_side_by_side": {
        "category": "general",
        "keywords": ["compare", "comparison", "vs", "difference", "versus", "side by side"],
        "description": "Compare two concepts side by side",
        "skeleton": '''from manim import *

class {class_name}(Scene):
    def construct(self):
        title = Text("{title}").scale(0.4).to_edge(UP, buff=0.4)
        self.play(Write(title))
        self.wait(0.5)

        left_group = VGroup(
            Text("{left_title}").scale(0.3),
            {left_content}
        ).arrange(DOWN, buff=0.3).move_to(LEFT * 3)

        right_group = VGroup(
            Text("{right_title}").scale(0.3),
            {right_content}
        ).arrange(DOWN, buff=0.3).move_to(RIGHT * 3)

        divider = Line(UP * 2, DOWN * 2, color=GREY, stroke_width=1)

        self.play(Create(divider))
        self.play(FadeIn(left_group), FadeIn(right_group))

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
''',
    },

    "definition_card": {
        "category": "general",
        "keywords": ["define", "definition", "what is", "explain", "concept", "theorem", "formula"],
        "description": "Show a concept definition with visual",
        "skeleton": '''from manim import *

class {class_name}(Scene):
    def construct(self):
        title = Text("{title}").scale(0.4).to_edge(UP, buff=0.3)
        self.play(Write(title))
        self.wait(0.5)
        self.play(FadeOut(title))

        definition = Text("{definition}").scale(0.3).move_to(UP * 0.5)
        self.play(Write(definition))
        self.wait(1.5)
        self.play(FadeOut(definition))

        formula = Text("{formula}").scale(0.35).move_to(UP * 0.5)
        self.play(Write(formula))
        self.wait(1)

        {visual_code}

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
''',
    },

    "sequence_animation": {
        "category": "algebra",
        "keywords": ["sequence", "series", "sum", "pattern", "progression", "arithmetic", "geometric"],
        "description": "Animate a mathematical sequence or series",
        "skeleton": '''from manim import *

class {class_name}(Scene):
    def construct(self):
        title = Text("{title}").scale(0.4).to_edge(UP, buff=0.4)
        self.play(Write(title))
        self.wait(0.5)

        terms = [{terms_list}]
        term_mobs = VGroup(*[Text(str(t)).scale(0.35) for t in terms])
        term_mobs.arrange(RIGHT, buff=0.4).move_to(ORIGIN)

        for mob in term_mobs:
            self.play(FadeIn(mob, shift=UP * 0.3), run_time=0.5)
        self.wait(1)

        {extra_code}

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
''',
    },

    "vector_visualization": {
        "category": "linear_algebra",
        "keywords": ["vector", "arrow", "magnitude", "direction", "addition", "component"],
        "description": "Visualize vectors on a plane",
        "skeleton": '''from manim import *
import numpy as np

class {class_name}(Scene):
    def construct(self):
        plane = NumberPlane(
            x_range=[-5, 5, 1], y_range=[-4, 4, 1],
            x_length=7, y_length=5,
            background_line_style={{"stroke_opacity": 0.3}}
        ).scale(0.65)
        self.play(Create(plane))

        {vectors_code}

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
''',
    },

    "probability_visual": {
        "category": "statistics",
        "keywords": ["probability", "dice", "coin", "chance", "event", "outcome", "bar chart", "bar graph"],
        "description": "Visualize probability concepts",
        "skeleton": '''from manim import *

class {class_name}(Scene):
    def construct(self):
        title = Text("{title}").scale(0.4).to_edge(UP, buff=0.4)
        self.play(Write(title))
        self.wait(0.5)
        self.play(FadeOut(title))

        {visual_code}

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
''',
    },

    "limits_visualization": {
        "category": "calculus",
        "keywords": ["limit", "approach", "tends to", "infinity", "epsilon", "delta"],
        "description": "Visualize limits of a function",
        "skeleton": '''from manim import *
import numpy as np

class {class_name}(Scene):
    def construct(self):
        axes = Axes(
            x_range={x_range}, y_range={y_range},
            x_length=6, y_length=4,
            axis_config={{"include_numbers": True}}
        ).scale(0.7)
        self.play(Create(axes))

        func = axes.plot({function}, color=BLUE, discontinuities={discontinuities}, dt=0.01)
        self.play(Create(func))
        self.wait(0.5)

        # approaching dot
        approach_x = {approach_x}
        dot = Dot(color=YELLOW).move_to(axes.c2p(approach_x - 1, {function}(approach_x - 1)))
        self.play(Create(dot))

        # animate approach
        self.play(dot.animate.move_to(axes.c2p(approach_x - 0.1, {function}(approach_x - 0.1))), run_time=2)

        limit_text = Text("{limit_label}").scale(0.25).to_edge(DOWN)
        self.play(Write(limit_text))

        self.wait(2)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
''',
    },
}


def match_template(prompt):
    """Find the best matching template for a user prompt. Returns (name, template) or (None, None)."""
    prompt_lower = prompt.lower()
    best_name = None
    best_score = 0

    for name, tmpl in TEMPLATES.items():
        score = sum(1 for kw in tmpl["keywords"] if kw in prompt_lower)
        if score > best_score:
            best_score = score
            best_name = name

    # need at least 1 keyword match
    if best_score >= 1 and best_name:
        return best_name, TEMPLATES[best_name]
    return None, None


def get_template_hint(prompt):
    """Build an LLM hint showing the matched template skeleton. Returns empty string if no match."""
    name, tmpl = match_template(prompt)
    if not tmpl:
        return ""

    hint = (
        f"\n=== TEMPLATE HINT ({name}) ===\n"
        f"A pre-validated template is available for this type of animation.\n"
        f"You SHOULD follow this structure closely and fill in the placeholders.\n"
        f"Template category: {tmpl['category']}\n\n"
        f"{tmpl['skeleton']}\n"
        f"=== END TEMPLATE ===\n"
    )
    logger.info(f"Template matched: {name} (category={tmpl['category']})")
    return hint


def list_templates():
    """Return a summary of available templates."""
    return {name: {"category": t["category"], "description": t["description"]} for name, t in TEMPLATES.items()}
