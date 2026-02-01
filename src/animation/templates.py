"""
Animation Templates Library

Pre-built Manim templates for common math concepts.
Templates are organized by category: calculus, linear_algebra, statistics, geometry, trigonometry.
"""

import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemplateLibrary:
    """
    Library of pre-built Manim animation templates.
    Each template is a complete, working Manim scene that can be customized.
    """
    
    def __init__(self):
        self.templates = self._load_templates()
        self.categories = list(self.templates.keys())
        logger.info(f"Template library loaded with {len(self.categories)} categories")
    
    def _load_templates(self) -> Dict[str, Dict[str, dict]]:
        """Load all template definitions."""
        
        templates = {
            "calculus": {
                "derivative_visualization": {
                    "name": "Derivative Visualization",
                    "description": "Shows a function and its derivative with tangent line animation",
                    "parameters": ["function", "x_range", "derivative_point"],
                    "code": self._template_derivative()
                },
                "integral_area": {
                    "name": "Integral as Area Under Curve",
                    "description": "Visualizes definite integral as area under a curve with Riemann sums",
                    "parameters": ["function", "x_range", "num_rectangles"],
                    "code": self._template_integral_area()
                },
                "limit_concept": {
                    "name": "Limit Concept",
                    "description": "Demonstrates the concept of limits with epsilon-delta visualization",
                    "parameters": ["function", "limit_point", "limit_value"],
                    "code": self._template_limit()
                },
                "chain_rule": {
                    "name": "Chain Rule",
                    "description": "Visual explanation of the chain rule for derivatives",
                    "parameters": ["outer_function", "inner_function"],
                    "code": self._template_chain_rule()
                }
            },
            
            "linear_algebra": {
                "vector_addition": {
                    "name": "Vector Addition",
                    "description": "Shows vector addition with parallelogram method",
                    "parameters": ["vector_a", "vector_b"],
                    "code": self._template_vector_addition()
                },
                "matrix_transformation": {
                    "name": "Matrix Transformation",
                    "description": "Visualizes how a 2x2 matrix transforms the plane",
                    "parameters": ["matrix"],
                    "code": self._template_matrix_transform()
                },
                "eigenvalue_visualization": {
                    "name": "Eigenvalues and Eigenvectors",
                    "description": "Shows eigenvectors as vectors that only scale under transformation",
                    "parameters": ["matrix"],
                    "code": self._template_eigenvalue()
                },
                "dot_product": {
                    "name": "Dot Product",
                    "description": "Geometric interpretation of dot product as projection",
                    "parameters": ["vector_a", "vector_b"],
                    "code": self._template_dot_product()
                }
            },
            
            "statistics": {
                "normal_distribution": {
                    "name": "Normal Distribution",
                    "description": "Animated bell curve with mean and standard deviation",
                    "parameters": ["mean", "std_dev"],
                    "code": self._template_normal_distribution()
                },
                "central_limit_theorem": {
                    "name": "Central Limit Theorem",
                    "description": "Demonstrates how sample means approach normal distribution",
                    "parameters": ["sample_size", "num_samples"],
                    "code": self._template_central_limit()
                },
                "linear_regression": {
                    "name": "Linear Regression",
                    "description": "Shows best fit line and residuals",
                    "parameters": ["data_points"],
                    "code": self._template_linear_regression()
                },
                "probability_tree": {
                    "name": "Probability Tree",
                    "description": "Animated probability tree diagram",
                    "parameters": ["events", "probabilities"],
                    "code": self._template_probability_tree()
                }
            },
            
            "geometry": {
                "pythagorean_theorem": {
                    "name": "Pythagorean Theorem",
                    "description": "Visual proof of a^2 + b^2 = c^2",
                    "parameters": ["side_a", "side_b"],
                    "code": self._template_pythagorean()
                },
                "circle_area": {
                    "name": "Circle Area Derivation",
                    "description": "Shows how circle area formula is derived",
                    "parameters": ["radius"],
                    "code": self._template_circle_area()
                },
                "similar_triangles": {
                    "name": "Similar Triangles",
                    "description": "Demonstrates properties of similar triangles",
                    "parameters": ["scale_factor"],
                    "code": self._template_similar_triangles()
                }
            },
            
            "trigonometry": {
                "unit_circle": {
                    "name": "Unit Circle",
                    "description": "Animated unit circle with sin and cos values",
                    "parameters": ["angle"],
                    "code": self._template_unit_circle()
                },
                "sin_wave": {
                    "name": "Sine Wave Generation",
                    "description": "Shows how sine wave is generated from circular motion",
                    "parameters": ["amplitude", "frequency"],
                    "code": self._template_sin_wave()
                },
                "trig_identities": {
                    "name": "Trigonometric Identities",
                    "description": "Visual proof of fundamental trig identities",
                    "parameters": ["identity_type"],
                    "code": self._template_trig_identity()
                }
            },
            
            "3d_scenes": {
                "3d_surface_plot": {
                    "name": "3D Surface Plot",
                    "description": "Rotating 3D surface visualization",
                    "parameters": ["function", "x_range", "y_range"],
                    "code": self._template_3d_surface()
                },
                "3d_vector_field": {
                    "name": "3D Vector Field",
                    "description": "3D vector field visualization",
                    "parameters": ["vector_function"],
                    "code": self._template_3d_vectors()
                },
                "3d_coordinate_system": {
                    "name": "3D Coordinate System",
                    "description": "Introduction to 3D coordinates",
                    "parameters": [],
                    "code": self._template_3d_coordinates()
                }
            }
        }
        
        return templates
    
    def get_categories(self) -> List[str]:
        """Get all available template categories."""
        return self.categories
    
    def get_templates_in_category(self, category: str) -> List[dict]:
        """Get all templates in a category."""
        if category not in self.templates:
            return []
        
        result = []
        for template_id, template_data in self.templates[category].items():
            result.append({
                "id": template_id,
                "name": template_data["name"],
                "description": template_data["description"],
                "parameters": template_data["parameters"]
            })
        return result
    
    def get_template(self, category: str, template_id: str) -> Optional[dict]:
        """Get a specific template by category and id."""
        if category not in self.templates:
            return None
        if template_id not in self.templates[category]:
            return None
        return self.templates[category][template_id]
    
    def get_template_code(self, category: str, template_id: str, 
                          color_theme: Optional[dict] = None) -> Optional[str]:
        """Get template code, optionally with color theme applied."""
        template = self.get_template(category, template_id)
        if not template:
            return None
        
        code = template["code"]
        
        if color_theme:
            code = self._apply_color_theme(code, color_theme)
        
        return code
    
    def _apply_color_theme(self, code: str, theme: dict) -> str:
        """Apply a color theme to template code."""
        color_map = {
            "BLUE": theme.get("primary", "BLUE"),
            "RED": theme.get("secondary", "RED"),
            "GREEN": theme.get("accent", "GREEN"),
            "YELLOW": theme.get("highlight", "YELLOW"),
            "WHITE": theme.get("text", "WHITE"),
            "GREY": theme.get("grid", "GREY"),
        }
        
        for old_color, new_color in color_map.items():
            if old_color != new_color:
                code = code.replace(f"color={old_color}", f"color={new_color}")
                code = code.replace(f"fill_color={old_color}", f"fill_color={new_color}")
                code = code.replace(f"stroke_color={old_color}", f"stroke_color={new_color}")
        
        return code
    
    def search_templates(self, query: str) -> List[dict]:
        """Search templates by name or description."""
        query = query.lower()
        results = []
        
        for category, templates in self.templates.items():
            for template_id, template_data in templates.items():
                name = template_data["name"].lower()
                desc = template_data["description"].lower()
                
                if query in name or query in desc:
                    results.append({
                        "category": category,
                        "id": template_id,
                        "name": template_data["name"],
                        "description": template_data["description"]
                    })
        
        return results
    
    # Template code generators
    
    def _template_derivative(self) -> str:
        return '''from manim import *

class DerivativeVisualization(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 10, 2],
            x_length=5,
            y_length=3,
            axis_config={"include_tip": True}
        ).scale(0.8)
        
        # Labels
        x_label = axes.get_x_axis_label("x").scale(0.4)
        y_label = axes.get_y_axis_label("f(x)").scale(0.4)
        
        # Function f(x) = x^2
        func = axes.plot(lambda x: x**2, color=BLUE, x_range=[-2.5, 2.5])
        func_label = MathTex(r"f(x) = x^2").scale(0.4).next_to(axes, UP)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(Create(func), Write(func_label))
        self.wait(1)
        
        # Point where we find derivative
        x_val = 1.5
        point = Dot(axes.c2p(x_val, x_val**2), color=RED)
        point_label = MathTex(f"({x_val}, {x_val**2:.2f})").scale(0.3).next_to(point, UR, buff=0.1)
        
        self.play(Create(point), Write(point_label))
        self.wait(1)
        
        # Tangent line at point
        slope = 2 * x_val
        tangent = axes.plot(
            lambda x: slope * (x - x_val) + x_val**2,
            color=YELLOW,
            x_range=[x_val - 1.5, x_val + 1.5]
        )
        
        slope_text = MathTex(r"f'(x) = 2x").scale(0.4)
        slope_value = MathTex(f"f'({x_val}) = {slope}").scale(0.4)
        slope_group = VGroup(slope_text, slope_value).arrange(DOWN, buff=0.2)
        slope_group.to_corner(UR, buff=0.3)
        
        self.play(Create(tangent))
        self.play(Write(slope_group))
        self.wait(2)
        
        # Animate moving point
        title = Text("The derivative gives the slope\\nof the tangent at any point").scale(0.3)
        title.to_edge(DOWN, buff=0.3)
        self.play(Write(title))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_integral_area(self) -> str:
        return '''from manim import *

class IntegralArea(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 6, 1],
            x_length=5,
            y_length=3,
            axis_config={"include_tip": True}
        ).scale(0.8).shift(DOWN * 0.5)
        
        # Function
        func = axes.plot(lambda x: 0.2 * x**2 + 1, color=BLUE, x_range=[0.1, 4.5])
        func_label = MathTex(r"f(x) = 0.2x^2 + 1").scale(0.35).next_to(axes, UP)
        
        self.play(Create(axes), Write(func_label))
        self.play(Create(func))
        self.wait(1)
        
        # Show Riemann sum with increasing rectangles
        for n_rects in [4, 8, 16]:
            rects = axes.get_riemann_rectangles(
                func,
                x_range=[1, 4],
                dx=3/n_rects,
                color=[BLUE, GREEN],
                fill_opacity=0.5
            )
            
            label = Text(f"{n_rects} rectangles").scale(0.3).to_corner(UR, buff=0.3)
            
            if n_rects == 4:
                self.play(Create(rects), Write(label))
            else:
                self.play(Transform(rects, rects), Transform(label, label))
            self.wait(1)
        
        # Final integral notation
        integral = MathTex(r"\\int_1^4 f(x)\\,dx").scale(0.4)
        integral.to_corner(UL, buff=0.3)
        
        explanation = Text("As rectangles get thinner,\\nsum approaches exact area").scale(0.25)
        explanation.to_edge(DOWN, buff=0.3)
        
        self.play(Write(integral), Write(explanation))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_limit(self) -> str:
        return '''from manim import *

class LimitConcept(Scene):
    def construct(self):
        # Title
        title = Text("Understanding Limits").scale(0.5).to_edge(UP, buff=0.3)
        self.play(Write(title))
        
        # Create axes
        axes = Axes(
            x_range=[-2, 4, 1],
            y_range=[-1, 5, 1],
            x_length=5,
            y_length=3
        ).scale(0.75).shift(DOWN * 0.3)
        
        # Function with a hole at x=2
        func_left = axes.plot(lambda x: x + 1, color=BLUE, x_range=[-1.5, 1.95])
        func_right = axes.plot(lambda x: x + 1, color=BLUE, x_range=[2.05, 3.5])
        hole = Circle(radius=0.08, color=BLUE).move_to(axes.c2p(2, 3))
        
        self.play(Create(axes))
        self.play(Create(func_left), Create(func_right), Create(hole))
        self.wait(1)
        
        # Show limit from left
        left_arrow = Arrow(
            axes.c2p(0.5, 1.5), axes.c2p(1.8, 2.8),
            color=GREEN, buff=0
        ).scale(0.6)
        left_text = MathTex(r"x \\to 2^-").scale(0.35).next_to(left_arrow, DOWN, buff=0.1)
        
        self.play(Create(left_arrow), Write(left_text))
        self.wait(1)
        
        # Show limit from right
        right_arrow = Arrow(
            axes.c2p(3.5, 4.5), axes.c2p(2.2, 3.2),
            color=RED, buff=0
        ).scale(0.6)
        right_text = MathTex(r"x \\to 2^+").scale(0.35).next_to(right_arrow, DOWN, buff=0.1)
        
        self.play(Create(right_arrow), Write(right_text))
        self.wait(1)
        
        # Limit statement
        limit_eq = MathTex(r"\\lim_{x \\to 2} f(x) = 3").scale(0.45)
        limit_eq.to_corner(UR, buff=0.3)
        
        self.play(Write(limit_eq))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_chain_rule(self) -> str:
        return '''from manim import *

class ChainRule(Scene):
    def construct(self):
        # Title
        title = Text("The Chain Rule").scale(0.5).to_edge(UP, buff=0.3)
        self.play(Write(title))
        self.wait(1)
        
        # Composite function
        comp_func = MathTex(r"y = f(g(x))").scale(0.5)
        comp_func.shift(UP * 1.5)
        self.play(Write(comp_func))
        self.wait(1)
        
        # Example
        example = MathTex(r"y = (x^2 + 1)^3").scale(0.5)
        example.next_to(comp_func, DOWN, buff=0.4)
        self.play(Write(example))
        self.wait(1)
        
        # Identify parts
        outer = MathTex(r"\\text{outer: } f(u) = u^3").scale(0.4)
        inner = MathTex(r"\\text{inner: } g(x) = x^2 + 1").scale(0.4)
        parts = VGroup(outer, inner).arrange(DOWN, buff=0.2)
        parts.next_to(example, DOWN, buff=0.5)
        
        self.play(Write(parts))
        self.wait(1)
        
        # Chain rule formula
        self.play(FadeOut(parts))
        
        formula = MathTex(r"\\frac{dy}{dx} = \\frac{df}{du} \\cdot \\frac{du}{dx}").scale(0.5)
        formula.shift(DOWN * 0.5)
        
        self.play(Write(formula))
        self.wait(1)
        
        # Apply to example
        result = MathTex(r"= 3(x^2+1)^2 \\cdot 2x").scale(0.45)
        result.next_to(formula, DOWN, buff=0.3)
        
        self.play(Write(result))
        self.wait(1)
        
        # Final simplified
        final = MathTex(r"= 6x(x^2+1)^2").scale(0.45)
        final.next_to(result, DOWN, buff=0.2)
        
        self.play(Write(final))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_vector_addition(self) -> str:
        return '''from manim import *

class VectorAddition(Scene):
    def construct(self):
        # Title
        title = Text("Vector Addition").scale(0.5).to_edge(UP, buff=0.3)
        self.play(Write(title))
        
        # Create plane
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            x_length=6,
            y_length=4,
            background_line_style={"stroke_opacity": 0.4}
        ).scale(0.7)
        
        self.play(Create(plane))
        
        # Define vectors
        vec_a = Arrow(plane.c2p(0, 0), plane.c2p(2, 1), color=RED, buff=0)
        vec_b = Arrow(plane.c2p(0, 0), plane.c2p(1, 2), color=BLUE, buff=0)
        
        label_a = MathTex(r"\\vec{a}").scale(0.4).next_to(vec_a, RIGHT, buff=0.1)
        label_b = MathTex(r"\\vec{b}").scale(0.4).next_to(vec_b, LEFT, buff=0.1)
        
        self.play(Create(vec_a), Write(label_a))
        self.play(Create(vec_b), Write(label_b))
        self.wait(1)
        
        # Move vector b to tip of a
        vec_b_moved = Arrow(plane.c2p(2, 1), plane.c2p(3, 3), color=BLUE, buff=0)
        
        self.play(Transform(vec_b.copy(), vec_b_moved))
        self.wait(1)
        
        # Show resultant
        vec_sum = Arrow(plane.c2p(0, 0), plane.c2p(3, 3), color=GREEN, buff=0)
        label_sum = MathTex(r"\\vec{a} + \\vec{b}").scale(0.4).next_to(vec_sum, RIGHT, buff=0.1)
        
        self.play(Create(vec_sum), Write(label_sum))
        self.wait(1)
        
        # Formula
        formula = MathTex(r"\\vec{a} + \\vec{b} = (2+1, 1+2) = (3, 3)").scale(0.4)
        formula.to_edge(DOWN, buff=0.3)
        
        self.play(Write(formula))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_matrix_transform(self) -> str:
        return '''from manim import *

class MatrixTransformation(Scene):
    def construct(self):
        # Title
        title = Text("Matrix Transformation").scale(0.45).to_edge(UP, buff=0.3)
        self.play(Write(title))
        
        # Create plane
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=3.5,
            background_line_style={"stroke_opacity": 0.4}
        ).scale(0.7)
        
        self.play(Create(plane))
        
        # Unit square
        square = Square(side_length=1, color=YELLOW, fill_opacity=0.3)
        square.move_to(plane.c2p(0.5, 0.5))
        
        self.play(Create(square))
        self.wait(1)
        
        # Show matrix
        matrix = MathTex(r"A = \\begin{bmatrix} 2 & 1 \\\\ 0 & 1 \\end{bmatrix}").scale(0.4)
        matrix.to_corner(UL, buff=0.3)
        
        self.play(Write(matrix))
        self.wait(1)
        
        # Apply transformation
        transformed = Polygon(
            plane.c2p(0, 0),
            plane.c2p(2, 0),
            plane.c2p(3, 1),
            plane.c2p(1, 1),
            color=GREEN,
            fill_opacity=0.3
        )
        
        explanation = Text("Shear transformation").scale(0.3)
        explanation.to_edge(DOWN, buff=0.3)
        
        self.play(Transform(square, transformed), Write(explanation))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_eigenvalue(self) -> str:
        return '''from manim import *

class EigenvalueVisualization(Scene):
    def construct(self):
        # Title
        title = Text("Eigenvectors: Vectors that\\nonly scale under transformation").scale(0.35)
        title.to_edge(UP, buff=0.3)
        self.play(Write(title))
        
        # Create plane
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=3.5,
            background_line_style={"stroke_opacity": 0.3}
        ).scale(0.7)
        
        self.play(Create(plane))
        
        # Regular vector (changes direction)
        reg_vec = Arrow(plane.c2p(0, 0), plane.c2p(1, 1), color=RED, buff=0)
        reg_label = Text("Regular vector").scale(0.25).next_to(reg_vec, UR, buff=0.1)
        
        self.play(Create(reg_vec), Write(reg_label))
        self.wait(1)
        
        # Eigenvector (only scales)
        eigen_vec = Arrow(plane.c2p(0, 0), plane.c2p(2, 0), color=GREEN, buff=0)
        eigen_label = Text("Eigenvector").scale(0.25).next_to(eigen_vec, DOWN, buff=0.1)
        
        self.play(Create(eigen_vec), Write(eigen_label))
        self.wait(1)
        
        # Show transformation
        self.play(FadeOut(reg_label), FadeOut(eigen_label))
        
        # Transform regular vector (rotates)
        reg_vec_new = Arrow(plane.c2p(0, 0), plane.c2p(2, 0.5), color=RED, buff=0)
        
        # Transform eigenvector (only stretches)
        eigen_vec_new = Arrow(plane.c2p(0, 0), plane.c2p(3, 0), color=GREEN, buff=0)
        
        self.play(
            Transform(reg_vec, reg_vec_new),
            Transform(eigen_vec, eigen_vec_new)
        )
        
        # Eigenvalue equation
        equation = MathTex(r"A\\vec{v} = \\lambda\\vec{v}").scale(0.45)
        equation.to_corner(UR, buff=0.3)
        
        self.play(Write(equation))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_dot_product(self) -> str:
        return '''from manim import *

class DotProduct(Scene):
    def construct(self):
        # Title
        title = Text("Dot Product as Projection").scale(0.45).to_edge(UP, buff=0.3)
        self.play(Write(title))
        
        # Create axes
        plane = NumberPlane(
            x_range=[-2, 5, 1],
            y_range=[-1, 4, 1],
            x_length=5,
            y_length=3.5,
            background_line_style={"stroke_opacity": 0.3}
        ).scale(0.7)
        
        self.play(Create(plane))
        
        # Vector a
        vec_a = Arrow(plane.c2p(0, 0), plane.c2p(4, 0), color=RED, buff=0)
        label_a = MathTex(r"\\vec{a}").scale(0.4).next_to(vec_a, DOWN, buff=0.1)
        
        # Vector b
        vec_b = Arrow(plane.c2p(0, 0), plane.c2p(3, 2), color=BLUE, buff=0)
        label_b = MathTex(r"\\vec{b}").scale(0.4).next_to(vec_b, UP, buff=0.1)
        
        self.play(Create(vec_a), Write(label_a))
        self.play(Create(vec_b), Write(label_b))
        self.wait(1)
        
        # Projection
        proj_point = plane.c2p(3, 0)
        proj_line = DashedLine(plane.c2p(3, 2), proj_point, color=YELLOW)
        proj_vec = Arrow(plane.c2p(0, 0), proj_point, color=GREEN, buff=0)
        
        self.play(Create(proj_line))
        self.play(Create(proj_vec))
        
        proj_label = MathTex(r"\\text{proj}_{\\vec{a}}\\vec{b}").scale(0.35)
        proj_label.next_to(proj_vec, DOWN, buff=0.1)
        self.play(Write(proj_label))
        self.wait(1)
        
        # Formula
        formula = MathTex(r"\\vec{a} \\cdot \\vec{b} = |\\vec{a}||\\vec{b}|\\cos\\theta").scale(0.4)
        formula.to_edge(DOWN, buff=0.3)
        
        self.play(Write(formula))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_normal_distribution(self) -> str:
        return '''from manim import *
import numpy as np

class NormalDistribution(Scene):
    def construct(self):
        # Title
        title = Text("Normal Distribution").scale(0.45).to_edge(UP, buff=0.3)
        self.play(Write(title))
        
        # Parameters
        mu = 0
        sigma = 1
        
        # Create axes
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.5, 0.1],
            x_length=6,
            y_length=3,
            axis_config={"include_tip": True}
        ).scale(0.7).shift(DOWN * 0.3)
        
        x_label = axes.get_x_axis_label("x").scale(0.4)
        y_label = axes.get_y_axis_label("f(x)").scale(0.4)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Normal distribution function
        def normal_pdf(x):
            return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        
        curve = axes.plot(normal_pdf, color=BLUE, x_range=[-3.5, 3.5])
        
        self.play(Create(curve))
        self.wait(1)
        
        # Mean line
        mean_line = DashedLine(
            axes.c2p(mu, 0), axes.c2p(mu, normal_pdf(mu)),
            color=RED
        )
        mean_label = MathTex(r"\\mu = 0").scale(0.35).next_to(mean_line, UP, buff=0.1)
        
        self.play(Create(mean_line), Write(mean_label))
        self.wait(1)
        
        # Standard deviation markers
        std_lines = VGroup(
            DashedLine(axes.c2p(-1, 0), axes.c2p(-1, normal_pdf(-1)), color=GREEN),
            DashedLine(axes.c2p(1, 0), axes.c2p(1, normal_pdf(1)), color=GREEN)
        )
        std_label = MathTex(r"\\sigma = 1").scale(0.35).to_corner(UR, buff=0.3)
        
        self.play(Create(std_lines), Write(std_label))
        self.wait(1)
        
        # 68-95-99.7 rule
        rule_text = Text("68% within 1 std dev").scale(0.25)
        rule_text.to_edge(DOWN, buff=0.3)
        
        area = axes.get_area(curve, x_range=[-1, 1], color=BLUE, opacity=0.3)
        
        self.play(FadeIn(area), Write(rule_text))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_central_limit(self) -> str:
        return '''from manim import *
import numpy as np

class CentralLimitTheorem(Scene):
    def construct(self):
        # Title
        title = Text("Central Limit Theorem").scale(0.45).to_edge(UP, buff=0.3)
        self.play(Write(title))
        self.wait(1)
        
        # Explanation
        explanation = Text("Sample means approach normal\\ndistribution as n increases").scale(0.3)
        explanation.next_to(title, DOWN, buff=0.2)
        self.play(Write(explanation))
        self.wait(1)
        
        # Show progression of distributions
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 0.5, 0.1],
            x_length=5,
            y_length=2.5
        ).scale(0.7).shift(DOWN * 0.5)
        
        self.play(Create(axes))
        
        # Simulate for different sample sizes
        sample_sizes = [1, 5, 30]
        colors = [RED, YELLOW, GREEN]
        
        for i, n in enumerate(sample_sizes):
            n_label = Text(f"n = {n}").scale(0.3)
            n_label.to_corner(UR, buff=0.3)
            
            if n == 1:
                # Uniform-ish distribution
                func = axes.plot(lambda x: 0.1 if 1 < x < 9 else 0, color=colors[i])
            elif n == 5:
                # Starting to look normal
                func = axes.plot(
                    lambda x: 0.3 * np.exp(-0.3 * (x - 5)**2),
                    color=colors[i],
                    x_range=[0.5, 9.5]
                )
            else:
                # Normal distribution
                func = axes.plot(
                    lambda x: 0.4 * np.exp(-0.5 * (x - 5)**2),
                    color=colors[i],
                    x_range=[0.5, 9.5]
                )
            
            self.play(Create(func), Write(n_label))
            self.wait(1)
            self.play(FadeOut(n_label))
        
        conclusion = Text("Larger samples = More normal").scale(0.28)
        conclusion.to_edge(DOWN, buff=0.3)
        self.play(Write(conclusion))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_linear_regression(self) -> str:
        return '''from manim import *

class LinearRegression(Scene):
    def construct(self):
        # Title
        title = Text("Linear Regression").scale(0.45).to_edge(UP, buff=0.3)
        self.play(Write(title))
        
        # Create axes
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 10, 2],
            x_length=5,
            y_length=3.5
        ).scale(0.7).shift(DOWN * 0.3)
        
        x_label = axes.get_x_axis_label("x").scale(0.35)
        y_label = axes.get_y_axis_label("y").scale(0.35)
        
        self.play(Create(axes), Write(x_label), Write(y_label))
        
        # Data points
        data = [(1, 2), (2, 2.5), (3, 3.5), (4, 4), (5, 5.5), (6, 5), (7, 7), (8, 7.5)]
        dots = VGroup(*[
            Dot(axes.c2p(x, y), color=BLUE, radius=0.06)
            for x, y in data
        ])
        
        self.play(Create(dots))
        self.wait(1)
        
        # Best fit line
        line = axes.plot(lambda x: 0.8 * x + 1, color=RED, x_range=[0.5, 8.5])
        line_label = MathTex(r"y = 0.8x + 1").scale(0.35).to_corner(UR, buff=0.3)
        
        self.play(Create(line), Write(line_label))
        self.wait(1)
        
        # Show residuals for one point
        residual = DashedLine(
            axes.c2p(5, 5.5),
            axes.c2p(5, 5),
            color=GREEN
        )
        residual_label = Text("residual").scale(0.25).next_to(residual, RIGHT, buff=0.05)
        
        self.play(Create(residual), Write(residual_label))
        
        # Explanation
        explanation = Text("Minimizes sum of\\nsquared residuals").scale(0.25)
        explanation.to_edge(DOWN, buff=0.3)
        self.play(Write(explanation))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_probability_tree(self) -> str:
        return '''from manim import *

class ProbabilityTree(Scene):
    def construct(self):
        # Title
        title = Text("Probability Tree").scale(0.45).to_edge(UP, buff=0.3)
        self.play(Write(title))
        
        # Root node
        root = Dot(ORIGIN + UP * 1, color=WHITE, radius=0.1)
        root_label = Text("Start").scale(0.25).next_to(root, UP, buff=0.1)
        
        self.play(Create(root), Write(root_label))
        
        # First level branches
        branch1_end = LEFT * 2 + DOWN * 0.5
        branch2_end = RIGHT * 2 + DOWN * 0.5
        
        branch1 = Line(root.get_center(), branch1_end, color=BLUE)
        branch2 = Line(root.get_center(), branch2_end, color=RED)
        
        node1 = Dot(branch1_end, color=BLUE, radius=0.1)
        node2 = Dot(branch2_end, color=RED, radius=0.1)
        
        prob1 = MathTex(r"P(A) = 0.6").scale(0.3).next_to(branch1, UP, buff=0.05)
        prob2 = MathTex(r"P(B) = 0.4").scale(0.3).next_to(branch2, UP, buff=0.05)
        
        self.play(
            Create(branch1), Create(branch2),
            Create(node1), Create(node2),
            Write(prob1), Write(prob2)
        )
        self.wait(1)
        
        # Second level branches from node1
        branch1a_end = LEFT * 3 + DOWN * 2
        branch1b_end = LEFT * 1 + DOWN * 2
        
        branch1a = Line(branch1_end, branch1a_end, color=GREEN)
        branch1b = Line(branch1_end, branch1b_end, color=YELLOW)
        
        node1a = Dot(branch1a_end, color=GREEN, radius=0.1)
        node1b = Dot(branch1b_end, color=YELLOW, radius=0.1)
        
        prob1a = MathTex(r"0.3").scale(0.25).next_to(branch1a, LEFT, buff=0.05)
        prob1b = MathTex(r"0.7").scale(0.25).next_to(branch1b, RIGHT, buff=0.05)
        
        self.play(
            Create(branch1a), Create(branch1b),
            Create(node1a), Create(node1b),
            Write(prob1a), Write(prob1b)
        )
        
        # Combined probability
        combined = MathTex(r"P(A \\cap C) = 0.6 \\times 0.3 = 0.18").scale(0.35)
        combined.to_edge(DOWN, buff=0.3)
        
        self.play(Write(combined))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_pythagorean(self) -> str:
        return '''from manim import *

class PythagoreanTheorem(Scene):
    def construct(self):
        # Title
        title = Text("Pythagorean Theorem").scale(0.45).to_edge(UP, buff=0.3)
        self.play(Write(title))
        
        # Right triangle
        a, b = 2, 1.5
        c = (a**2 + b**2)**0.5
        
        triangle = Polygon(
            ORIGIN, RIGHT * a, RIGHT * a + UP * b,
            color=WHITE
        ).scale(0.9).shift(LEFT * 1.5)
        
        self.play(Create(triangle))
        self.wait(1)
        
        # Labels
        a_label = MathTex("a").scale(0.4).next_to(triangle, DOWN, buff=0.1)
        b_label = MathTex("b").scale(0.4).next_to(triangle, RIGHT, buff=0.1)
        c_label = MathTex("c").scale(0.4).move_to(triangle.get_center() + UP * 0.3 + LEFT * 0.3)
        
        self.play(Write(a_label), Write(b_label), Write(c_label))
        self.wait(1)
        
        # Squares on each side
        sq_a = Square(side_length=a * 0.9, color=RED, fill_opacity=0.3)
        sq_a.next_to(triangle, DOWN, buff=0)
        
        sq_b = Square(side_length=b * 0.9, color=BLUE, fill_opacity=0.3)
        sq_b.next_to(triangle, RIGHT, buff=0)
        
        self.play(Create(sq_a))
        self.play(Create(sq_b))
        self.wait(1)
        
        # Formula
        formula = MathTex(r"a^2 + b^2 = c^2").scale(0.5)
        formula.to_edge(DOWN, buff=0.3)
        
        self.play(Write(formula))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_circle_area(self) -> str:
        return '''from manim import *
import numpy as np

class CircleArea(Scene):
    def construct(self):
        # Title
        title = Text("Deriving Circle Area").scale(0.45).to_edge(UP, buff=0.3)
        self.play(Write(title))
        
        # Circle
        circle = Circle(radius=1.5, color=BLUE, fill_opacity=0.3)
        circle.shift(LEFT * 2)
        
        self.play(Create(circle))
        self.wait(1)
        
        # Radius label
        radius = Line(circle.get_center(), circle.get_right(), color=RED)
        r_label = MathTex("r").scale(0.4).next_to(radius, DOWN, buff=0.05)
        
        self.play(Create(radius), Write(r_label))
        self.wait(1)
        
        # Divide into sectors
        sectors = VGroup()
        n_sectors = 8
        for i in range(n_sectors):
            sector = Sector(
                outer_radius=1.5,
                angle=2 * np.pi / n_sectors,
                start_angle=i * 2 * np.pi / n_sectors,
                color=BLUE if i % 2 == 0 else GREEN,
                fill_opacity=0.5
            )
            sector.shift(LEFT * 2)
            sectors.add(sector)
        
        self.play(FadeOut(circle), FadeIn(sectors))
        self.wait(1)
        
        # Rearrange to approximate rectangle
        explanation = Text("Rearranging sectors...").scale(0.3)
        explanation.to_edge(DOWN, buff=0.5)
        self.play(Write(explanation))
        
        self.play(sectors.animate.shift(RIGHT * 3))
        self.wait(1)
        
        # Formula
        self.play(FadeOut(explanation))
        
        formula1 = MathTex(r"\\text{width} \\approx \\pi r").scale(0.4)
        formula2 = MathTex(r"\\text{height} = r").scale(0.4)
        formula3 = MathTex(r"A = \\pi r \\times r = \\pi r^2").scale(0.45)
        
        formulas = VGroup(formula1, formula2, formula3).arrange(DOWN, buff=0.2)
        formulas.to_edge(DOWN, buff=0.3)
        
        self.play(Write(formula1))
        self.play(Write(formula2))
        self.play(Write(formula3))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_similar_triangles(self) -> str:
        return '''from manim import *

class SimilarTriangles(Scene):
    def construct(self):
        # Title
        title = Text("Similar Triangles").scale(0.45).to_edge(UP, buff=0.3)
        self.play(Write(title))
        
        # Small triangle
        small = Polygon(
            ORIGIN, RIGHT * 1.5, RIGHT * 1.5 + UP * 1,
            color=BLUE, fill_opacity=0.3
        ).shift(LEFT * 3)
        
        # Large triangle (scaled by 2)
        large = Polygon(
            ORIGIN, RIGHT * 3, RIGHT * 3 + UP * 2,
            color=RED, fill_opacity=0.3
        ).shift(RIGHT * 0.5)
        
        self.play(Create(small))
        self.play(Create(large))
        self.wait(1)
        
        # Labels
        small_labels = VGroup(
            MathTex("a").scale(0.35).next_to(small, DOWN, buff=0.05),
            MathTex("b").scale(0.35).next_to(small, RIGHT, buff=0.05)
        )
        
        large_labels = VGroup(
            MathTex("2a").scale(0.35).next_to(large, DOWN, buff=0.05),
            MathTex("2b").scale(0.35).next_to(large, RIGHT, buff=0.05)
        )
        
        self.play(Write(small_labels), Write(large_labels))
        self.wait(1)
        
        # Property
        property_text = MathTex(r"\\frac{a}{2a} = \\frac{b}{2b} = \\frac{1}{2}").scale(0.45)
        property_text.to_edge(DOWN, buff=0.4)
        
        self.play(Write(property_text))
        
        # Explanation
        explanation = Text("Corresponding sides\\nare proportional").scale(0.28)
        explanation.next_to(property_text, UP, buff=0.2)
        
        self.play(Write(explanation))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_unit_circle(self) -> str:
        return '''from manim import *
import numpy as np

class UnitCircle(Scene):
    def construct(self):
        # Title
        title = Text("The Unit Circle").scale(0.45).to_edge(UP, buff=0.3)
        self.play(Write(title))
        
        # Axes
        axes = Axes(
            x_range=[-1.5, 1.5, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True}
        ).scale(0.7)
        
        # Unit circle
        circle = Circle(radius=1 * 0.7, color=BLUE)
        
        self.play(Create(axes), Create(circle))
        
        # Angle and point
        angle = 45 * DEGREES
        point = Dot(
            [0.7 * np.cos(angle), 0.7 * np.sin(angle), 0],
            color=RED
        )
        
        # Radius line
        radius = Line(ORIGIN, point.get_center(), color=YELLOW)
        
        # Arc showing angle
        arc = Arc(radius=0.2, angle=angle, color=GREEN)
        angle_label = MathTex(r"\\theta").scale(0.35).move_to(arc.get_center() + RIGHT * 0.15 + UP * 0.1)
        
        self.play(Create(radius), Create(point), Create(arc), Write(angle_label))
        self.wait(1)
        
        # Projections
        cos_line = DashedLine(point.get_center(), [point.get_x(), 0, 0], color=RED)
        sin_line = DashedLine(point.get_center(), [0, point.get_y(), 0], color=BLUE)
        
        cos_label = MathTex(r"\\cos\\theta").scale(0.3).next_to(cos_line, DOWN, buff=0.05)
        sin_label = MathTex(r"\\sin\\theta").scale(0.3).next_to(sin_line, LEFT, buff=0.05)
        
        self.play(Create(cos_line), Create(sin_line))
        self.play(Write(cos_label), Write(sin_label))
        self.wait(1)
        
        # Coordinates
        coords = MathTex(r"(\\cos\\theta, \\sin\\theta)").scale(0.35)
        coords.next_to(point, UR, buff=0.1)
        
        self.play(Write(coords))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_sin_wave(self) -> str:
        return '''from manim import *
import numpy as np

class SineWaveGeneration(Scene):
    def construct(self):
        # Title
        title = Text("Sine Wave from Circular Motion").scale(0.4).to_edge(UP, buff=0.3)
        self.play(Write(title))
        
        # Circle
        circle = Circle(radius=1, color=BLUE).shift(LEFT * 3)
        self.play(Create(circle))
        
        # Axes for sine wave
        axes = Axes(
            x_range=[0, 2 * np.pi, np.pi / 2],
            y_range=[-1.5, 1.5, 0.5],
            x_length=5,
            y_length=2,
            axis_config={"include_tip": True}
        ).shift(RIGHT * 1.5)
        
        self.play(Create(axes))
        
        # Animate point on circle and sine wave
        dot = Dot(color=RED)
        dot.move_to(circle.get_right())
        
        trace = TracedPath(dot.get_center, stroke_color=RED, stroke_width=2)
        
        # Sine curve
        sine_curve = axes.plot(lambda x: np.sin(x), color=GREEN, x_range=[0, 2 * np.pi])
        
        self.add(trace)
        self.play(Create(dot))
        
        # Connect with line
        connecting_line = always_redraw(
            lambda: DashedLine(
                dot.get_center(),
                axes.c2p(0, dot.get_y() - circle.get_y()),
                color=YELLOW
            )
        )
        self.add(connecting_line)
        
        # Rotate
        self.play(
            Rotate(dot, angle=2 * np.pi, about_point=circle.get_center()),
            Create(sine_curve),
            run_time=4
        )
        
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_trig_identity(self) -> str:
        return '''from manim import *
import numpy as np

class TrigIdentity(Scene):
    def construct(self):
        # Title
        title = Text("Pythagorean Identity").scale(0.45).to_edge(UP, buff=0.3)
        self.play(Write(title))
        
        # Unit circle with right triangle
        circle = Circle(radius=1.5, color=BLUE)
        
        self.play(Create(circle))
        
        # Point on circle
        angle = 40 * DEGREES
        x = 1.5 * np.cos(angle)
        y = 1.5 * np.sin(angle)
        
        point = Dot([x, y, 0], color=RED)
        
        # Triangle
        triangle = Polygon(
            ORIGIN, [x, 0, 0], [x, y, 0],
            color=WHITE
        )
        
        self.play(Create(triangle), Create(point))
        self.wait(1)
        
        # Labels
        cos_label = MathTex(r"\\cos\\theta").scale(0.35).next_to([x/2, 0, 0], DOWN, buff=0.1)
        sin_label = MathTex(r"\\sin\\theta").scale(0.35).next_to([x, y/2, 0], RIGHT, buff=0.1)
        one_label = MathTex("1").scale(0.35).move_to([x/2 - 0.2, y/2 + 0.2, 0])
        
        self.play(Write(cos_label), Write(sin_label), Write(one_label))
        self.wait(1)
        
        # Pythagorean theorem application
        step1 = MathTex(r"(\\cos\\theta)^2 + (\\sin\\theta)^2 = 1^2").scale(0.45)
        step1.shift(DOWN * 2)
        
        self.play(Write(step1))
        self.wait(1)
        
        # Final identity
        identity = MathTex(r"\\cos^2\\theta + \\sin^2\\theta = 1").scale(0.5)
        identity.next_to(step1, DOWN, buff=0.3)
        
        box = SurroundingRectangle(identity, color=YELLOW, buff=0.1)
        
        self.play(Write(identity))
        self.play(Create(box))
        self.wait(2)
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_3d_surface(self) -> str:
        return '''from manim import *
import numpy as np

class Surface3D(ThreeDScene):
    def construct(self):
        # Set camera
        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)
        
        # Create 3D axes
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-2, 2, 1],
            x_length=5,
            y_length=5,
            z_length=3
        )
        
        # Surface z = sin(x) * cos(y)
        surface = Surface(
            lambda u, v: axes.c2p(u, v, np.sin(u) * np.cos(v)),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(30, 30),
            fill_opacity=0.7,
            checkerboard_colors=[BLUE_D, BLUE_E]
        )
        
        # Labels
        title = Text("z = sin(x)cos(y)").scale(0.4)
        title.to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)
        
        self.play(Create(axes))
        self.play(Create(surface))
        
        # Rotate camera
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)
        self.stop_ambient_camera_rotation()
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_3d_vectors(self) -> str:
        return '''from manim import *

class Vectors3D(ThreeDScene):
    def construct(self):
        # Set camera
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        
        # Create 3D axes
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            z_length=5
        )
        
        # Title
        title = Text("3D Vectors").scale(0.4)
        title.to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)
        
        self.play(Create(axes))
        
        # Create vectors
        vec_a = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(2, 1, 1),
            color=RED
        )
        
        vec_b = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(1, 2, 0),
            color=BLUE
        )
        
        vec_c = Arrow3D(
            start=axes.c2p(0, 0, 0),
            end=axes.c2p(0, 1, 2),
            color=GREEN
        )
        
        self.play(Create(vec_a))
        self.play(Create(vec_b))
        self.play(Create(vec_c))
        
        # Rotate to show 3D
        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(4)
        self.stop_ambient_camera_rotation()
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''

    def _template_3d_coordinates(self) -> str:
        return '''from manim import *

class Coordinates3D(ThreeDScene):
    def construct(self):
        # Set camera
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        
        # Create 3D axes
        axes = ThreeDAxes(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            z_range=[-4, 4, 1],
            x_length=6,
            y_length=6,
            z_length=6
        )
        
        # Axis labels
        x_label = axes.get_x_axis_label("x")
        y_label = axes.get_y_axis_label("y")
        z_label = axes.get_z_axis_label("z")
        
        # Title
        title = Text("3D Coordinate System").scale(0.4)
        title.to_corner(UL)
        self.add_fixed_in_frame_mobjects(title)
        
        self.play(Create(axes))
        self.play(Write(x_label), Write(y_label), Write(z_label))
        
        # Point in 3D
        point = Dot3D(axes.c2p(2, 3, 2), color=RED, radius=0.1)
        
        # Dashed lines to show coordinates
        x_line = DashedLine(axes.c2p(0, 0, 0), axes.c2p(2, 0, 0), color=RED)
        y_line = DashedLine(axes.c2p(2, 0, 0), axes.c2p(2, 3, 0), color=GREEN)
        z_line = DashedLine(axes.c2p(2, 3, 0), axes.c2p(2, 3, 2), color=BLUE)
        
        self.play(Create(x_line))
        self.play(Create(y_line))
        self.play(Create(z_line))
        self.play(Create(point))
        
        # Coordinate label
        coord_label = Text("P(2, 3, 2)").scale(0.3)
        coord_label.to_corner(UR)
        self.add_fixed_in_frame_mobjects(coord_label)
        self.play(Write(coord_label))
        
        # Rotate
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(4)
        self.stop_ambient_camera_rotation()
        
        self.play(*[FadeOut(mob) for mob in self.mobjects])
'''


# Singleton instance
_template_library = None


def get_template_library() -> TemplateLibrary:
    global _template_library
    if _template_library is None:
        _template_library = TemplateLibrary()
    return _template_library
