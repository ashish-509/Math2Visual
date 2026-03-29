import re

# manim-specific terms that help retrieval find the right docs
_MANIM_BOOSTERS = {
    "plot": ["Axes", "plot", "graph", "NumberPlane"],
    "graph": ["Axes", "plot", "NumberPlane"],
    "circle": ["Circle", "Arc", "Dot", "geometry"],
    "triangle": ["Triangle", "Polygon", "geometry"],
    "square": ["Square", "Rectangle", "geometry"],
    "matrix": ["Matrix", "IntegerMatrix", "linear algebra"],
    "vector": ["Arrow", "Vector", "NumberPlane"],
    "derivative": ["tangent_line", "Axes", "plot", "secant"],
    "integral": ["get_area", "Axes", "area under"],
    "limit": ["Axes", "plot", "ValueTracker"],
    "transform": ["Transform", "ReplacementTransform", "TransformMatchingTex"],
    "animate": ["animate", "self.play", "Animation"],
    "text": ["Text", "MathTex", "Tex", "Write"],
    "color": ["set_color", "color", "RED", "BLUE"],
    "3d": ["ThreeDScene", "Surface", "ThreeDAxes"],
    "bar": ["BarChart", "chart"],
    "number line": ["NumberLine", "include_numbers"],
    "table": ["Table", "MobjectTable"],
    "angle": ["Angle", "Arc", "unit circle"],
}


def enhance_query(prompt):
    """Add manim-specific keywords to a RAG query for better retrieval."""
    lower = prompt.lower()
    extras = set()

    for trigger, boost_terms in _MANIM_BOOSTERS.items():
        if trigger in lower:
            extras.update(boost_terms)

    if not extras:
        return prompt

    suffix = " " + " ".join(extras)
    return prompt + suffix
