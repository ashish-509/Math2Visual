import re
from typing import Tuple

def clean_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

# A conservative rule-based parser: 
# TODO : EXTEND
def spoken_math_to_latex(spoken: str) -> str:
    if not spoken:
        return ""
    s = spoken.lower()
    # basic replacements
    reps = {
        "plus": "+",
        "minus": "-",
        "times": "\\times ",
        "multiplied by": "\\times ",
        "divided by": "/",
        "over": "/",
        "squared": "^2",
        "cubed": "^3",
        "equals": "=",
        "is equal to": "=",
        "square root of": "\\sqrt{",
        "integral from": "integral_from_marker",  # marker used for patterning below
        "limit as": "limit_as_marker"
    }
    for k, v in reps.items():
        s = s.replace(k, v)
    # post-process sqrt: naive approach
    s = re.sub(r"\\sqrt\{(.+?)\}", r"\\sqrt{\1}", s)
    # simple fraction pattern: "a over b" -> \frac{a}{b}
    s = re.sub(r"([0-9a-zA-Z\)\}]+)\s*/\s*([0-9a-zA-Z\(\{]+)", r"\\frac{\1}{\2}", s)
    return clean_whitespace(s)
