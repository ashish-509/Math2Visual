# Simple spoken math to LaTeX parser
# Converts basic spoken math phrases to LaTeX code.
# Extend the rules below for more math expressions!

import re
from typing import Tuple

def clean_whitespace(s: str) -> str:
    # Replace multiple spaces/tabs/newlines with a single space and trim ends.
    return re.sub(r"\s+", " ", s).strip()

# Main function: spoken math to LaTeX
# Example: "x squared plus y" -> "x^2 + y"
def spoken_math_to_latex(spoken: str) -> str:
    if not spoken:
        return ""
    s = spoken.lower()
    # Replace common math words with LaTeX symbols
    replacements = {
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
        "integral from": "integral_from_marker",  # marker for future extension
        "limit as": "limit_as_marker"
    }
    for word, latex in replacements.items():
        s = s.replace(word, latex)
    # Handle square roots
    s = re.sub(r"\\sqrt\{(.+?)\}", r"\\sqrt{\1}", s)
    # Handle fractions: "a over b" or "a / b" -> \frac{a}{b}
    s = re.sub(r"([0-9a-zA-Z\)\}]+)\s*/\s*([0-9a-zA-Z\(\{]+)", r"\\frac{\1}{\2}", s)
    # Clean up spaces
    return clean_whitespace(s)

# Example usage:
if __name__ == "__main__":
    test = "x squared plus y over z"
    print("Spoken:", test)
    print("LaTeX:", spoken_math_to_latex(test))
