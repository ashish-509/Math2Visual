"""Turn on-screen math notation into words a TTS engine reads naturally.

Applied to the text shown in the animation (so the narration prompt never
contains raw symbols) and to the final narration right before synthesis.
"""

import re

_LATEX_WORDS = {
    "pi": "pi", "theta": "theta", "alpha": "alpha", "beta": "beta", "gamma": "gamma",
    "delta": "delta", "lambda": "lambda", "mu": "mu", "sigma": "sigma", "omega": "omega",
    "epsilon": "epsilon", "phi": "phi", "rho": "rho", "tau": "tau", "infty": " infinity ",
    "cdot": " times ", "times": " times ", "div": " divided by ", "pm": " plus or minus ",
    "to": " approaches ", "rightarrow": " approaches ", "approx": " is approximately ",
    "le": " is less than or equal to ", "leq": " is less than or equal to ",
    "ge": " is greater than or equal to ", "geq": " is greater than or equal to ",
    "ne": " is not equal to ", "neq": " is not equal to ", "int": " the integral of ",
    "sum": " the sum of ", "lim": " lim ",
}

_SYMBOLS = {
    "π": " pi ", "θ": " theta ", "α": " alpha ", "β": " beta ", "γ": " gamma ", "δ": " delta ",
    "Δ": " delta ", "λ": " lambda ", "μ": " mu ", "σ": " sigma ", "ω": " omega ", "ε": " epsilon ",
    "φ": " phi ", "ρ": " rho ", "τ": " tau ", "∞": " infinity ", "∑": " the sum of ", "Σ": " the sum of ",
    "∫": " the integral of ", "√": " the square root of ", "≈": " is approximately ",
    "≠": " is not equal to ", "≤": " is less than or equal to ", "≥": " is greater than or equal to ",
    "→": " approaches ", "±": " plus or minus ", "×": " times ", "·": " times ", "÷": " divided by ",
    "°": " degrees ", "∈": " is in ", "∂": " partial ", "²": "^2", "³": "^3", "⁴": "^4", "⁵": "^5",
    "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9", "⁰": "^0", "¹": "^1", "ⁿ": "^n", "−": "-", "–": "-",
}

_FUNC_WORDS = {
    "sin": "sine", "cos": "cosine", "tan": "tangent", "cot": "cotangent", "sec": "secant",
    "csc": "cosecant", "arcsin": "arc sine", "arccos": "arc cosine", "arctan": "arc tangent",
    "sinh": "hyperbolic sine", "cosh": "hyperbolic cosine", "tanh": "hyperbolic tangent",
    "ln": "the natural log", "log": "log", "exp": "e to the power",
}


def speakable_math(text: str) -> str:
    """Convert formulas/symbols in *text* to spoken words; ordinary prose passes through."""
    if not text:
        return ""
    s = text

    # LaTeX
    s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1) over (\2)", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"the square root of (\1)", s)
    s = re.sub(r"\\(?:text|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:left|right|quad|,|;|!)", " ", s)
    s = re.sub(r"\\([a-zA-Z]+)", lambda m: _LATEX_WORDS.get(m.group(1), m.group(1)), s)
    s = re.sub(r"[{}$]", " ", s)

    # Unicode symbols / superscripts
    s = "".join(_SYMBOLS.get(ch, ch) for ch in s)

    # Structural notation
    s = re.sub(r"(?<=[\w)])\*\*(?=[\w(])", "^", s)  # python power, not markdown
    s = re.sub(r"\|([^|]{1,40})\|", r"the absolute value of \1", s)
    s = re.sub(r"\bsqrt\s*\(([^()]*)\)", r"the square root of (\1)", s)
    s = re.sub(r"\bsqrt\s*(\w+)", r"the square root of \1", s)
    s = re.sub(r"\b(arcsin|arccos|arctan|sinh|cosh|tanh|sin|cos|tan|cot|sec|csc|ln|log|exp)\s*\(([^()]*)\)",
               lambda m: f"{_FUNC_WORDS.get(m.group(1), m.group(1))} of {m.group(2)}", s)
    s = re.sub(r"\b(sin|cos|tan)\b", lambda m: _FUNC_WORDS[m.group(1)], s)
    s = re.sub(r"\blim\b", "the limit as", s)
    s = re.sub(r"\b([fghpuvyFGHPQ])\(([^()]{1,40})\)", r"\1 of \2", s)  # f(x) -> f of x
    s = re.sub(r"\bd([a-zA-Z])\s*/\s*d([a-zA-Z])\b", r"d\1 by d\2", s)  # dy/dx
    s = re.sub(r"\^\s*2\b", " squared", s)
    s = re.sub(r"\^\s*3\b", " cubed", s)
    s = re.sub(r"\^\s*\(([^()]*)\)", r" to the power of \1", s)
    s = re.sub(r"\^\s*(-?\w+(?:\.\d+)?)", r" to the power of \1", s)
    s = re.sub(r"(\w)_\{?(\w+)\}?", r"\1 sub \2", s)
    s = re.sub(r"\b(\d+|[a-zA-Z])!(?=\s*[=+*/)\-])", r"\1 factorial", s)  # 5! = 120, not "Wow!"
    s = re.sub(r"(\d)\s*%", r"\1 percent", s)

    # Operators
    s = re.sub(r"<=", " is less than or equal to ", s)
    s = re.sub(r">=", " is greater than or equal to ", s)
    s = re.sub(r"!=", " is not equal to ", s)
    s = re.sub(r"->", " approaches ", s)
    s = re.sub(r"==?", " equals ", s)
    s = re.sub(r"<", " is less than ", s)
    s = re.sub(r">", " is greater than ", s)
    s = re.sub(r"\+", " plus ", s)
    s = re.sub(r"\*", " times ", s)
    s = re.sub(r"(?<=[\w)])\s*/\s*(?=[\w(])", " over ", s)
    s = re.sub(r"\s-\s", " minus ", s)
    s = re.sub(r"(?<=[\w)])-(?=\d)", " minus ", s)
    s = re.sub(r"(?<=\d)-(?=[a-zA-Z(])", " minus ", s)
    s = re.sub(r"(?<![\w)])-(?=\d)", "negative ", s)

    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()
