"""
Color Theme Manager

Provides pre-defined color themes for Manim animations.
Users can select a theme and the colors will be applied to generated code.
"""

import logging
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColorThemeManager:
    """
    Manages color themes for Manim animations.
    Each theme defines colors for different element types.
    """
    
    def __init__(self):
        self.themes = self._load_themes()
        self.theme_names = list(self.themes.keys())
        logger.info(f"Color theme manager loaded with {len(self.theme_names)} themes")
    
    def _load_themes(self) -> Dict[str, dict]:
        """Load all predefined color themes."""
        
        themes = {
            "default": {
                "name": "Default",
                "description": "Standard Manim colors - clean and professional",
                "colors": {
                    "primary": "BLUE",
                    "secondary": "RED",
                    "accent": "GREEN",
                    "highlight": "YELLOW",
                    "text": "WHITE",
                    "background": "BLACK",
                    "grid": "GREY",
                    "positive": "GREEN",
                    "negative": "RED",
                    "neutral": "GREY"
                },
                "preview_colors": ["#58C4DD", "#FC6255", "#83C167", "#FFFF00"]
            },
            
            "3blue1brown": {
                "name": "3Blue1Brown",
                "description": "Inspired by 3Blue1Brown videos - warm and educational",
                "colors": {
                    "primary": "BLUE_C",
                    "secondary": "GOLD",
                    "accent": "MAROON_B",
                    "highlight": "YELLOW",
                    "text": "WHITE",
                    "background": "BLACK",
                    "grid": "GREY_B",
                    "positive": "GREEN_C",
                    "negative": "RED_C",
                    "neutral": "GREY_B"
                },
                "preview_colors": ["#58C4DD", "#C9B354", "#C55F73", "#FFFF00"]
            },
            
            "pastel": {
                "name": "Pastel",
                "description": "Soft pastel colors - gentle on the eyes",
                "colors": {
                    "primary": "TEAL_C",
                    "secondary": "PINK",
                    "accent": "LIGHT_PINK",
                    "highlight": "PURE_GREEN",
                    "text": "WHITE",
                    "background": "BLACK",
                    "grid": "GREY_A",
                    "positive": "TEAL_B",
                    "negative": "LIGHT_PINK",
                    "neutral": "GREY_A"
                },
                "preview_colors": ["#5CD0B3", "#D147BD", "#DC75CD", "#00FF00"]
            },
            
            "neon": {
                "name": "Neon",
                "description": "Vibrant neon colors - high contrast and eye-catching",
                "colors": {
                    "primary": "PURE_BLUE",
                    "secondary": "PURE_RED",
                    "accent": "PURE_GREEN",
                    "highlight": "YELLOW_A",
                    "text": "WHITE",
                    "background": "BLACK",
                    "grid": "GREY_D",
                    "positive": "PURE_GREEN",
                    "negative": "PURE_RED",
                    "neutral": "GREY_C"
                },
                "preview_colors": ["#0000FF", "#FF0000", "#00FF00", "#FFFF33"]
            },
            
            "ocean": {
                "name": "Ocean",
                "description": "Cool ocean-inspired colors - calming blues and teals",
                "colors": {
                    "primary": "BLUE_D",
                    "secondary": "TEAL_D",
                    "accent": "BLUE_A",
                    "highlight": "GREEN_A",
                    "text": "WHITE",
                    "background": "BLACK",
                    "grid": "BLUE_E",
                    "positive": "TEAL_C",
                    "negative": "BLUE_B",
                    "neutral": "BLUE_E"
                },
                "preview_colors": ["#236B8E", "#55C1A7", "#C7E9F1", "#C9E2AE"]
            },
            
            "sunset": {
                "name": "Sunset",
                "description": "Warm sunset colors - oranges, reds, and purples",
                "colors": {
                    "primary": "ORANGE",
                    "secondary": "RED_C",
                    "accent": "PURPLE_B",
                    "highlight": "GOLD",
                    "text": "WHITE",
                    "background": "BLACK",
                    "grid": "RED_E",
                    "positive": "GOLD",
                    "negative": "RED_D",
                    "neutral": "MAROON_E"
                },
                "preview_colors": ["#FF862F", "#FC6255", "#9A72AC", "#C9B354"]
            },
            
            "forest": {
                "name": "Forest",
                "description": "Natural forest colors - greens and earthy tones",
                "colors": {
                    "primary": "GREEN_D",
                    "secondary": "GREEN_A",
                    "accent": "GOLD_A",
                    "highlight": "YELLOW_D",
                    "text": "WHITE",
                    "background": "BLACK",
                    "grid": "GREEN_E",
                    "positive": "GREEN_C",
                    "negative": "RED_B",
                    "neutral": "GREEN_E"
                },
                "preview_colors": ["#77B05D", "#C9E2AE", "#F0D770", "#B0B000"]
            },
            
            "monochrome": {
                "name": "Monochrome",
                "description": "Black and white - minimal and clean",
                "colors": {
                    "primary": "WHITE",
                    "secondary": "GREY_B",
                    "accent": "GREY_A",
                    "highlight": "WHITE",
                    "text": "WHITE",
                    "background": "BLACK",
                    "grid": "GREY_D",
                    "positive": "WHITE",
                    "negative": "GREY_C",
                    "neutral": "GREY_B"
                },
                "preview_colors": ["#FFFFFF", "#AAAAAA", "#DDDDDD", "#FFFFFF"]
            },
            
            "candy": {
                "name": "Candy",
                "description": "Sweet candy colors - fun and playful",
                "colors": {
                    "primary": "PURPLE_A",
                    "secondary": "PINK",
                    "accent": "BLUE_A",
                    "highlight": "GOLD_A",
                    "text": "WHITE",
                    "background": "BLACK",
                    "grid": "PURPLE_E",
                    "positive": "TEAL_A",
                    "negative": "RED_A",
                    "neutral": "PURPLE_D"
                },
                "preview_colors": ["#CAA3E8", "#D147BD", "#C7E9F1", "#F0D770"]
            },
            
            "fire": {
                "name": "Fire",
                "description": "Hot fire colors - reds, oranges, and yellows",
                "colors": {
                    "primary": "RED_B",
                    "secondary": "ORANGE",
                    "accent": "YELLOW_C",
                    "highlight": "YELLOW_A",
                    "text": "WHITE",
                    "background": "BLACK",
                    "grid": "RED_E",
                    "positive": "YELLOW_B",
                    "negative": "RED_D",
                    "neutral": "MAROON_D"
                },
                "preview_colors": ["#FF8080", "#FF862F", "#FFEA94", "#FFFF33"]
            }
        }
        
        return themes
    
    def get_theme_names(self) -> List[str]:
        """Get list of all available theme names."""
        return self.theme_names
    
    def get_themes_list(self) -> List[dict]:
        """Get list of all themes with metadata for UI display."""
        result = []
        for theme_id, theme_data in self.themes.items():
            result.append({
                "id": theme_id,
                "name": theme_data["name"],
                "description": theme_data["description"],
                "preview_colors": theme_data["preview_colors"]
            })
        return result
    
    def get_theme(self, theme_id: str) -> Optional[dict]:
        """Get a specific theme by ID."""
        return self.themes.get(theme_id)
    
    def get_theme_colors(self, theme_id: str) -> Optional[dict]:
        """Get just the colors dict for a theme."""
        theme = self.get_theme(theme_id)
        if theme:
            return theme["colors"]
        return None
    
    def apply_theme_to_code(self, code: str, theme_id: str) -> str:
        """
        Apply a color theme to Manim code.
        Replaces default color references with theme colors.
        """
        theme = self.get_theme(theme_id)
        if not theme:
            return code
        
        colors = theme["colors"]
        
        # Map of standard colors to theme colors
        replacements = {
            "BLUE": colors.get("primary", "BLUE"),
            "RED": colors.get("secondary", "RED"),
            "GREEN": colors.get("accent", "GREEN"),
            "YELLOW": colors.get("highlight", "YELLOW"),
            "WHITE": colors.get("text", "WHITE"),
        }
        
        result = code
        
        for original, replacement in replacements.items():
            if original != replacement:
                # Replace color=COLOR patterns
                result = result.replace(f"color={original}", f"color={replacement}")
                result = result.replace(f"color={original},", f"color={replacement},")
                result = result.replace(f"color={original})", f"color={replacement})")
                
                # Replace fill_color patterns
                result = result.replace(f"fill_color={original}", f"fill_color={replacement}")
                
                # Replace stroke_color patterns
                result = result.replace(f"stroke_color={original}", f"stroke_color={replacement}")
        
        return result
    
    def get_color_code_snippet(self, theme_id: str) -> str:
        """
        Generate a code snippet that defines theme colors as variables.
        Can be added to the start of generated code.
        """
        theme = self.get_theme(theme_id)
        if not theme:
            return ""
        
        colors = theme["colors"]
        snippet = f'''# Color Theme: {theme["name"]}
PRIMARY_COLOR = {colors["primary"]}
SECONDARY_COLOR = {colors["secondary"]}
ACCENT_COLOR = {colors["accent"]}
HIGHLIGHT_COLOR = {colors["highlight"]}
TEXT_COLOR = {colors["text"]}

'''
        return snippet


# Singleton instance
_color_theme_manager = None


def get_color_theme_manager() -> ColorThemeManager:
    global _color_theme_manager
    if _color_theme_manager is None:
        _color_theme_manager = ColorThemeManager()
    return _color_theme_manager
