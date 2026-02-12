"""
Simple RST to Markdown converter. Handles headers, code blocks, links, and common
directives without needing pandoc or any external tools.
"""

import re
import logging

logger = logging.getLogger(__name__)


class RSTtoMarkdownConverter:
    """Converts RST content to Markdown format."""
    
    # RST uses these chars for section underlines (in order of typical precedence)
    SECTION_CHARS = ['=', '-', '~', '^', '"', '#', '*']
    
    def __init__(self):
        self.header_levels = {}
    
    def convert(self, rst):
        """Main conversion method."""
        if not rst:
            return ""
        
        self.header_levels = {}  # reset for each doc
        
        text = rst
        text = self._convert_code_blocks(text)
        text = self._convert_headers(text)
        text = self._convert_inline(text)
        text = self._convert_lists(text)
        text = self._convert_links(text)
        text = self._convert_directives(text)
        text = self._cleanup(text)
        
        return text
    
    def _convert_code_blocks(self, text):
        """Handle code-block directives and literal blocks."""
        
        # .. code-block:: python
        #    code here
        def replace_block(m):
            lang = m.group(1) or ""
            code = self._dedent(m.group(2))
            return f"\n```{lang}\n{code.strip()}\n```\n"
        
        text = re.sub(
            r'\.\.\s+code-block::\s*(\w+)?\s*\n((?:\s+.*\n?)+)',
            replace_block, text, flags=re.MULTILINE
        )
        
        # Literal blocks (:: followed by indented text)
        def replace_literal(m):
            code = self._dedent(m.group(1))
            return f"\n```\n{code.strip()}\n```\n"
        
        text = re.sub(
            r'::\s*\n((?:\s{3,}.*\n?)+)',
            replace_literal, text, flags=re.MULTILINE
        )
        
        return text
    
    def _convert_headers(self, text):
        """Convert RST underline-style headers to # style."""
        lines = text.split('\n')
        result = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check for overline + title + underline style
            if i + 2 < len(lines):
                over = lines[i].strip()
                title = lines[i + 1].strip()
                under = lines[i + 2].strip()
                
                if (len(over) >= 3 and len(set(over)) == 1 and 
                    over[0] in self.SECTION_CHARS and over == under and title):
                    level = self._get_header_level(over[0])
                    result.append('#' * level + ' ' + title)
                    i += 3
                    continue
            
            # Check for title + underline style
            if i + 1 < len(lines):
                under = lines[i + 1].strip()
                
                if (len(under) >= 3 and len(set(under)) == 1 and
                    under[0] in self.SECTION_CHARS and line.strip()):
                    level = self._get_header_level(under[0])
                    result.append('#' * level + ' ' + line.strip())
                    i += 2
                    continue
            
            result.append(line)
            i += 1
        
        return '\n'.join(result)
    
    def _get_header_level(self, char):
        """Assign header levels based on first occurrence of each char."""
        if char not in self.header_levels:
            self.header_levels[char] = len(self.header_levels) + 1
        return min(self.header_levels[char], 6)
    
    def _convert_inline(self, text):
        """Convert inline formatting and roles."""
        # ``code`` -> `code`
        text = re.sub(r'``(.+?)``', r'`\1`', text)
        
        # Various RST roles -> backticks
        # :role:`text` -> `text`
        text = re.sub(r':\w+:`(.+?)`', r'`\1`', text)
        
        # :meth:`foo` and :func:`foo` -> `foo()`
        text = re.sub(r':meth:`(.+?)`', r'`\1()`', text)
        text = re.sub(r':func:`(.+?)`', r'`\1()`', text)
        
        return text
    
    def _convert_lists(self, text):
        """Convert RST list syntax."""
        # #. -> 1.
        text = re.sub(r'^(\s*)#\.\s+', r'\g<1>1. ', text, flags=re.MULTILINE)
        return text
    
    def _convert_links(self, text):
        """Convert RST links to markdown."""
        # `Text <URL>`_ -> [Text](URL)
        text = re.sub(r'`(.+?)\s+<(.+?)>`_', r'[\1](\2)', text)
        
        # `text`__ and `text`_ -> just text
        text = re.sub(r'`(.+?)`__?', r'\1', text)
        
        return text
    
    def _convert_directives(self, text):
        """Convert common directives to markdown equivalents."""
        
        # Note, warning, tip -> blockquotes
        for directive, label in [('note', 'Note'), ('warning', 'Warning'), 
                                  ('tip', 'Tip'), ('important', 'Important')]:
            pattern = rf'\.\.\s+{directive}::\s*\n((?:\s+.*\n?)+)'
            text = re.sub(
                pattern,
                lambda m, l=label: f"\n> **{l}:** {self._dedent(m.group(1)).strip()}\n",
                text, flags=re.MULTILINE
            )
        
        # seealso
        text = re.sub(
            r'\.\.\s+seealso::\s*\n((?:\s+.*\n?)+)',
            lambda m: f"\n**See also:** {self._dedent(m.group(1)).strip()}\n",
            text, flags=re.MULTILINE
        )
        
        # Images
        text = re.sub(r'\.\.\s+image::\s*(.+)', r'![Image](\1)', text)
        text = re.sub(r'\.\.\s+figure::\s*(.+)', r'![Figure](\1)', text)
        
        # Remove toctree and contents directives
        text = re.sub(r'\.\.\s+toctree::.*?(?=\n\S|\Z)', '', text, flags=re.DOTALL)
        text = re.sub(r'\.\.\s+contents::.*?(?=\n\S|\Z)', '', text, flags=re.DOTALL)
        
        return text
    
    def _dedent(self, text):
        """Remove common leading whitespace."""
        lines = text.split('\n')
        
        # Find min indent (ignoring blank lines)
        min_indent = float('inf')
        for line in lines:
            if line.strip():
                min_indent = min(min_indent, len(line) - len(line.lstrip()))
        
        if min_indent == float('inf'):
            return text
        
        return '\n'.join(line[min_indent:] if len(line) >= min_indent else line 
                         for line in lines)
    
    def _cleanup(self, text):
        """Final cleanup pass."""
        # Remove RST comments
        text = re.sub(r'^\.\..*$', '', text, flags=re.MULTILINE)
        
        # Remove stray ::
        text = re.sub(r'^\s*::\s*$', '', text, flags=re.MULTILINE)
        
        # Collapse multiple blank lines
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        
        # Strip trailing whitespace
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        
        return text.strip()


def rst_to_markdown(content):
    """Quick helper function."""
    return RSTtoMarkdownConverter().convert(content)


if __name__ == "__main__":
    # Quick test
    test = '''
Title
=====

Some text with **bold** and *italic*.

Code Example
------------

.. code-block:: python

    from manim import *
    circle = Circle()

.. note::
    Important note here.

See `Manim Docs <https://docs.manim.community/>`_ for more.
'''
    
    print(rst_to_markdown(test))
