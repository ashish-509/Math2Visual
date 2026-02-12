"""
Document chunker for RAG. Splits text into smaller pieces while trying to
preserve semantic boundaries (headers, paragraphs, code blocks).
"""

import re


class SemanticChunker:
    """Chunks documents while preserving structure."""
    
    def __init__(self, chunk_size=600, overlap=100, min_size=100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_size = min_size
    
    def chunk_document(self, content, metadata=None):
        """Split a document into chunks."""
        if not content or not content.strip():
            return []
        
        sections = self._split_by_structure(content)
        
        chunks = []
        for sec in sections:
            if len(sec['text']) > self.chunk_size:
                chunks.extend(self._split_section(sec))
            elif len(sec['text']) >= self.min_size:
                chunks.append(sec)
        
        # Add metadata
        for i, c in enumerate(chunks):
            c['metadata'] = {**(metadata or {}), 'chunk_id': i, 'total': len(chunks), 'size': len(c['text'])}
        
        return chunks
    
    def _split_by_structure(self, text):
        """Split text by headers, protecting code blocks."""
        # Extract and protect code blocks
        code_blocks = []
        def save_code(m):
            code_blocks.append(m.group(0))
            return f"__CB_{len(code_blocks)-1}__"
        
        text = re.sub(r'```[\s\S]*?```', save_code, text)
        
        # Split by headers
        sections = []
        current_text = ""
        current_header = "Introduction"
        
        for line in text.split('\n'):
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous
                if current_text.strip():
                    restored = self._restore_code(current_text, code_blocks)
                    sections.append({'text': restored.strip(), 'header': current_header})
                
                current_header = header_match.group(2).strip()
                current_text = line + '\n'
            else:
                current_text += line + '\n'
        
        # Don't forget the last section
        if current_text.strip():
            restored = self._restore_code(current_text, code_blocks)
            sections.append({'text': restored.strip(), 'header': current_header})
        
        return sections or [{'text': text, 'header': 'Document'}]
    
    def _restore_code(self, text, blocks):
        """Put code blocks back."""
        for i, code in enumerate(blocks):
            text = text.replace(f"__CB_{i}__", code)
        return text
    
    def _split_section(self, section):
        """Split a large section into smaller chunks."""
        text = section['text']
        header = section['header']
        
        paragraphs = text.split('\n\n')
        chunks = []
        current = ""
        
        for para in paragraphs:
            if len(current) + len(para) + 2 <= self.chunk_size:
                current += para + '\n\n'
            else:
                if len(current) >= self.min_size:
                    chunks.append({'text': current.strip(), 'header': header})
                    # Keep some overlap
                    overlap = current[-self.overlap:] if len(current) > self.overlap else current
                    current = overlap + para + '\n\n'
                else:
                    current += para + '\n\n'
        
        if current.strip() and len(current) >= self.min_size:
            chunks.append({'text': current.strip(), 'header': header})
        
        # Fall back to sentence splitting if needed
        if not chunks:
            chunks = self._split_by_sentences(text, header)
        
        return chunks or [section]
    
    def _split_by_sentences(self, text, header):
        """Last resort: split by sentences."""
        sentences = re.split(r'[.!?]+\s+', text)
        chunks = []
        current = ""
        
        for sent in sentences:
            if len(current) + len(sent) <= self.chunk_size:
                current += sent + '. '
            else:
                if current.strip():
                    chunks.append({'text': current.strip(), 'header': header})
                current = sent + '. '
        
        if current.strip():
            chunks.append({'text': current.strip(), 'header': header})
        
        return chunks
    
    def get_stats(self, chunks):
        """Get statistics about chunks."""
        if not chunks:
            return {'count': 0, 'total_chars': 0, 'avg_size': 0}
        
        sizes = [len(c.get('text', '')) for c in chunks]
        return {
            'count': len(chunks),
            'total_chars': sum(sizes),
            'avg_size': sum(sizes) // len(sizes),
            'min': min(sizes),
            'max': max(sizes)
        }


# Alias for backward compatibility
class DocumentChunker(SemanticChunker):
    def __init__(self, chunk_size=800, overlap=150):
        super().__init__(chunk_size=chunk_size, overlap=overlap)
    
    def chunk_by_section(self, text):
        return self.chunk_document(text)
    
    def chunk_simple(self, text):
        """Simple fixed-size chunking."""
        chunks = []
        pos = 0
        
        while pos < len(text):
            end = pos + self.chunk_size
            chunk = text[pos:end]
            
            # Try to break at a good point
            if end < len(text):
                for sep in ['. ', '\n\n', '\n', ' ']:
                    idx = chunk.rfind(sep)
                    if idx > self.chunk_size * 0.7:
                        chunk = text[pos:pos + idx + len(sep)]
                        end = pos + idx + len(sep)
                        break
            
            if chunk.strip():
                chunks.append({'text': chunk.strip(), 'header': f'Part {len(chunks)+1}'})
            
            pos = end - self.overlap
        
        return chunks
