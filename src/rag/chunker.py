"""
Enhanced document chunking with semantic-aware splitting.
Preserves code blocks, headers, and semantic boundaries for better retrieval.
"""

import re
from typing import List, Dict, Optional


class SemanticChunker:
    
    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        preserve_code_blocks: bool = True,
        min_chunk_size: int = 100
    ):                      
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_code_blocks = preserve_code_blocks
        self.min_chunk_size = min_chunk_size
    

    def chunk_document(
        self,
        content: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
       
        if not content or not content.strip():
            return []
        
        # Split by markdown structure (headers + code blocks)
        sections = self._split_by_structure(content)
        
        # Further split large sections
        chunks = []
        for section in sections:
            if len(section['text']) > self.chunk_size:
                sub_chunks = self._split_large_section(section)
                chunks.extend(sub_chunks)
            elif len(section['text']) >= self.min_chunk_size:
                chunks.append(section)
        
        # Add metadata and numbering
        for i, chunk in enumerate(chunks):
            chunk['metadata'] = {
                **(metadata or {}),
                'chunk_id': i,
                'total_chunks': len(chunks),
                'size': len(chunk['text'])
            }
        
        return chunks

    
    def _split_by_structure(self, text: str) -> List[Dict]:
        
        sections = []
        
        # Regex patterns
        header_pattern = r'^(#{1,6})\s+(.+)$'
        code_block_pattern = r'```[\s\S]*?```'
        
        # Extract code blocks first to protect them
        code_blocks = []
        def store_code_block(match):
            idx = len(code_blocks)
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{idx}__"
        
        if self.preserve_code_blocks:
            text = re.sub(code_block_pattern, store_code_block, text, flags=re.MULTILINE)
        
        # Split by headers
        lines = text.split('\n')
        current_section = ""
        current_header = "Introduction"
        
        for line in lines:
            header_match = re.match(header_pattern, line)
            
            if header_match:
                # Save previous section
                if current_section.strip():
                    # Restore code blocks
                    restored = self._restore_code_blocks(current_section, code_blocks)
                    sections.append({
                        'text': restored.strip(),
                        'header': current_header,
                        'size': len(restored)
                    })
                
                # Start new section
                level = len(header_match.group(1))
                current_header = header_match.group(2).strip()
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        # Last section
        if current_section.strip():
            restored = self._restore_code_blocks(current_section, code_blocks)
            sections.append({
                'text': restored.strip(),
                'header': current_header,
                'size': len(restored)
            })
        
        return sections if sections else [{'text': text, 'header': 'Document', 'size': len(text)}]
    

    def _restore_code_blocks(self, text: str, code_blocks: List[str]) -> str:
        
        for idx, code in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{idx}__", code)
        return text
    

    def _split_large_section(self, section: Dict) -> List[Dict]:
       
        text = section['text']
        header = section['header']
        
        # Try splitting by paragraphs first
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # Check if adding this paragraph exceeds limit
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk += para + '\n\n'
            else:
                # Save current chunk if it's substantial
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'header': header,
                        'size': len(current_chunk)
                    })
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap(current_chunk)
                    current_chunk = overlap_text + para + '\n\n'
                else:
                    current_chunk += para + '\n\n'
        
        # Last chunk
        if current_chunk.strip() and len(current_chunk) >= self.min_chunk_size:
            chunks.append({
                'text': current_chunk.strip(),
                'header': header,
                'size': len(current_chunk)
            })
        
        # If no good split found, fall back to sentence splitting
        if not chunks:
            chunks = self._split_by_sentences(text, header)
        
        return chunks if chunks else [section]
    

    def _split_by_sentences(self, text: str, header: str) -> List[Dict]:
       
        # Simple sentence boundary detection
        sentence_pattern = r'[.!?]+\s+'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        current_chunk = ""
        
        for sent in sentences:
            if len(current_chunk) + len(sent) <= self.chunk_size:
                current_chunk += sent + '. '
            else:
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'header': header,
                        'size': len(current_chunk)
                    })
                
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = overlap_text + sent + '. '
        
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'header': header,
                'size': len(current_chunk)
            })
        
        return chunks
    

    def _get_overlap(self, text: str) -> str:
        
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to break at sentence boundary
        overlap_start = len(text) - self.chunk_overlap
        overlap_text = text[overlap_start:]
        
        # Find first sentence start
        sentence_start = re.search(r'[.!?]\s+', overlap_text)
        if sentence_start:
            overlap_text = overlap_text[sentence_start.end():]
        
        return overlap_text
    
    
    def get_stats(self, chunks: List[Dict]) -> Dict:
       
        if not chunks:
            return {
                'num_chunks': 0,
                'total_chars': 0,
                'avg_chunk_size': 0,
                'min_size': 0,
                'max_size': 0
            }
        
        sizes = [c.get('size', len(c.get('text', ''))) for c in chunks]
        
        return {
            'num_chunks': len(chunks),
            'total_chars': sum(sizes),
            'avg_chunk_size': int(sum(sizes) / len(sizes)),
            'min_size': min(sizes),
            'max_size': max(sizes)
        }


# Backward compatibility alias
class DocumentChunker(SemanticChunker):
    
    def __init__(self, chunk_size=800, overlap=150):
        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
    

    def chunk_by_section(self, text: str) -> List[Dict]:
       
        chunks = self.chunk_document(text)
        # Convert to old format
        return [{'text': c['text'], 'header': c['header'], 'size': c['size']} 
                for c in chunks]

    
    def chunk_simple(self, text: str) -> List[Dict]:
       
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_len:
                for sep in ['. ', '\n\n', '\n', ' ']:
                    break_point = chunk_text.rfind(sep)
                    if break_point > self.chunk_size * 0.7:
                        end = start + break_point + len(sep)
                        chunk_text = text[start:end]
                        break
            
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text.strip(),
                    'header': f'Chunk {len(chunks) + 1}',
                    'size': len(chunk_text)
                })
            
            start = end - self.chunk_overlap
        
        return chunks
