# Takes large docs and splits them into smaller pieces for better retrieval

import re
from typing import List, Dict

class DocumentChunker:
    def __init__(self, chunk_size=800, overlap=150):
        # chunk_size: max chars per chunk
        # overlap: chars that overlap between chunks to keep context
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_by_section(self, text):
        """
        Split doc by markdown headers first, then by size if needed.
        Returns list of dicts with chunk text and metadata
        """
        chunks = []
        
        # find all headers (# Header, ## Header, etc)
        section_pattern = r'^(#{1,6})\s+(.+)$'
        lines = text.split('\n')
        
        current_section = ""
        current_header = "Introduction"
        current_level = 0
        
        for line in lines:
            header_match = re.match(section_pattern, line, re.MULTILINE)
            
            if header_match:
                # save previous section if it exists
                if current_section.strip():
                    section_chunks = self._split_large_section(current_section, current_header)
                    chunks.extend(section_chunks)
                
                # start new section
                level = len(header_match.group(1))
                current_header = header_match.group(2).strip()
                current_level = level
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        # don't forget last section
        if current_section.strip():
            section_chunks = self._split_large_section(current_section, current_header)
            chunks.extend(section_chunks)
        
        return chunks
    
    def _split_large_section(self, text, header):
        # Break up sections that are bigger than chunk_size
        chunks = []
        
        if len(text) <= self.chunk_size:
            chunks.append({
                'text': text.strip(),
                'header': header,
                'size': len(text)
            })
            return chunks
        
        # section is too big, need to split it
        words = text.split()
        current_chunk = ""
        
        for word in words:
            if len(current_chunk) + len(word) + 1 <= self.chunk_size:
                current_chunk += word + ' '
            else:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'header': header,
                        'size': len(current_chunk)
                    })
                
                # start new chunk with overlap
                overlap_words = current_chunk.split()[-15:]  # keep last 15 words
                current_chunk = ' '.join(overlap_words) + ' ' + word + ' '
        
        # last chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'header': header,
                'size': len(current_chunk)
            })
        
        return chunks
    
    def chunk_simple(self, text):
        """
        Simple chunking - just split by size with overlap.
        Useful for unstructured text without clear sections
        """
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # try to break at sentence boundary
            if end < text_len:
                last_period = chunk_text.rfind('.')
                last_newline = chunk_text.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.7:  # dont break too early
                    end = start + break_point + 1
                    chunk_text = text[start:end]
            
            chunks.append({
                'text': chunk_text.strip(),
                'header': f'Chunk {len(chunks) + 1}',
                'size': len(chunk_text)
            })
            
            start = end - self.overlap
        
        return chunks

    def get_stats(self, chunks):
        """Quick stats about chunks"""
        total_size = sum(c['size'] for c in chunks)
        avg_size = total_size / len(chunks) if chunks else 0
        
        return {
            'num_chunks': len(chunks),
            'total_chars': total_size,
            'avg_chunk_size': int(avg_size),
            'min_size': min((c['size'] for c in chunks), default=0),
            'max_size': max((c['size'] for c in chunks), default=0)
        }
