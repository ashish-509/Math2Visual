"""
This module handles downloading, processing, and creating embeddings
for Manim documentation to be used in the RAG pipeline.
"""

import os
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import json
import time
from urllib.parse import urljoin, urlparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ManimeDocumentProcessor:
    def __init__(self, base_url: str = "https://docs.manim.community/en/stable/"):
        """
        Initialize the Manim documentation processor.
        
        Args:
            base_url: Base URL for Manim documentation
        """
        self.base_url = base_url
        self.processed_docs = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_manim_documentation_links(self) -> List[str]:
        """
        Extract all documentation links from Manim documentation.
        
        Returns:
            List of documentation URLs
        """
        try:
            response = self.session.get(self.base_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            links = []
            
            # Find all links in the documentation
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/') or href.startswith('http'):
                    full_url = urljoin(self.base_url, href)
                    if 'docs.manim.community' in full_url and full_url not in links:
                        links.append(full_url)
            
            # Filter for relevant documentation pages
            filtered_links = [
                link for link in links 
                if any(keyword in link.lower() for keyword in [
                    'reference', 'tutorial', 'example', 'guide', 'api',
                    'mobject', 'animation', 'scene', 'camera', 'config'
                ])
            ]
            
            logger.info(f"Found {len(filtered_links)} relevant documentation links")
            return filtered_links[:50]  # Limit to first 50 for initial implementation
            
        except Exception as e:
            logger.error(f"Error fetching documentation links: {e}")
            return []
    
    def extract_content_from_url(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a single documentation URL.
        
        Args:
            url: URL to extract content from
            
        Returns:
            Dictionary containing extracted content
        """
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.text.strip() if title else "No Title"
            
            # Extract main content
            content_selectors = [
                'main', 'article', '.content', '.documentation',
                '#main-content', '.rst-content'
            ]
            
            content_text = ""
            for selector in content_selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    content_text = content_div.get_text(separator=' ', strip=True)
                    break
            
            # If no main content found, get body text
            if not content_text:
                content_text = soup.get_text(separator=' ', strip=True)
            
            # Extract code examples
            code_blocks = []
            for code in soup.find_all(['code', 'pre']):
                code_text = code.get_text(strip=True)
                if len(code_text) > 20:  # Filter out small code snippets
                    code_blocks.append(code_text)
            
            return {
                'url': url,
                'title': title_text,
                'content': content_text,
                'code_examples': code_blocks,
                'length': len(content_text)
            }
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None
    
    def process_documentation(self, max_docs: int = 30) -> List[Dict[str, Any]]:
        """
        Process multiple documentation pages.
        
        Args:
            max_docs: Maximum number of documents to process
            
        Returns:
            List of processed documents
        """
        links = self.get_manim_documentation_links()
        processed_docs = []
        
        for i, link in enumerate(links[:max_docs]):
            logger.info(f"Processing {i+1}/{min(len(links), max_docs)}: {link}")
            
            doc = self.extract_content_from_url(link)
            if doc and doc['length'] > 100:  # Filter out very short documents
                processed_docs.append(doc)
            
            time.sleep(0.5)
        
        self.processed_docs = processed_docs
        logger.info(f"Successfully processed {len(processed_docs)} documents")
        return processed_docs
    
    def save_processed_docs(self, filename: str = "manim_docs.json"):
        """
        Save processed documents to a JSON file.
        
        Args:
            filename: Output filename
        """
        if not self.processed_docs:
            logger.warning("No processed documents to save")
            return
        
        output_path = os.path.join(os.path.dirname(__file__), filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.processed_docs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.processed_docs)} documents to {output_path}")
    
    def load_processed_docs(self, filename: str = "manim_docs.json") -> List[Dict[str, Any]]:
        """
        Load processed documents from a JSON file.
        
        Args:
            filename: Input filename
            
        Returns:
            List of loaded documents
        """
        file_path = os.path.join(os.path.dirname(__file__), filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} not found")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                docs = json.load(f)
            
            self.processed_docs = docs
            logger.info(f"Loaded {len(docs)} documents from {file_path}")
            return docs
            
        except Exception as e:
            logger.error(f"Error loading documents from {file_path}: {e}")
            return []

if __name__ == "__main__":
    processor = ManimeDocumentProcessor()
    
    print("Processing Manim documentation...")
    docs = processor.process_documentation(max_docs=20)
    
    processor.save_processed_docs()
    
    print(f"Processed {len(docs)} documents")
    if docs:
        print(f"Sample document: {docs[0]['title']}")
        print(f"Content length: {docs[0]['length']} characters")
