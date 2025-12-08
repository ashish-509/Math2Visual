# Math2Visual RAG System

This directory contains the RAG (Retrieval-Augmented Generation) implementation for the Math2Visual project, which enhances our fine-tuned CodeLlama model with Manim documentation retrieval.

## Overview

The RAG system improves our Math2Visual project by:
1. **Retrieving relevant Manim documentation** based on user queries
2. **Providing context** to our fine-tuned CodeLlama model
3. **Generating more accurate and comprehensive** Manim code
4. **Including best practices** from official documentation

## Architecture

```
User Query → Document Retrieval → Context Enhancement → Code Generation
     ↓              ↓                     ↓                 ↓
Natural Language → Vector Search → Enhanced Prompt → Fine-tuned Model
```

## Components

### 1. Document Processor (`document_processor.py`)
- Scrapes and processes Manim documentation from official sources
- Extracts content, code examples, and metadata
- Handles content chunking for better retrieval

### 2. Vector Store (`vector_store.py`)
- Creates embeddings using Sentence Transformers
- Manages ChromaDB for efficient similarity search
- Supports filtering by content type (documentation vs code examples)

### 3. RAG Pipeline (`rag_pipeline.py`)
- Orchestrates the complete RAG workflow
- Combines retrieval with your fine-tuned CodeLlama model
- Provides enhanced prompts with documentation context

### 4. Enhanced Streamlit App (`app_with_rag.py`)
- User-friendly interface with RAG capabilities
- Shows retrieved documentation references
- Configurable RAG parameters

## Setup Instructions

### 1. Install Dependencies
```bash
cd RAG
pip install -r requirements.txt
```

### 2. Automatic Setup
```bash
python setup.py
```

This will:
- Install all required packages
- Download and process Manim documentation
- Create vector embeddings
- Test the RAG pipeline

### 3. Manual Setup (Alternative)

#### Process Documentation
```bash
python document_processor.py
```

#### Create Vector Store
```bash
python vector_store.py
```

#### Test RAG Pipeline
```bash
python rag_pipeline.py
```

## Usage

### Running the Enhanced App
```bash
streamlit run app_with_rag.py
```

### Using RAG Pipeline Programmatically
```python
from rag_pipeline import Math2VisualRAG

# Initialize RAG pipeline
rag = Math2VisualRAG()

# Setup knowledge base (first time only)
rag.setup_knowledge_base()

# Generate code with RAG
result = rag.generate_manim_code("Create a sine wave animation")
if result['success']:
    print(result['code'])
```

## Features

### Enhanced Code Generation
- **Context-aware**: Uses relevant Manim documentation
- **Best practices**: Incorporates official examples and patterns
- **Comprehensive**: Includes proper imports and scene structure

### Documentation Retrieval
- **Semantic search**: Finds relevant docs using similarity
- **Multi-type retrieval**: Both explanatory content and code examples
- **Relevance scoring**: Returns most relevant results first

### User Interface
- **Visual feedback**: Shows retrieved documentation references
- **Configurable parameters**: Adjust generation and retrieval settings
- **Example prompts**: Quick-start templates for common animations

## Configuration

### Model Settings
- `max_new_tokens`: Control output length (100-1000)
- `temperature`: Control creativity (0.1-2.0)
- `top_p`: Control diversity (0.1-1.0)

### RAG Settings
- `n_results`: Number of documents to retrieve (3-10)
- `include_code`: Whether to include code examples
- `chunk_size`: Document chunk size for better retrieval

## Examples

### Basic Usage
```python
rag = Math2VisualRAG()
result = rag.generate_manim_code("Draw a circle transforming into a square")
```

### With Custom Parameters
```python
result = rag.generate_manim_code(
    "Create 3D plot animation",
    max_new_tokens=500,
    temperature=0.8
)
```

## File Structure

```
RAG/
├── document_processor.py    # Documentation scraping and processing
├── vector_store.py         # Vector embeddings and similarity search
├── rag_pipeline.py         # Complete RAG workflow
├── app_with_rag.py         # Enhanced Streamlit application
├── setup.py                # Automated setup script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── chroma_db/             # Vector database (created automatically)
└── manim_docs.json        # Processed documentation (created automatically)
```

## Dependencies

- **transformers**: For CodeLlama model
- **chromadb**: Vector database
- **sentence-transformers**: Text embeddings
- **streamlit**: Web interface
- **beautifulsoup4**: Web scraping
- **langchain**: RAG utilities


## Integration with our Project

This RAG system is designed to work seamlessly with our existing Math2Visual project:

1. **Maintains compatibility** with our fine-tuned CodeLlama model
2. **Extends functionality** without breaking existing code
3. **Provides fallback** to original generation if RAG fails
4. **Easy integration** with our current Streamlit app

## Future Enhancements

- **Custom dataset integration**: Include our training data in retrieval
- **Multi-modal retrieval**: Include images from Manim gallery
- **Advanced filtering**: Filter by animation complexity or topic
- **Caching**: Cache generated code for common queries
- **Feedback loop**: Learn from user corrections and preferences

## Contributing

To extend the RAG system:
1. Add new document sources in `document_processor.py`
2. Implement custom embedding models in `vector_store.py`
3. Enhance prompt engineering in `rag_pipeline.py`
4. Add new UI features in `app_with_rag.py`
