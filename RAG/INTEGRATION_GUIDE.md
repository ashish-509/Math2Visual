# Math2Visual RAG Implementation Guide

## Team Integration Instructions

Hello team! I've implemented a RAG (Retrieval-Augmented Generation) system for our Math2Visual project. This document will guide you through understanding and using the enhanced system I've built.

## What is RAG and How I've Enhanced Our Project

I've integrated **RAG (Retrieval-Augmented Generation)** to combine our fine-tuned CodeLlama model with real-time access to Manim documentation. Here's what our enhanced system now does:

1. **Searches through Manim documentation** to find relevant examples
2. **Understands current best practices** and API changes
3. **Generates more accurate and comprehensive code** by incorporating official examples
4. **Provides documentation references** so users can learn more

## System Architecture I've Implemented

Here's how I've enhanced our original system:

```
Our Previous System:
User Query → Fine-tuned CodeLlama → Manim Code

My Enhanced System with RAG:
User Query → Document Search → Context + Query → Fine-tuned CodeLlama → Enhanced Manim Code
```

## New Project Structure I've Created

```
Major_Project/
├── code/
│   ├── app.py                   # Our original app (I've enhanced with RAG toggle)
│   ├── finetuningCode.ipynb     # Our fine-tuning notebook
│   └── finetuned-codellama-manim/ # Our trained model
├── RAG/                         # New RAG system I've built
│   ├── document_processor.py    # My Manim docs scraper
│   ├── vector_store.py          # My vector embeddings system
│   ├── rag_pipeline.py          # My RAG workflow implementation
│   ├── app_with_rag.py          # Enhanced Streamlit app I created
│   ├── setup.py                 # Automated setup script I wrote
│   ├── test_rag.py              # Testing script I built
│   ├── requirements.txt         # RAG dependencies I've identified
│   └── README.md                # Technical documentation I wrote
└── docs/                        # Our existing documentation
```

## Team Setup Instructions

### Prerequisites
- Our fine-tuned CodeLlama model should be trained and available
- Python environment with our existing dependencies
- Internet connection for downloading Manim documentation

### 1. Install RAG Dependencies I've Added

```bash
cd RAG
pip install -r requirements.txt
```

- `chromadb`: Vector database for document search
- `sentence-transformers`: For creating text embeddings
- `beautifulsoup4`: For web scraping Manim docs
- `langchain`: RAG utilities



```bash
# Windows
setup.bat

# Or manually
python setup.py
```

My automation script will:
- Download and process Manim documentation (~30 pages)
- Create vector embeddings for semantic search
- Test the complete pipeline I've built
- Setup the vector database

### 3. Manual Setup (If You Prefer Step-by-Step)

```bash
# Test my RAG components first
python test_rag.py

# Process documentation using my scraper
python document_processor.py

# Create vector store using my implementation
python vector_store.py

# Test my full RAG pipeline
python rag_pipeline.py
```

## How to Use My Enhanced System

### Option 1: My Enhanced RAG App (Recommended for Team) 
```bash
cd RAG
streamlit run app_with_rag.py
```

Features I've added:
- Complete RAG interface
- Shows retrieved documentation
- Configurable parameters
- Better error handling

### Option 2: Our Original App with My RAG Enhancement
```bash
cd code
streamlit run app.py
```

I've enhanced our original app with:
- Checkbox to enable RAG
- Fallback to standard generation
- Integration with my RAG system

### Option 3: Programmatic Use of My RAG System
```python
from RAG.rag_pipeline import Math2VisualRAG

# Initialize my RAG system
rag = Math2VisualRAG()

# Generate enhanced code using my implementation
result = rag.generate_manim_code("Create a sine wave animation")
if result['success']:
    print("Generated Code:")
    print(result['code'])
    
    print("\nRetrieved Documentation:")
    for doc in result['retrieved_docs']:
        print(f"- {doc['metadata']['title']}")
```

## Benefits My Implementation Brings to Our Project

### 1. Improved Code Quality
- **Current best practices**: Always uses latest Manim patterns
- **Complete imports**: Includes all necessary imports automatically
- **Proper structure**: Follows official documentation structure

### 2. Better User Experience
- **Documentation references**: Users can see where code patterns come from
- **Learning opportunity**: Users learn Manim concepts while generating code
- **Fallback protection**: If my RAG fails, falls back to our original model

### 3. Academic Value I've Added
- **Novel approach**: I've combined fine-tuning with real-time retrieval
- **Measurable improvement**: We can now compare with/without RAG performance
- **Research contribution**: I've documented effectiveness of RAG in code generation

## Example: Before vs After My Enhancement

### Our Original Output (Fine-tuned model only):
```python
from manim import *

class SineWave(Scene):
    def construct(self):
        axes = Axes()
        func = axes.plot(lambda x: np.sin(x))
        self.add(axes, func)
```

### My Enhanced Output (With RAG):
```python
from manim import *
import numpy as np

class SineWaveAnimation(Scene):
    def construct(self):
        # Create coordinate system
        axes = Axes(
            x_range=[-2*PI, 2*PI, PI/2],
            y_range=[-1.5, 1.5, 0.5],
            x_length=10,
            y_length=6
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y")
        
        # Create sine function
        sine_function = axes.plot(lambda x: np.sin(x), color=BLUE)
        sine_label = MathTex("y = \\sin(x)").next_to(axes, UP)
        
        # Animate creation
        self.play(Create(axes), Write(axes_labels))
        self.play(Write(sine_label))
        self.play(Create(sine_function), run_time=3)
        self.wait(2)
```

Notice my enhanced version includes:
- Proper axis configuration from documentation
- Labels and mathematical notation
- Animation timing and structure
- Complete imports

## Configuration Options I've Provided

### Generation Parameters
- `max_new_tokens`: Length of generated code (100-1000)
- `temperature`: Creativity level (0.1-2.0)
- `top_p`: Diversity control (0.1-1.0)

### My RAG Parameters
- `n_results`: Number of documents to retrieve (3-10)
- `include_code`: Whether to include code examples
- `max_docs`: Documentation pages to process (10-50)

## Troubleshooting Guide I've Prepared

### Common Issues

1. **Model not found**
   - Ensure our fine-tuned model is in the correct path
   - Update `model_path` in my RAG configuration

2. **Memory issues**
   - Reduce `max_docs` parameter in my configuration
   - Use CPU instead of GPU for embeddings
   - Reduce batch size in my vector store

3. **Internet connection issues**
   - My documentation download requires internet
   - Once downloaded, my system works offline
   - Use cached version if available

4. **Import errors**
   - Install my requirements: `pip install -r requirements.txt`
   - Check Python path configuration
   - Verify virtual environment

### Performance Tips I've Optimized

1. **First run takes longer** (downloads and processes docs)
2. **Subsequent runs are fast** (uses my cached embeddings)
3. **GPU recommended** for model inference
4. **CPU sufficient** for my document retrieval

## Integration with Our Major Project Report

### My Technical Contribution
1. **Novel Architecture**: I've combined fine-tuning + RAG for mathematical code generation
2. **Domain-Specific Application**: Mathematics education through animation
3. **Performance Comparison**: Quantitative analysis of with/without my RAG system

### Research Questions We Can Now Address
1. How does my RAG implementation improve code generation quality?
2. What types of queries benefit most from my documentation retrieval?
3. How does context length affect generation quality in my system?
4. Can my RAG reduce hallucination in code generation?

### Evaluation Metrics We Can Use
1. **Code Correctness**: Syntax and runtime errors
2. **Manim Compliance**: Use of proper Manim patterns
3. **Completeness**: Inclusion of imports, proper structure
4. **User Satisfaction**: Subjective quality assessment

## Future Enhancements I'm Planning

1. **Custom Dataset Integration**: Include our fine-tuning data in retrieval
2. **Multi-modal RAG**: Include Manim gallery images
3. **Interactive Learning**: Learn from user corrections
4. **Advanced Filtering**: Filter by complexity, topic, etc.

## Summary

I've transformed our Math2Visual project from a simple fine-tuned model to a sophisticated system that combines the benefits of:
- **Specialized training** (our fine-tuned model)
- **Current knowledge** (real-time documentation access)
- **Best practices** (official examples and patterns)

The result is more accurate, comprehensive, and educational code generation that better serves our users while providing significant academic value for our major project.

---

**Ready to get started, team?** Run `setup.bat` (Windows) or `python setup.py` to begin using my RAG implementation!
