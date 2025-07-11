# Crawler Module Roadmap

## Overview
The crawler module is designed to extract documentation from the Manim Community website and use it to create an AI-powered Manim code generator. This module consists of web scraping components and an AI-based code generation system.

## Architecture Flow
```
Manim Docs Website → Web Crawler → Markdown File → AI Model → Generated Manim Code
```

## File Structure Analysis

### 1. `crawl.py` - Main Web Scraping Script
**Purpose**: Standalone Python script for crawling Manim documentation

**Key Components**:
- **Dependencies**: 
  - `asyncio`: For asynchronous web crawling
  - `crawl4ai.AsyncWebCrawler`: Main web crawling library
- **Functionality**:
  - Creates AsyncWebCrawler instance
  - Crawls `https://docs.manim.community/en/stable/index.html`
  - Extracts content and converts to markdown format
  - Saves output to `markdown_output.md`

**Execution Flow**:
```python
1. Initialize AsyncWebCrawler
2. Crawl Manim documentation URL
3. Extract content as markdown
4. Save to file
5. Print extracted content
```

### 2. `crawler.ipynb` - Interactive Crawler Notebook
**Purpose**: Jupyter notebook version for interactive development and testing

**Cell Structure**:
- **Cell 1**: Install crawl4ai package
- **Cell 2**: Import required libraries
- **Cell 3**: Define async main function
- **Cell 4**: Apply nest_asyncio for notebook compatibility
- **Cell 5**: Execute crawler function
- **Cell 6**: Alternative crawler implementation
- **Cell 7**: Execute alternative crawler

**Key Features**:
- Interactive development environment
- Multiple crawler implementations
- Uses `nest_asyncio` to handle async execution in Jupyter

### 3. `chat_with_manim_docs.ipynb` - AI Code Generator
**Purpose**: AI-powered Manim code generator using scraped documentation

**Cell Structure**:
- **Cell 1**: Install Google GenAI package
- **Cell 2**: Load scraped documentation from markdown file
- **Cell 3**: Display loaded documentation
- **Cell 4**: Initialize Google GenAI client with API key
- **Cell 5**: Generate Manim code using AI model

**AI Integration**:
- **Model**: Gemini-2.5-flash
- **Input**: User prompt + Manim documentation
- **Output**: Generated Manim code
- **Example**: Visualization of Pythagorean theorem

### 4. `markdown_output.md` - Scraped Documentation
**Purpose**: Contains the scraped Manim community documentation in markdown format

**Content Structure**:
- Navigation menu and site structure
- Installation guides
- Tutorials and guides
- Reference manual
- API documentation
- Code examples

### 5. `markdown_output - Copy.md` - Backup Documentation
**Purpose**: Backup copy of the scraped documentation

## Technical Implementation Details

### Web Crawling Strategy
1. **Asynchronous Crawling**: Uses `AsyncWebCrawler` for efficient web scraping
2. **Target URL**: Manim Community documentation homepage
3. **Output Format**: Markdown for better AI model consumption
4. **Error Handling**: Basic async/await pattern with context managers

### AI Code Generation Pipeline
1. **Documentation Loading**: Reads scraped markdown content
2. **Prompt Engineering**: Combines user request with documentation context
3. **AI Model**: Google's Gemini-2.5-flash for code generation
4. **Context Window**: Entire documentation as reference material

## Usage Workflow

### Step 1: Documentation Scraping
```bash
# Option 1: Run standalone script
python crawl.py

# Option 2: Use interactive notebook
# Execute crawler.ipynb cells sequentially
```

### Step 2: AI Code Generation
```python
# Execute chat_with_manim_docs.ipynb
# Modify prompt in final cell for different Manim visualizations
```

## Dependencies and Requirements

### Core Libraries
- `crawl4ai`: Web crawling framework
- `google-genai`: Google's Generative AI SDK
- `asyncio`: Asynchronous programming
- `nest_asyncio`: Jupyter async compatibility

### System Requirements
- Python 3.7+
- Internet connection for crawling and AI API
- Google GenAI API key

## Configuration

### API Configuration
- **Google GenAI API Key**: Required for code generation
- **Model**: Gemini-2.5-flash (configurable)
- **Crawling URL**: Manim documentation site (configurable)

### File Paths
- Documentation output: `markdown_output.md`
- Backup output: `markdown_output - Copy.md`

## Potential Improvements

### 1. Enhanced Crawling
- **Recursive Crawling**: Crawl multiple documentation pages
- **Content Filtering**: Extract only relevant sections
- **Update Mechanism**: Automatic documentation updates

### 2. Better AI Integration
- **Chunking Strategy**: Handle large documentation efficiently
- **Prompt Optimization**: Better prompt engineering for code quality
- **Multi-model Support**: Support for different AI providers

### 3. Code Quality
- **Error Handling**: Robust error handling throughout pipeline
- **Logging**: Comprehensive logging system
- **Testing**: Unit tests for crawler and AI components

### 4. User Experience
- **Interactive UI**: Web interface for code generation
- **Code Validation**: Syntax checking for generated code
- **Examples Gallery**: Pre-built examples and templates

## Security Considerations

### API Key Management
- **Current**: Hardcoded API key (security risk)
- **Recommended**: Environment variables or secure key management

### Web Scraping Ethics
- **Respect robots.txt**: Check site's crawling policies
- **Rate Limiting**: Implement proper request throttling
- **Terms of Service**: Ensure compliance with Manim site terms

## Future Roadmap

### Phase 1: Core Improvements
- Implement proper error handling
- Add configuration management
- Create comprehensive test suite

### Phase 2: Feature Expansion
- Multi-page documentation crawling
- Advanced prompt engineering
- Code quality assessment

### Phase 3: Production Ready
- Web application interface
- User authentication system
- Code execution environment

## Conclusion

The crawler module provides a solid foundation for creating an AI-powered Manim code generator. The current implementation demonstrates the core concept of web scraping documentation and using it as context for AI code generation. With the suggested improvements, this system can become a powerful tool for Manim developers and educators.
