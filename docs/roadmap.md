# Manim Code Dataset Roadmap

## 1. **Goal**
Create a high-quality, diverse dataset of Manim code snippets, each paired with a plain English math prompt, for training/fine-tuning LLMs to generate Manim code from natural language descriptions.

---

## 2. **Key Sources to Scrape**

| Source                | What We Get          | How to Scrape          | Prompt Quality       |
|-----------------------|---------------------- |----------------------- |---------------------|
| GitHub repos          | Raw Manim code        | Script + regex         | Manual/AI labeling  |
| Manim docs/examples   | Code + description    | BeautifulSoup          | Good                |
| Blogs/tutorials       | Code + text           | BeautifulSoup          | Good                |
| YouTube notebooks     | Code                  | Manual download        | Manual/AI labeling  |

---

## 3. **Step-by-Step Workflow**

### **A. Gather Raw Data**

#### 1. **GitHub Repositories**
- Search for relevant repos:
  - [`3Blue1Brown/manim`](https://github.com/3b1b/manim)
  - [`ManimCommunity/manim`](https://github.com/ManimCommunity/manim)
  - [`ManimCommunity/examples`](https://github.com/ManimCommunity/examples)
  - [`Awesome Manim`](https://github.com/ManimCommunity/awesome-manim) 
- Clone/download repositories locally.

#### 2. **Manim Official Docs**
- Visit: [Manim documentation examples](https://docs.manim.community/en/stable/examples.html)
- Use BeautifulSoup to scrape code blocks and paired text descriptions.

#### 3. **Blogs & Tutorials**
- Identify blog posts, tutorials, and Medium articles with Manim examples.
- Use BeautifulSoup to extract code and surrounding explanatory text.

#### 4. **YouTube Notebooks**
- Download Python notebooks/scripts attached to 3Blue1Brown videos and others.

---

### **B. Extract and Organize Code**

#### 1. **Extract Manim Scenes from Python Files**
- Walk through all `.py` files recursively.
- Use regex or Python’s `ast` module to extract `class ...Scene(...):` blocks.
- Collect preceding or inline comments/docstrings as possible prompts.

#### 2. **Extract from HTML Pages**
- For docs/blogs, scrape all code blocks and their nearest descriptions.

#### 3. **Deduplicate**
- Hash and remove duplicate code snippets.

---

### **C. Prompt Creation**

#### 1. **If Prompt Available**
- Use directly (from docs/blogs/examples).

#### 2. **If Not Available**
- Use preceding comments, or
- Generate prompts with GPT-4/Claude (“Summarize this Manim code in a plain English math instruction.”).
- Mark any unclear/unlabeled cases as `"unknown"` for future human/AI labeling.

---

### **D. Data Structuring**

- Store each data pair (jsonl format) as:
  ```json
  {"prompt": "Plot the graph of y = x^2", "code": "<manim code here>"}
---

### **E. Final Cleaning & Validation**

- **Run all code snippets** to ensure each Manim code block executes without errors.
- Visually check: Verify that the code output matches the prompt description, as feasible.
- **Log gold standard pairs**: Maintain a curated list of hand-checked (high-quality) prompt-code pairs for model evaluation and testing.

---

## 4. **Scripts & Automation**

- **GitHub Repo Download**: Script for automating the download/clone of public Manim-related repositories.
- **Python File Traversal & Extraction**: Script to recursively scan `.py` files and extract `class ...Scene(...)` blocks using regex or Python's `ast` module.
- **HTML Scraping**: Use `requests` and `BeautifulSoup` to extract code and description blocks from documentation, blogs, or tutorials.
- **Prompt Generation (AI-assisted)**: Script to batch-submit code snippets to LLM APIs (e.g., GPT-4, Claude) for generating English prompts or summarizations.
- **Deduplication**: Script to hash and remove duplicate code snippets.
- **Validation**:
  - Script to batch-run and check Manim code for syntax/runtime errors.
  - Script to batch-render short animation previews for visual validation.

---

## 5. **Best Practices**

- **Document scraping and extraction process** for transparency and reproducibility.
- **Use Git for version control**: Keep all scripts, dataset versions, and processing logs in a Git repository.
- **Metadata**: For each entry, keep track of:
  - Source URL or repository name
  - Prompt origin (manual, AI-generated, or unknown)
  - Validation status (validated, error, unchecked)

---


## 6. **Checklist**

Mark each step `[x]` when complete!

- [ ] **Gather code from all major sources**
  - [ ] Clone/download GitHub repositories (e.g., 3Blue1Brown, Manim Community, examples)
  - [ ] Scrape Manim docs/examples
  - [ ] Extract from blogs, tutorials, and YouTube notebooks

- [ ] **Extract and deduplicate all Manim Scene classes**
  - [ ] Recursively scan all `.py` files for `class ...Scene(...)`
  - [ ] Use regex or AST parsing to extract full code blocks
  - [ ] Remove duplicate scenes (hashing)

- [ ] **Pair with prompts (manual or AI-generated)**
  - [ ] Use comments/docstrings where available
  - [ ] Generate missing prompts via LLMs (e.g., GPT-4/Claude)
  - [ ] Mark "unknown" where prompt is missing for future labeling

- [ ] **Store as JSONL, with clear metadata**
  - [ ] For each pair, store `{"prompt": "...", "code": "...", "source": "...", "origin": "...", "validated": ...}`
  - [ ] Use consistent formatting for all entries

- [ ] **Validate code and check for errors**
  - [ ] Batch-run code snippets to ensure they execute in Manim
  - [ ] Flag and fix (or remove) erroneous code blocks
  - [ ] Visually verify animation output matches prompt

- [ ] **Prepare test/gold set for evaluation**
  - [ ] Curate a set of 50–300 hand-checked, diverse prompt-code pairs
  - [ ] Reserve for evaluation only (do not include in training)

---

