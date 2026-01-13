# Math2Visual : From Natural Language prompts to Animated Mathematical videos

## Introduction

Mathematics is best understood visually, but creating high-quality math animations is time-consuming and requires coding skills. This project aims to bridge that gap by building a robust, scalable, and optimized system that allows users to generate Manim animations from natural language prompts either typed or spoken. The system leverages fine-tuned large language models (LLMs), Retrieval-Augmented Generation (RAG) with Manim documentation, and advanced speech technologies to deliver a seamless, interactive, and educational experience.

## Project Pipeline Overview

1. **User Input (Text or Speech):**
   - The user enters a math concept to explain, either by typing or using speech-to-text (STT).

2. **Model Selection:**
   - The user selects one of the fine-tuned models (CodeLlama, Mistral, or Phi-2) from the Streamlit UI.

3. **RAG Context Injection:**
   - The system retrieves relevant context from Manim documentation (scraped and converted to markdown) and injects it into the prompt, ensuring the LLM understands Manim syntax and reduces hallucinations.

4. **Manim Code Generation:**
   - The selected LLM generates Manim code for the given math concept, using the provided context.

5. **Code Compilation and Animation Preview:**
   - On clicking the "Generate Animated Video" button, the system compiles the generated Manim code and displays the animation in the UI.

6. **Video Explanation Generation:**
   - If the user clicks "Explain Video", another LLM generates a script to explain the animation based on the Manim code.

7. **Text-to-Speech (TTS) with Timing Synchronization:**
   - The script is converted to speech using a TTS model, with careful timing and pauses to make the narration sound natural, professional, and human-like.

8. **Final Output:**
   - The user receives a professional, academic-quality animated video with synchronized, human-sounding narration.

## Key Features

- **Robust and Scalable:** Modular pipeline, easy to add new models or update components.
- **Optimized:** Efficient use of memory (QLoRA, quantization), fast inference, and caching.
- **User-Friendly:** Streamlit UI with clear options and feedback.
- **Voice and Text Support:** Both input and output support speech and text.
- **RAG for Accuracy:** Reduces LLM hallucination by grounding code generation in real Manim docs.
- **Professional Output:** Synchronized narration and animation for a polished, academic feel.

## Future Scope
- Add more LLMs and support for other animation libraries.
- Deploy as a web service or integrate with LMS platforms.
- Improve TTS for multilingual and emotional narration.

---

# How It Works

1. **Prompt Input:** User enters or speaks a math concept.
2. **Model Selection:** User chooses a fine-tuned LLM.
3. **RAG Context:** System fetches relevant Manim doc snippets.
4. **Code Generation:** LLM generates Manim code.
5. **Animation:** Code is compiled and animation previewed.
6. **Explanation:** LLM generates script, TTS narrates with timing.
7. **Final Video:** User gets a narrated, animated math explanation.

---

# Tech Stack
- Python, Streamlit, Manim
- Transformers (HuggingFace), peft, trl, accelerate, bitsandbytes
- SpeechRecognition, TTS libraries
- Docker (for deployment)

---