"""
This is an enhanced version of your original app that incorporates
RAG (Retrieval-Augmented Generation) with Manim documentation.
"""

import streamlit as st
import torch
import sys
import os
from typing import Dict, Any

# Add RAG module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RAG'))

from rag_pipeline import Math2VisualRAG

# Page configuration
st.set_page_config(
    page_title="Math2Visual RAG", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .code-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .doc-reference {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        border-left: 3px solid #17a2b8;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'knowledge_base_setup' not in st.session_state:
    st.session_state.knowledge_base_setup = False

@st.cache_resource(show_spinner="Loading RAG pipeline...")
def load_rag_pipeline():
    
    try:
        return Math2VisualRAG()
    except Exception as e:
        st.error(f"Error loading RAG pipeline: {e}")
        return None

def setup_knowledge_base(rag_pipeline, max_docs=20):
    if rag_pipeline and not st.session_state.knowledge_base_setup:
        with st.spinner("Setting up knowledge base with Manim documentation..."):
            try:
                rag_pipeline.setup_knowledge_base(max_docs=max_docs)
                st.session_state.knowledge_base_setup = True
                st.success("Knowledge base setup complete!")
            except Exception as e:
                st.error(f"Error setting up knowledge base: {e}")

def display_retrieved_docs(retrieved_docs):
    """Display retrieved documentation references."""
    if not retrieved_docs:
        return
    
    st.subheader("Retrieved Documentation References")
    
    for i, doc in enumerate(retrieved_docs):
        with st.expander(f"Reference {i+1}: {doc['metadata']['title']}"):
            st.markdown(f"**Type:** {doc['metadata']['type'].title()}")
            st.markdown(f"**Source:** {doc['metadata']['url']}")
            st.markdown(f"**Content:**")
            st.markdown(f"<div class='doc-reference'>{doc['content'][:500]}...</div>", 
                       unsafe_allow_html=True)

def main():
    st.markdown("<div class='main-header'> Math2Visual RAG — Enhanced Manim Code Generator</div>", 
                unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the enhanced Math2Visual generator! This version uses **RAG (Retrieval-Augmented Generation)** 
    to provide more accurate and comprehensive Manim code by leveraging the official Manim documentation.
    """)
    
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("Model Settings")
        max_tokens = st.slider("Max New Tokens", 100, 1000, 400)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7)
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9)
        
        st.subheader("RAG Settings")
        max_docs_to_setup = st.slider("Max Docs for Knowledge Base", 10, 50, 20)
        include_code_examples = st.checkbox("Include Code Examples", value=True)
        
        st.subheader("Knowledge Base Status")
        if st.session_state.knowledge_base_setup:
            st.success("Ready")
        else:
            st.warning("Not setup")
        
        if st.button("Refresh Knowledge Base"):
            st.session_state.knowledge_base_setup = False
            st.session_state.rag_pipeline = None
    
    if st.session_state.rag_pipeline is None:
        st.session_state.rag_pipeline = load_rag_pipeline()
    
    rag_pipeline = st.session_state.rag_pipeline
    
    if rag_pipeline is None:
        st.error("RAG pipeline not loaded. Please check your model path and dependencies.")
        return
    
    setup_knowledge_base(rag_pipeline, max_docs_to_setup)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Your Math Animation Request")
        prompt = st.text_area(
            "Describe the mathematical animation you want to create:",
            placeholder="e.g., Create a sine wave animation that shows the wave propagating from -2π to 2π with a frequency of 1 Hz...",
            height=150
        )
        
        st.markdown("**Example prompts:**")
        example_prompts = [
            "Create a sine wave animation",
            "Draw a circle that transforms into a square",
            "Animate the graph of y = x^2 with domain [-3, 3]",
            "Show the Pythagorean theorem with animated triangles",
            "Create a 3D plot of z = sin(x) * cos(y)",
            "Animate the limit of (sin x)/x as x approaches 0"
        ]
        
        cols = st.columns(3)
        for i, example in enumerate(example_prompts):
            with cols[i % 3]:
                if st.button(example, key=f"example_{i}"):
                    prompt = example
                    st.rerun()
    
    with col2:
        st.subheader("Generate")
        
        if st.button("Generate Manim Code", type="primary", use_container_width=True):
            if not prompt.strip():
                st.warning("Please enter a prompt.")
            elif not st.session_state.knowledge_base_setup:
                st.error("Knowledge base not setup. Please wait for setup to complete.")
            else:
                generate_code(rag_pipeline, prompt, max_tokens, temperature, top_p, include_code_examples)

def generate_code(rag_pipeline, prompt, max_tokens, temperature, top_p, include_code_examples):
    
    with st.spinner("Searching documentation and generating code..."):
        try:
            result = rag_pipeline.generate_manim_code(
                user_query=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            if result['success']:
                st.subheader("Generated Manim Code")
                st.markdown("<div class='code-container'>", unsafe_allow_html=True)
                st.code(result['code'], language="python")
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.download_button(
                    label="Download Code",
                    data=result['code'],
                    file_name=f"manim_animation_{hash(prompt) % 10000}.py",
                    mime="text/plain"
                )
                
                if result['retrieved_docs']:
                    display_retrieved_docs(result['retrieved_docs'])
                
                st.success("Code generation complete!")
                
                with st.expander("View Enhanced Prompt (Debug)"):
                    st.text_area("Enhanced Prompt", result['enhanced_prompt'], height=300)
                    
            else:
                st.error(f"Generation failed: {result['error']}")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
