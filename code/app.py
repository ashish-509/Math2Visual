import streamlit as st
import torch
import sys
import os
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'RAG')) 

st.set_page_config(page_title="Math2Visual", layout="centered")

st.title("Math2Visual — Manim Code Generator")
st.markdown("Enter a description and get **Manim-ready Python code**.")

with st.sidebar:
    st.header("Configuration")
    use_rag = st.checkbox("Enable RAG", value=False)
    
    if use_rag:
        st.info("RAG mode uses Manim documentation to enhance code generation")
        if st.button("Open Enhanced RAG App"):
            st.markdown("[Click here to use the enhanced RAG interface](../RAG/app_with_rag.py)")
    else:
        st.info("Standard mode uses only the fine-tuned model")

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    try:
        model_path = "./finetuned-codellama-manim"  # model path
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

pipe = load_model()

prompt = st.text_area("Prompt", placeholder="e.g. Draw a sine wave from -π to π...", height=150)

if st.button("Generate Manim Code"):
    if pipe is None:
        st.error("Model not loaded.")
    elif not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating code..."):
            if use_rag:
                try:
                    from rag_pipeline import Math2VisualRAG
                    
                    rag = Math2VisualRAG()
                    result = rag.generate_manim_code(prompt, max_new_tokens=300)
                    
                    if result['success']:
                        manim_code = result['code']
                        st.code(manim_code, language="python")
                        st.download_button("Save Code", manim_code, "generated_manim_code.py")
                        
                        if result['retrieved_docs']:
                            st.subheader("Retrieved Documentation")
                            for i, doc in enumerate(result['retrieved_docs']):
                                with st.expander(f"Reference {i+1}: {doc['metadata']['title']}"):
                                    st.write(doc['content'][:300] + "...")
                        
                        st.success("Done with RAG!")
                    else:
                        st.error(f"RAG generation failed: {result['error']}. Falling back to standard mode.")
                        use_rag = False
                        
                except Exception as e:
                    st.error(f"RAG error: {e}. Falling back to standard mode.")
                    use_rag = False
            
            if not use_rag:
                full_prompt = f"### Prompt:\n{prompt.strip()}\n### Manim Code:\n"
                try:
                    output = pipe(full_prompt, max_new_tokens=300, do_sample=True, top_p=0.9, temperature=0.7)
                    generated = output[0]["generated_text"]
                    manim_code = generated.split("### Manim Code:")[-1].strip()

                    st.code(manim_code, language="python")
                    st.download_button("Save Code", manim_code, "generated_manim_code.py")
                    st.success("Done!")
                except Exception as e:
                    st.error(f"Generation failed: {e}")
