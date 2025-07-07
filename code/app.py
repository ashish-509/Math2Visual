import streamlit as st
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline

st.set_page_config(page_title="Math2Visual", layout="centered")

st.title("🎬 Math2Visual — Manim Code Generator")
st.markdown("Enter a description and get **Manim-ready Python code** using your fine-tuned Code LLaMA model.")

# -- Load the model and tokenizer
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    try:
        model_path = "./finetuned-codellama-manim"  # Replace if needed
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

pipe = load_model()

# -- Prompt input
prompt = st.text_area("📝 Prompt", placeholder="e.g. Draw a sine wave from -π to π...", height=150)

# -- Generate button
if st.button("🚀 Generate Manim Code"):
    if pipe is None:
        st.error("❌ Model not loaded.")
    elif not prompt.strip():
        st.warning("⚠️ Please enter a prompt.")
    else:
        with st.spinner("Generating code..."):
            full_prompt = f"### Prompt:\n{prompt.strip()}\n### Manim Code:\n"
            try:
                output = pipe(full_prompt, max_new_tokens=300, do_sample=True, top_p=0.9, temperature=0.7)
                generated = output[0]["generated_text"]
                manim_code = generated.split("### Manim Code:")[-1].strip()

                st.code(manim_code, language="python")
                st.download_button("💾 Save Code", manim_code, "generated_manim_code.py")
                st.success("✅ Done!")
            except Exception as e:
                st.error(f"⚠️ Generation failed: {e}")
