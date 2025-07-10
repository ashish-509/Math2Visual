import streamlit as st
import time
import json
from datetime import datetime

st.set_page_config(
    page_title="Math2Visual",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Hide Streamlit branding and menu */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* Dark theme color scheme */
    .stApp {
        background: linear-gradient(135deg, #181c24 0%, #232946 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #f4f4f4;
    }
    .main-container {
        background: #232946;
        border-radius: 18px;
        padding: 2.5rem 2rem 2rem 2rem;
        margin: 2.5rem auto 2rem auto;
        max-width: 900px;
        box-shadow: 0 8px 32px rgba(20, 20, 40, 0.25);
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #eebbc3;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #b8c1ec;
        text-align: center;
        margin-bottom: 2.2rem;
        font-weight: 400;
    }
    .input-section {
        background: #232946;
        padding: 1.5rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #393e5c;
    }
    .stButton > button {
        background: linear-gradient(90deg, #eebbc3 0%, #b8c1ec 100%);
        color: #232946;
        border: none;
        padding: 0.9rem 2.2rem;
        border-radius: 22px;
        font-weight: 700;
        font-size: 1.08rem;
        transition: all 0.2s;
        box-shadow: 0 2px 8px rgba(238,187,195,0.10);
        width: auto;
        min-width: 180px;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #b8c1ec 0%, #eebbc3 100%);
        color: #232946;
        box-shadow: 0 4px 16px rgba(238,187,195,0.18);
    }
    .code-section {
        background: #121629;
        border-radius: 12px;
        padding: 1.5rem 1.2rem;
        margin: 2rem 0 1.5rem 0;
        border: 1px solid #393e5c;
        color: #f4f4f4;
    }
    .action-buttons {
        background: #232946;
        padding: 1.2rem 1rem;
        border-radius: 12px;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 1px 4px rgba(44, 62, 80, 0.10);
    }
    .video-section {
        background: #181c24;
        padding: 1.5rem 1.2rem;
        border-radius: 12px;
        margin: 2rem 0 1.5rem 0;
        border: 1px solid #393e5c;
        color: #f4f4f4;
    }
    .status-success {
        background: #393e5c;
        border: 1px solid #eebbc3;
        padding: 1.2rem 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #eebbc3;
        font-size: 1.08rem;
    }
    /* Remove Streamlit textarea label */
    .stTextArea label {
        display: none;
    }
    
    /* Custom textarea styling */
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 1.5px solid #b8c1ec;
        font-size: 1.05rem;
        padding: 1rem;
        background: #121629;
        color: #f4f4f4;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #eebbc3;
        box-shadow: 0 0 0 2px #b8c1ec;
    }
    .stCodeBlock, .stCodeBlock pre, .stCodeBlock code {
        background: #121629 !important;
        color: #f4f4f4 !important;
        font-size: 1.05rem;
    }
    
    /* Placeholder styles */
    .placeholder-large {
        background: #121629;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border: 1px solid #393e5c;
        text-align: center;
    }
    .video-play {
        display: inline-block;
        width: 64px;
        height: 64px;
        margin: 0 auto;
    }
    .video-play svg {
        width: 100%;
        height: 100%;
    }
    .video-actions {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1rem;
    }
    .download-btn, .export-btn {
        background: #eebbc3;
        color: #232946;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 22px;
        font-weight: 700;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .download-btn:hover, .export-btn:hover {
        background: #b8c1ec;
        color: #232946;
    }
    .centered-btn {
        text-align: center;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

if 'generated_code' not in st.session_state:
    st.session_state.generated_code = ""
if 'generation_history' not in st.session_state:
    st.session_state.generation_history = []

def main():
    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.markdown('<div class="main-header">Math<span>2</span>VISUAL</div>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">From Natural Language Prompts to Animated Math Videos</p>', unsafe_allow_html=True)
        st.markdown('<div style="height:2rem;"></div>', unsafe_allow_html=True)
        st.markdown("### ⚙️ Model Settings")
        model_option = st.selectbox(
            "Select AI Model",
            ["CodeLlama-7B (Fine-tuned)", "CodeLlama-13B", "Mistral-7B", "Phi-2"],
            help="Choose the AI model for Manim code generation"
        )
        st.markdown("### 🎥 Video Settings")
        video_quality = st.select_slider(
            "Video Quality",
            options=["Low", "Medium", "High", "4K Ultra HD"],
            value="Medium"
        )
        video_duration = st.slider(
            "Max Duration (seconds)",
            min_value=5,
            max_value=300,
            value=60,
            step=10
        )
        animation_style = st.selectbox(
            "Animation Style",
            ["Smooth", "Fast", "Detailed", "Minimalist"]
        )
        st.markdown("### 📋 Recent Generations")
        if st.session_state.generation_history:
            for i, item in enumerate(st.session_state.generation_history[-3:]):
                with st.expander(f"Generation {len(st.session_state.generation_history) - i}"):
                    st.write(f"**Prompt:** {item['prompt'][:40]}...")
                    st.write(f"**Time:** {item['timestamp']}")
        else:
            st.info("No recent generations")

    with right_col:
        st.markdown('### Enter your Prompt here : ')
        custom_prompt = "Visualize the Pythagorean theorem with an animated right triangle and squares on each side."
        user_prompt = st.text_area(
            label="Enter your prompt here",
            value=custom_prompt,
            height=80,
            placeholder="Enter your prompt here",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="centered-btn">', unsafe_allow_html=True)
        generate_code = st.button("GenerateCode")

        manim_code = '''from manim import *

class PythagorasTheorem(Scene):
    def construct(self):
        # Create the right triangle
        triangle = Polygon(
            [0, 0, 0], [3, 0, 0], [0, 4, 0], color=BLUE
        )
        triangle_label = MathTex(r"a", color=WHITE).next_to([1.5, 0, 0], DOWN)
        triangle_label2 = MathTex(r"b", color=WHITE).next_to([0, 2, 0], LEFT)
        triangle_label3 = MathTex(r"c", color=WHITE).next_to([1.5, 2, 0], RIGHT)
        
        # Squares on each side
        square_a = Square(3, color=GREEN).move_to([1.5, -1.5, 0])
        square_b = Square(4, color=YELLOW).move_to([-2, 2, 0])
        square_c = Square(5, color=RED).move_to([3, 4, 0])
        
        # Animate triangle
        self.play(Create(triangle))
        self.play(Write(triangle_label), Write(triangle_label2), Write(triangle_label3))
        self.wait(1)
        
        # Animate squares
        self.play(Create(square_a), Create(square_b))
        self.wait(1)
        self.play(Create(square_c))
        self.wait(2)
        
        # Show the relationship
        theorem = MathTex(r"a^2 + b^2 = c^2", font_size=48).to_edge(UP)
        self.play(Write(theorem))
        self.wait(2)
'''

        st.markdown('<div class="placeholder-actions">'
                    '<span id="copy-btn" style="cursor:pointer;">🗐 copy</span> &nbsp;&nbsp; '
                    '<span>⬈ Export</span>'
                    '</div>', unsafe_allow_html=True)
        st.code(manim_code, language="python")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('''
        <script>
        function copyCode() {
            var codeBlocks = document.querySelectorAll('pre');
            if (codeBlocks.length > 0) {
                var code = codeBlocks[codeBlocks.length-1].innerText;
                navigator.clipboard.writeText(code);
            }
        }
        document.addEventListener('DOMContentLoaded', function() {
            var btn = document.getElementById('copy-btn');
            if (btn) btn.onclick = copyCode;
        });
        </script>
        ''', unsafe_allow_html=True)

        st.markdown('<div class="centered-btn">', unsafe_allow_html=True)
        st.button("GenerateVideo")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="placeholder-large">', unsafe_allow_html=True)
        st.markdown('<div class="video-play"><svg viewBox="0 0 64 64"><circle cx="32" cy="32" r="30" stroke="#222" stroke-width="4" fill="none"/><polygon points="26,20 48,32 26,44" fill="#222"/></svg></div>', unsafe_allow_html=True)
        st.markdown('<div style="text-align:center; color:#444; font-size:1.1rem; margin-top:1rem;">Generated video will appear here</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="video-actions">', unsafe_allow_html=True)
        st.markdown('<button class="download-btn">&#8681; download</button>', unsafe_allow_html=True)
        st.markdown('<button class="export-btn">&#8680; Export</button>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        

if __name__ == "__main__":
    main()