# This module loads the finetuned Mistral model and generates Manim code
# It uses the LoRA adapter from finetuning/Checkpoint-500

import os
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinetunedMistralClient:
    """
    Loads the finetuned Mistral model with LoRA adapter for Manim code generation.
    This model was trained specifically on Manim documentation and code examples.
    """
    
    def __init__(self, checkpoint_path=None):
        """
        Initialize the finetuned model.
        
        checkpoint_path: Path to the LoRA adapter checkpoint folder.
                        If None, uses the default path in finetuning/Checkpoint-500
        """
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        
        # Set default checkpoint path
        if checkpoint_path is None:
            # Go from src/llm to project root, then to finetuning/Checkpoint-500
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            checkpoint_path = os.path.join(project_root, "finetuning", "Checkpoint-500")
        
        self.checkpoint_path = checkpoint_path
        logger.info(f"FinetunedMistralClient initialized. Checkpoint: {checkpoint_path}")
        logger.info(f"Device: {self.device}")
    
    def load_model(self):
        """
        Load the base Mistral model and apply the LoRA adapter.
        This is called lazily when first generating code to save memory.
        """
        if self.is_loaded:
            logger.info("Model already loaded")
            return True
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            logger.info("Loading base Mistral-7B model...")
            
            # The base model name is in the adapter config
            base_model_name = "mistralai/Mistral-7B-v0.1"
            
            # Load tokenizer from checkpoint (has any special tokens)
            logger.info(f"Loading tokenizer from {self.checkpoint_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.checkpoint_path,
                trust_remote_code=True
            )
            
            # Make sure padding token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model with memory optimization
            logger.info(f"Loading base model: {base_model_name}")
            
            # Use 4-bit quantization to save GPU memory
            if self.device == "cuda":
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # CPU mode - load in float16 (slower but uses less memory)
                logger.warning("Loading on CPU - this will be slow!")
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            
            # Apply the LoRA adapter
            logger.info(f"Applying LoRA adapter from {self.checkpoint_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.checkpoint_path
            )
            
            # Set to evaluation mode
            self.model.eval()
            self.is_loaded = True
            
            logger.info("Finetuned Mistral model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load finetuned model: {e}")
            logger.error("Make sure you have: transformers, peft, bitsandbytes installed")
            return False
    
    def generate(self, prompt, max_new_tokens=1024, temperature=0.7, top_p=0.9):
        """
        Generate Manim code from the given prompt.
        
        prompt: The augmented prompt (with RAG context)
        max_new_tokens: Maximum tokens to generate
        temperature: Controls randomness (lower = more focused)
        top_p: Nucleus sampling parameter
        
        Returns: Generated Manim code as a string
        """
        # Load model if not already loaded
        if not self.is_loaded:
            success = self.load_model()
            if not success:
                return "# Error: Could not load the finetuned model"
        
        try:
            # Format the prompt for code generation
            formatted_prompt = self._format_prompt(prompt)
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048  # Leave room for generation
            )
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with the model
            logger.info("Generating Manim code...")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the output
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Extract just the generated code (remove the prompt)
            code = self._extract_code(generated_text, formatted_prompt)
            
            logger.info("Code generation complete")
            return code
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return f"# Error generating code: {str(e)}"
    
    def _format_prompt(self, prompt):
        """
        Format the prompt for the finetuned model.
        Uses a clear structure that the model was trained on.
        """
        system_message = """You are a Manim code expert. Generate clean, working Manim code using modern Manim Community Edition API.

IMPORTANT - Use these CORRECT modern Manim methods:
- Use Create() instead of ShowCreation() 
- Use Uncreate() instead of ShowDestruction()
- Use FadeIn() and FadeOut() for fading
- Use Transform() for morphing between objects
- Use Write() for text and equations
- Use DrawBorderThenFill() for shapes
- Use self.play() to animate
- Use self.wait() to pause

For math equations:
- Use MathTex(r"...") for ANY formula with math symbols (=, +, -, ×, fractions, greek letters)
- Use Text("...") for plain text labels (no LaTeX, ASCII only)
- NEVER use Tex() for formulas - always use MathTex()
- For multiplication: MathTex(r"a \\times b")
- Escape backslashes: \\times, \\frac, \\pi, \\sqrt

CRITICAL LAYOUT RULES - ZERO OVERLAP ALLOWED:
1. SEQUENCE IS CRITICAL - NEVER show new content while old is in center:
   a) FIRST: Move old content away with self.play(old.animate.scale(0.25).to_corner(UL))
   b) THEN: Show new content in center
   c) NEVER animate both in same self.play()

2. OLD CONTENT POSITIONS:
   - 1st old: to_corner(UL) with buff=0.3
   - 2nd old: move_to(LEFT*6)
   - 3rd old: to_corner(DL) with buff=0.3
   - More than 3: FadeOut oldest

3. CENTER ZONE: New content at ORIGIN, scale 0.7

4. PATTERN:
   self.play(old.animate.scale(0.25).to_corner(UL))  # Move old away FIRST
   new = MathTex(r"...").scale(0.7)  # Create new
   self.play(Write(new))  # Show new in center
   self.wait(1)

5. Use VGroup() to move related items together
6. Title at top edge, scale 0.6, never move
7. Max 30 chars per line

Rules:
1. Use standard ASCII characters only (no unicode symbols like pi)
2. Write clear comments explaining each step
3. Use descriptive variable names
4. Always start with: from manim import *
5. Create a Scene class that inherits from Scene
6. Implement the construct method properly
7. Never use deprecated methods like ShowCreation
8. Total animation should be 15-30 seconds
9. NEVER use self.play(self.add(...)) - this is invalid
10. self.add(obj) = instant appearance (no animation)
11. self.play(Write(obj)) or self.play(FadeIn(obj)) = animated appearance
12. self.play() only takes Animation objects: Create(), Write(), FadeIn(), Transform(), etc."""
        
        formatted = f"""### System:
{system_message}

### User Request:
{prompt}

### Generated Manim Code:
```python
from manim import *

"""
        return formatted
    
    def _extract_code(self, full_text, prompt):
        """
        Extract just the generated code from the full model output.
        """
        import re
        
        # Try to find the code after our prompt marker
        marker = "### Generated Manim Code:"
        
        if marker in full_text:
            code_part = full_text.split(marker)[-1]
        else:
            # Fallback: remove the input prompt
            code_part = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
        
        # Clean up the code
        code_part = code_part.strip()
        
        # Remove markdown code fences if present
        if code_part.startswith("```python"):
            code_part = code_part[9:]
        if code_part.startswith("```"):
            code_part = code_part[3:]
        if code_part.endswith("```"):
            code_part = code_part[:-3]
        
        # Make sure we have the import statement
        if not code_part.strip().startswith("from manim import"):
            code_part = "from manim import *\n\n" + code_part
        
        # Fix deprecated Manim API calls
        code_part = self._fix_deprecated_manim_calls(code_part)
        
        return code_part.strip()
    
    def _fix_deprecated_manim_calls(self, code):
        import re
        
        # Dictionary of deprecated -> modern replacements
        replacements = {
            r'\bShowCreation\b': 'Create',
            r'\bShowDestruction\b': 'Uncreate',
            r'\bWiggleOutThenIn\b': 'Wiggle',
        }
        
        for old_pattern, new_name in replacements.items():
            code = re.sub(old_pattern, new_name, code)
        
        # Fix unicode characters in Text() calls
        code = code.replace('Text("π', 'Text("pi')
        code = code.replace("Text('π", "Text('pi")
        code = code.replace('"\u03c0"', '"pi"')
        code = code.replace("'\u03c0'", "'pi'")
        
        return code
    
    def get_status(self):
        """
        Get the current status of the model.
        """
        return {
            "loaded": self.is_loaded,
            "device": self.device,
            "checkpoint": self.checkpoint_path,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }


# Singleton instance for caching
_cached_client = None


def get_finetuned_client():
    """
    Get a cached instance of the finetuned client.
    This avoids reloading the model multiple times.
    """
    global _cached_client
    if _cached_client is None:
        _cached_client = FinetunedMistralClient()
    return _cached_client
