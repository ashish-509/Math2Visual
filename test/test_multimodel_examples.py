import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pipeline.math2visual_pipeline import Math2VisualPipeline

def example_basic_usage():
    """Basic usage with default model (Mistral)"""
    print(" Example 1: Basic Usage \n")
    
    # create pipeline with default settings
    pipeline = Math2VisualPipeline(use_rag=True, preferred_model='mistral')
    
    # check which models are available
    print("Model Status:")
    status = pipeline.get_model_status()
    for name, info in status.items():
        health = "healthy" if info['healthy'] else "down"
        pref = " (preferred)" if info['preferred'] else ""
        print(f"  {name}: {health}{pref}")
    
    print("\nGenerating code with automatic model fallback...")
    code, success, error = pipeline.generate_code("draw a blue circle")
    
    if success:
        print(f" Generated {len(code)} characters of code")
    else:
        print(f" Failed: {error}")

def example_model_switching():
    """How to switch between models dynamically"""
    print("\n Example 2: Switching Models \n")
    
    pipeline = Math2VisualPipeline(preferred_model='mistral')
    
    # try with Mistral first
    print("Using Mistral...")
    code1, success1, _ = pipeline.generate_code("create a red square")
    
    # switch to CodeLlama
    print("\nSwitching to CodeLlama...")
    pipeline.switch_model('codellama')
    code2, success2, _ = pipeline.generate_code("create a green triangle")
    
    # switch to Phi-2
    print("\nSwitching to Phi-2...")
    pipeline.switch_model('phi2')
    code3, success3, _ = pipeline.generate_code("animate text")
    
    print(f"\nResults:")
    print(f"  Mistral: {'' if success1 else ''}")
    print(f"  CodeLlama: {'' if success2 else ''}")
    print(f"  Phi-2: {'' if success3 else ''}")

def example_custom_endpoints():
    """Using custom model endpoints"""
    print("\n Example 3: Custom Endpoints \n")
    
    # we can override endpoints via environment variables
    # or pass a custom client with your own config
    
    from llm.client import LLMClient
    
    custom_models = {
        'mistral': 'http://my-server.com:8001/generate',
        'codellama': 'http://my-server.com:8002/generate',
        'phi2': 'http://localhost:9000/generate'
    }
    
    client = LLMClient(preferred_model='mistral', models_config=custom_models)
    pipeline = Math2VisualPipeline(llm_client=client)
    
    print("Using custom endpoints:")
    for name, endpoint in client.models.items():
        print(f"  {name}: {endpoint}")

def example_rag_with_multimodel():
    """Combining RAG with multi-model support"""
    print("\n Example 4: RAG + Multi-Model \n")
    
    # RAG will retrieve relevant Manim docs
    # Multi-model will try different LLMs if one fails
    pipeline = Math2VisualPipeline(
        use_rag=True,
        preferred_model='mistral'
    )
    
    # check RAG status
    rag_stats = pipeline.get_rag_stats()
    print(f"RAG Status: {rag_stats.get('status', 'unknown')}")
    
    if rag_stats.get('status') == 'ready':
        print(f"  Indexed chunks: {rag_stats['num_chunks']}")
    
    # generate with RAG-enhanced prompts and model fallback
    print("\nGenerating with RAG + multi-model...")
    code, success, error = pipeline.generate_code("animate a circle growing")
    
    if success:
        print(f" Success! Generated {len(code)} chars")
    else:
        print(f" All models failed: {error}")

if __name__ == "__main__":
    print("Math2Visual Multi-Model Examples")
    print("\nNote: These examples require LLM servers running")
    print("Start at least one model server to see full functionality\n")
    
    # run examples
    try:
        example_basic_usage()
        example_model_switching()
        example_custom_endpoints()
        example_rag_with_multimodel()
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("This is expected if no model servers are running")
    
    print("Examples complete!")
