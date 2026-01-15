# Test the multi-model system with health checks and automatic fallback
# This shows how the system switches between different AI models when one fails

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm.client import LLMClient, LLMClientError

def test_multi_model_system():
    print("Testing Multi-Model AI System")

    # Create a client that prefers Mistral model
    print("Setting up AI client with Mistral as the first choice...")
    client = LLMClient(preferred_model='mistral')

    print("\nModels that are configured:")
    for name, address in client.model_endpoints.items():
        print(f"  {name}: {address}")

    # Check which models are working
    print("\nChecking if each model is responding:")
    status = client.get_model_status()

    for model_name, info in status.items():
        health_icon = "✓ Working" if info['healthy'] else "✗ Not working"
        preferred_mark = " (first choice)" if info['preferred'] else ""
        print(f"  {health_icon} {model_name}: {info['address']}{preferred_mark}")

    # Show the order in which models will be tried
    print("\nOrder of models to try:")
    fallback_order = client.get_fallback_order()
    print(f"  Will try: {' -> '.join(fallback_order)}")

    # Try to generate some code (this will fail if no servers are running)
    print("\nTesting code generation:")
    print("Trying to create a simple Manim animation...")
    print("(Note: This needs at least one AI server running)")

    test_request = "Create a Manim scene with a blue square"

    try:
        result = client.generate(test_request, timeout_seconds=10)
        print(f"\nSuccess! Generated {len(result)} characters of code")
        print("\nCode preview:")
        print(result[:200] + "..." if len(result) > 200 else result)
    except LLMClientError as e:
        print(f"\nFailed as expected (no servers running): {e}")
        print("To test fully, start an AI server on one of the ports above.")

    print("\nTest finished!")

if __name__ == "__main__":
    test_multi_model_system()
