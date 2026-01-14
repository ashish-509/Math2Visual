# Test script for RAG pipeline
# Run this to verify chunking, retrieval, and context management work correctly

import sys
import os

# add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag.rag_pipeline import RAGPipeline

def test_rag():
    print(" RAG Pipeline Test \n")

    # find the markdown doc
    doc_path = os.path.join(
        os.path.dirname(__file__),
        '..', '..',
        'crawler',
        'markdown_output.md'
    )

    if not os.path.exists(doc_path):
        print(f"ERROR: Doc file not found at {doc_path}")
        print("Please run crawler/crawl.py first to generate the docs")
        return

    # create pipeline
    print("Creating RAG pipeline...")
    rag = RAGPipeline(
        chunk_size=800,
        chunk_overlap=150,
        max_context_tokens=2000,
        top_k_chunks=3
    )

    # load and index
    print(f"\nLoading and indexing {doc_path}...")
    success = rag.load_and_index(doc_path)

    if not success:
        print("Failed to load/index docs")
        return

    # show stats
    stats = rag.get_stats()
    print(f"\nRAG Statistics:")
    print(f"  Status: {stats['status']}")
    print(f"  Chunks: {stats['num_chunks']}")
    print(f"  Vocabulary size: {stats['vocab_size']}")
    print(f"  Top-K: {stats['top_k']}")
    print(f"  Max context tokens: {stats['max_context_tokens']}")

    # test queries
    test_queries = [
        "create a circle animation",
        "write text on screen",
        "plot a mathematical function",
        "transform one shape into another",
        "animate a graph"
    ]

    print("\n Testing Queries \n")

    for i, query in enumerate(test_queries, 1):
        print(f"[Query {i}] {query}")

        context = rag.retrieve_context(query)

        if context:
            lines = context.split('\n')
            preview_lines = lines[:15]  # show first 15 lines
            preview = '\n'.join(preview_lines)
            print(preview)

            if len(lines) > 15:
                print(f"\n... ({len(lines) - 15} more lines)")
        else:
            print("No context retrieved")

        print()

    # test full augmented prompt
    print("\n Full Augmented Prompt Example \n")
    test_query = "animate a red circle growing from center"
    base_prompt = "Generate Manim code for the following request:"

    augmented = rag.augment_prompt(test_query, base_prompt)

    print(augmented[:1000])  # show first 1000 chars
    if len(augmented) > 1000:
        print(f"\n... (total {len(augmented)} characters)")

    print("\n Test Complete ")

if __name__ == "__main__":
    test_rag()