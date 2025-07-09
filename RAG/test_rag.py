import sys
import os

sys.path.append(os.path.dirname(__file__)) # Add current directory to path

def test_document_processor():
    print("Testing Document Processor...")
    try:
        from document_processor import ManimeDocumentProcessor
        
        processor = ManimeDocumentProcessor()
        
        links = processor.get_manim_documentation_links()
        print(f"Found {len(links)} documentation links")
        
        if links:
            doc = processor.extract_content_from_url(links[0])
            if doc:
                print(f"Successfully extracted content from: {doc['title']}")
                print(f"Content length: {doc['length']} characters")
            else:
                print("Failed to extract content")
        
        return True
        
    except Exception as e:
        print(f"Document processor test failed: {e}")
        return False

def test_vector_store():
    print("\nTesting Vector Store...")
    try:
        from vector_store import ManimeVectorStore
        
        test_docs = [
            {
                'url': 'https://test.com/doc1',
                'title': 'Test Animation Tutorial',
                'content': 'This tutorial shows how to create animations using Manim. You can create scenes with various objects like circles, squares, and text. The Scene class is the base for all animations.',
                'code_examples': ['from manim import *\n\nclass TestScene(Scene):\n    def construct(self):\n        circle = Circle()\n        self.play(Create(circle))'],
                'length': 200
            },
            {
                'url': 'https://test.com/doc2', 
                'title': 'Mathematical Plotting',
                'content': 'Learn to plot mathematical functions in Manim. You can use FunctionGraph to plot any mathematical function. Common functions include sine, cosine, and polynomial functions.',
                'code_examples': ['axes = Axes()\nfunc = axes.plot(lambda x: np.sin(x))'],
                'length': 180
            }
        ]
        
        vector_store = ManimeVectorStore()
        
        vector_store.add_documents(test_docs)
        print("Successfully added test documents to vector store")
        
        results = vector_store.search("how to create circle animation", n_results=2)
        print(f"Search returned {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"Result {i+1}: {result['metadata']['title']}")
        
        stats = vector_store.get_collection_stats()
        print(f"Vector store contains {stats['total_chunks']} chunks")
        
        return True
        
    except Exception as e:
        print(f"Vector store test failed: {e}")
        return False

def test_without_model():
    print("\nTesting RAG Pipeline (without model)...")
    try:
        from rag_pipeline import Math2VisualRAG

        rag = Math2VisualRAG()
        
        query = "create sine wave animation"
        docs = rag.retrieve_relevant_docs(query, n_results=3)
        print(f"Retrieved {len(docs)} relevant documents for query: '{query}'")
        
        for i, doc in enumerate(docs):
            print(f"   Doc {i+1}: {doc['metadata']['title']}")
        
        enhanced_prompt = rag.create_enhanced_prompt(query, docs)
        print(f"Created enhanced prompt ({len(enhanced_prompt)} characters)")
        
        return True
        
    except Exception as e:
        print(f"RAG pipeline test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Math2Visual RAG System Tests")
    print("=" * 50)
    
    tests = [
        test_document_processor,
        test_vector_store,
        test_without_model
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"Test failed with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! RAG system is working correctly.")
        print("\nNext steps:")
        print("1. Run: python setup.py (to setup full system)")
        print("2. Run: streamlit run app_with_rag.py (to use enhanced app)")
    else:
        print("Some tests failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Check if ChromaDB is properly installed")

if __name__ == "__main__":
    main()
