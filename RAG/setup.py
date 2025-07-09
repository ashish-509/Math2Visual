"""
This script handles the complete setup of the RAG system including:
1. Installing dependencies
2. Processing Manim documentation
3. Creating vector embeddings
4. Testing the system
"""

import subprocess
import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_requirements():
    logger.info("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing requirements: {e}")
        return False
    return True

def setup_rag_system():
    logger.info("Setting up RAG system...")
    
    try:
        from document_processor import ManimeDocumentProcessor
        from vector_store import ManimeVectorStore
        from rag_pipeline import Math2VisualRAG
        
        logger.info("Step 1: Processing Manim documentation...")
        processor = ManimeDocumentProcessor()
        
        docs = processor.load_processed_docs()
        if not docs:
            docs = processor.process_documentation(max_docs=30)
            processor.save_processed_docs()
            logger.info(f"Processed {len(docs)} documents")
        else:
            logger.info(f"Loaded {len(docs)} existing documents")
        
        logger.info("Step 2: Creating vector embeddings...")
        vector_store = ManimeVectorStore()
        
        stats = vector_store.get_collection_stats()
        if stats['total_chunks'] == 0:
            vector_store.add_documents(docs)
            stats = vector_store.get_collection_stats()
            logger.info(f"Created {stats['total_chunks']} vector embeddings")
        else:
            logger.info(f"Vector store already contains {stats['total_chunks']} embeddings")
        
        logger.info("Step 3: Testing RAG pipeline...")
        rag = Math2VisualRAG(vector_store=vector_store)
        
        test_query = "create a simple sine wave animation"
        result = rag.generate_manim_code(test_query, max_new_tokens=200)
        
        if result['success']:
            logger.info("RAG pipeline test successful")
            logger.info(f"Generated code length: {len(result['code'])} characters")
            logger.info(f"Retrieved {len(result['retrieved_docs'])} relevant documents")
        else:
            logger.error(f"RAG pipeline test failed: {result['error']}")
            return False
        
        logger.info("RAG system setup complete!")
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please make sure all requirements are installed")
        return False
    except Exception as e:
        logger.error(f"Setup error: {e}")
        return False

def main():
    print("Math2Visual RAG Setup")
    print("=" * 50)
    
    rag_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(rag_dir)
    
    if not install_requirements():
        print("Setup failed: Could not install requirements")
        return
    
    if not setup_rag_system():
        print("Setup failed: Could not setup RAG system")
        return
    
    print("\n Setup completed successfully! \n")
    print("\nNext steps:")
    print("1. Make sure fine-tuned model is available at '../code/finetuned-codellama-manim'")
    print("2. Run the enhanced app: streamlit run app_with_rag.py")
    print("3. Or test the RAG pipeline directly: python rag_pipeline.py")

if __name__ == "__main__":
    main()
