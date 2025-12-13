"""
Quick interactive test script for testing queries against the vector store.
Run this after running test_pipeline.py to test queries interactively.
"""
import sys
from pathlib import Path

# Add project root to path before importing src modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from pathlib import Path

from langchain_community.vectorstores import Chroma

from src.indexing.embedder import get_embedding_function


def interactive_query_test(persist_directory: str = "./chroma_db_test"):
    """
    Interactive query testing interface.
    
    Args:
        persist_directory: Directory where ChromaDB is stored
    """
    print(f"{'='*80}")
    print("INTERACTIVE QUERY TEST")
    print(f"{'='*80}")
    print(f"Loading vector store from {persist_directory}...")
    
    # Load embeddings and vector store
    embeddings = get_embedding_function()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    print("Vector store loaded!")
    print("\nEnter queries to test (type 'quit' or 'exit' to stop):\n")
    
    while True:
        query = input("Query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        try:
            # Get number of results
            k_input = input("Number of results (default 5): ").strip()
            k = int(k_input) if k_input else 5
            
            # Perform search
            results = vectorstore.similarity_search_with_score(query, k=k)
            
            print(f"\n{'='*80}")
            print(f"Found {len(results)} results for: \"{query}\"")
            print(f"{'='*80}\n")
            
            for i, (doc, score) in enumerate(results, 1):
                print(f"Result {i} (Score: {score:.4f})")
                print(f"  SFS: {doc.metadata.get('sfs_nr', 'N/A')}")
                print(f"  Type: {doc.metadata.get('typ', 'N/A')}")
                print(f"  Paragraph: {doc.metadata.get('paragraf', 'N/A')}")
                print(f"  Title: {doc.metadata.get('titel', 'N/A')[:60]}...")
                print(f"  Content: {doc.page_content[:150]}...")
                print()
            
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    persist_dir = project_root / "chroma_db_test"
    
    interactive_query_test(str(persist_dir))

