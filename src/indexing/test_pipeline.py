"""
Test script for the complete RAG pipeline.
Tests document loading, chunking, embedding, and retrieval.
"""
import sys
from pathlib import Path
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.loader import load_sfs_documents
from src.indexing.chunker import chunk_documents
from src.indexing.embedder import get_embedding_function


def print_chunk_info(chunks: List[Document], max_chunks: int = 10):
    """
    Print information about the created chunks.
    
    Args:
        chunks: List of chunk Documents
        max_chunks: Maximum number of chunks to print details for
    """
    print(f"\n{'='*80}")
    print(f"CHUNK SUMMARY")
    print(f"{'='*80}")
    print(f"Total chunks created: {len(chunks)}")
    
    if not chunks:
        print("No chunks to display.")
        return
    
    # Count chunks by paragraph
    para_counts = {}
    for chunk in chunks:
        para = chunk.metadata.get("paragraf", "unknown")
        para_counts[para] = para_counts.get(para, 0) + 1
    
    print(f"\nChunks per paragraph (top 10):")
    for para, count in sorted(para_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {para}: {count} chunk(s)")
    
    # Show sample chunks
    print(f"\n{'='*80}")
    print(f"SAMPLE CHUNKS (first {min(max_chunks, len(chunks))} chunks)")
    print(f"{'='*80}")
    
    for i, chunk in enumerate(chunks[:max_chunks], 1):
        print(f"\n--- Chunk {i} ---")
        print(f"SFS nr: {chunk.metadata.get('sfs_nr', 'N/A')}")
        print(f"Type: {chunk.metadata.get('typ', 'N/A')}")
        print(f"Paragraph: {chunk.metadata.get('paragraf', 'N/A')}")
        if 'subchunk_index' in chunk.metadata:
            print(f"Subchunk index: {chunk.metadata['subchunk_index']}")
        if 'merged_paragraphs' in chunk.metadata:
            print(f"Merged paragraphs: {chunk.metadata['merged_paragraphs']}")
        if 'is_short_document' in chunk.metadata and chunk.metadata['is_short_document']:
            print(f"Short document: Yes (total words: {chunk.metadata.get('total_words', 'N/A')})")
        if 'has_surrounding_context' in chunk.metadata and chunk.metadata.get('has_surrounding_context'):
            print(f"Has surrounding context: Yes")
            if 'context_paragraphs' in chunk.metadata and chunk.metadata['context_paragraphs']:
                print(f"Context paragraphs: {chunk.metadata['context_paragraphs']}")
        print(f"Word count: ~{len(chunk.page_content.split())} words")
        print(f"Content preview:")
        print(f"  {chunk.page_content[:200]}...")
        if len(chunk.page_content) > 200:
            print(f"  ... ({len(chunk.page_content) - 200} more characters)")


def test_query(
    vectorstore: Chroma,
    query: str,
    k: int = 5,
    show_scores: bool = True
):
    """
    Test a query against the vector store and display results.
    
    Args:
        vectorstore: ChromaDB vector store
        query: Query string to search for
        k: Number of results to return
        show_scores: Whether to show similarity scores
    """
    print(f"\n{'='*80}")
    print(f"QUERY TEST")
    print(f"{'='*80}")
    print(f"Query: \"{query}\"")
    print(f"Retrieving top {k} results...\n")
    
    # Perform similarity search
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    if not results:
        print("No results found.")
        return
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"{'='*80}")
        print(f"Result {i} (Score: {score:.4f})" if show_scores else f"Result {i}")
        print(f"{'='*80}")
        print(f"SFS nr: {doc.metadata.get('sfs_nr', 'N/A')}")
        print(f"Type: {doc.metadata.get('typ', 'N/A')}")
        print(f"Title: {doc.metadata.get('titel', 'N/A')[:100]}...")
        print(f"Paragraph: {doc.metadata.get('paragraf', 'N/A')}")
        if 'subchunk_index' in doc.metadata:
            print(f"Subchunk index: {doc.metadata['subchunk_index']}")
        if 'merged_paragraphs' in doc.metadata:
            print(f"Merged paragraphs: {doc.metadata['merged_paragraphs']}")
        if 'is_short_document' in doc.metadata and doc.metadata['is_short_document']:
            print(f"Short document: Yes (total words: {doc.metadata.get('total_words', 'N/A')})")
        if 'has_surrounding_context' in doc.metadata and doc.metadata.get('has_surrounding_context'):
            print(f"Has surrounding context: Yes")
            if 'context_paragraphs' in doc.metadata and doc.metadata['context_paragraphs']:
                print(f"Context paragraphs: {doc.metadata['context_paragraphs']}")
        if 'organ' in doc.metadata:
            print(f"Organ: {doc.metadata['organ']}")
        print(f"\nContent:")
        print(f"{doc.page_content}")
        print()


def run_pipeline_test(
    jsonl_path: str,
    persist_directory: str = "./chroma_db_test",
    num_docs: int = 10,
    test_queries: List[str] = None
):
    """
    Run the complete pipeline test.
    
    Args:
        jsonl_path: Path to filtered JSONL file
        persist_directory: Directory to store ChromaDB
        num_docs: Number of documents to process (for faster testing)
        test_queries: List of test queries to run
    """
    print(f"{'='*80}")
    print(f"RAG PIPELINE TEST")
    print(f"{'='*80}")
    
    # Step 1: Load documents
    print(f"\n[1/4] Loading documents from {jsonl_path}...")
    all_documents = load_sfs_documents(jsonl_path)
    print(f"Loaded {len(all_documents)} total documents")
    
    # Use subset for testing
    documents = all_documents[:num_docs] if num_docs else all_documents
    print(f"Processing {len(documents)} documents for testing")
    
    if not documents:
        print("No documents to process!")
        return None
    
    # Show first document info
    print(f"\nFirst document:")
    print(f"  SFS nr: {documents[0].metadata.get('sfs_nr', 'N/A')}")
    print(f"  Type: {documents[0].metadata.get('typ', 'N/A')}")
    print(f"  Title: {documents[0].metadata.get('titel', 'N/A')[:80]}...")
    print(f"  Content length: {len(documents[0].page_content)} characters")
    
    # Step 2: Chunk documents
    print(f"\n[2/4] Chunking documents...")
    chunks = chunk_documents(
        documents,
        min_words=100,
        max_words=1200,
        overlap_sentences=5,
        include_surrounding_paragraphs=True,
        short_document_threshold=300
    )
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Show chunk information
    print_chunk_info(chunks, max_chunks=5)
    
    # Step 3: Create embeddings and vector store
    print(f"\n[3/4] Creating embeddings and vector store...")
    print("Loading embedding model (this may take a moment on first run)...")
    embeddings = get_embedding_function()
    print("Embedding model loaded.")
    
    print(f"Creating vector store in {persist_directory}...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("Vector store created successfully!")
    
    # Step 4: Test queries
    print(f"\n[4/4] Testing queries...")
    
    if test_queries is None:
        # Default test queries
        test_queries = [
            "skatt",
            "bostad",
            "arbetsrätt",
            "miljöskydd",
            "förvaltning"
        ]
    
    for query in test_queries:
        test_query(vectorstore, query, k=3)
    
    return vectorstore


if __name__ == "__main__":
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    jsonl_file = project_root / "data" / "sfs_lagboken_1990plus_filtered.jsonl"
    
    # Custom test queries (optional - will use defaults if None)
    custom_queries = [
        "När får bidraget betalas ut för investeringsbidrag?",
        "skatt för singapor",
    ]
    
    # Run the test pipeline
    vectorstore = run_pipeline_test(
        jsonl_path=str(jsonl_file),
        persist_directory="./chroma_db_test",
        num_docs=1000,  # Change to None to process all documents
        test_queries=custom_queries
    )
    
    if vectorstore:
        print(f"\n{'='*80}")
        print("Pipeline test completed successfully!")
        print(f"Vector store saved to: ./chroma_db_test")
        print(f"{'='*80}")

