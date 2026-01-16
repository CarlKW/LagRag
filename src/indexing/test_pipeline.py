import sys
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.loader import load_sfs_documents
from src.indexing.chunker import chunk_documents
from src.indexing.embedder import get_embedding_function


def print_chunk_info(chunks: List[Document], max_chunks: int = 10):
    print(f"\nTotal chunks created: {len(chunks)}")
    
    if not chunks:
        return
    
    para_counts = {}
    for chunk in chunks:
        para = chunk.metadata.get("paragraf", "unknown")
        para_counts[para] = para_counts.get(para, 0) + 1
    
    print("\nChunks per paragraph (top 10):")
    for para, count in sorted(para_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {para}: {count}")
    
    print(f"\nSample chunks (first {min(max_chunks, len(chunks))}):")
    for i, chunk in enumerate(chunks[:max_chunks], 1):
        print(f"\nChunk {i}:")
        print(f"  SFS: {chunk.metadata.get('sfs_nr', 'N/A')}")
        print(f"  Paragraph: {chunk.metadata.get('paragraf', 'N/A')}")
        if 'subchunk_index' in chunk.metadata:
            print(f"  Subchunk: {chunk.metadata['subchunk_index']}")
        print(f"  Preview: {chunk.page_content[:200]}...")


def test_query(vectorstore: Chroma, query: str, k: int = 5):
    print(f"\nQuery: {query}")
    print(f"Top {k} results:\n")
    
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    if not results:
        print("No results found.")
        return
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\nResult {i} (score: {score:.4f})")
        print(f"  SFS: {doc.metadata.get('sfs_nr', 'N/A')}")
        print(f"  Paragraph: {doc.metadata.get('paragraf', 'N/A')}")
        print(f"  Title: {doc.metadata.get('titel', 'N/A')[:80]}")
        print(f"  {doc.page_content}")


def run_pipeline_test(jsonl_path: str, persist_directory: str = "./chroma_db_test",
                      num_docs: int = 10, test_queries: List[str] = None):
    print("RAG PIPELINE TEST\n")
    
    print(f"Loading documents from {jsonl_path}...")
    all_documents = load_sfs_documents(jsonl_path)
    print(f"Loaded {len(all_documents)} documents")
    
    documents = all_documents[:num_docs] if num_docs else all_documents
    print(f"Processing {len(documents)} documents\n")
    
    if not documents:
        return None
    
    print(f"First document: {documents[0].metadata.get('sfs_nr')} - {documents[0].metadata.get('titel', '')[:60]}")
    
    print("\nChunking documents...")
    chunks = chunk_documents(
        documents,
        min_words=30,
        max_words=200,
        overlap_sentences=2,
        include_surrounding_paragraphs=True,
        short_document_threshold=100
    )
    print(f"Created {len(chunks)} chunks")
    
    print_chunk_info(chunks, max_chunks=5)
    
    print("\nCreating embeddings and vector store...")
    embeddings = get_embedding_function()
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="sfs_paragraphs",
        collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Vector store created with {vectorstore._collection.count()} documents")
    
    print("\nTesting queries...")
    if test_queries is None:
        test_queries = ["skatt", "bostad", "arbetsrätt", "miljöskydd", "förvaltning"]
    
    for query in test_queries:
        test_query(vectorstore, query, k=3)
    
    return vectorstore


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    jsonl_file = project_root / "data" / "sfs_lagboken_1990plus_filtered.jsonl"
    
    custom_queries = [
        "Narkotika får föras in till eller ut från landet",
        "Ett villkor för att introduktionsersättning",
        "begäran om att skicka ett viktigt meddelande till allmänheten"
    ]
    
    vectorstore = run_pipeline_test(
        jsonl_path=str(jsonl_file),
        persist_directory="./chroma_db_pipeline",
        num_docs=None,
        test_queries=custom_queries
    )
    
    if vectorstore:
        print("\nPipeline test completed successfully!")
