"""
Main RAG pipeline orchestration script.

This script integrates all components:
1. Chunking (already done, loads from ChromaDB)
2. Embedding (already done, loads from ChromaDB)
3. Retrieval (RerankingRetriever or BasicRetriever)
4. Generation (RAGGenerator with active retrieval)
"""
import sys
import argparse
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.generator.retriever import RerankingRetriever
from src.generator.retriever_basic import BasicRetriever
from src.generation.lm_wrapper import LocalHFModel, get_local_lm
from src.generation.genAI import RAGGenerator, ContextChunk
from src.generation.adapters import retriever_results_to_context_chunks


def initialize_pipeline(
    chroma_db_path: str = "./chroma_db_test",
    retriever_type: str = "reranking",
    collection_name: str = "sfs_paragraphs",
    k_initial: int = 50,
    k_final: int = 10,
    reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual",
    lm_model_path: str = "gpt2",
    max_retrieval_rounds: int = 2,
    high_threshold: float = 0.75,
    low_threshold: float = 0.40,
):
    """
    Initialize the complete RAG pipeline.
    
    Args:
        chroma_db_path: Path to ChromaDB directory
        retriever_type: "reranking" or "basic"
        collection_name: ChromaDB collection name
        k_initial: Initial retrieval count for reranking retriever
        k_final: Final retrieval count after reranking
        reranker_model: Model name for reranker
        lm_model_path: Path to language model for generation
        max_retrieval_rounds: Maximum active retrieval rounds
        high_threshold: High confidence threshold for answers
        low_threshold: Low confidence threshold for answers
        
    Returns:
        Tuple of (retriever, generator)
    """
    print("=" * 80)
    print("Initializing RAG Pipeline")
    print("=" * 80)
    
    # Initialize retriever
    print(f"\n[1/3] Initializing {retriever_type} retriever...")
    if retriever_type == "reranking":
        retriever = RerankingRetriever(
            persist_directory=chroma_db_path,
            k_initial=k_initial,
            k_final=k_final,
            reranker_model=reranker_model,
            collection_name=collection_name,
        )
    elif retriever_type == "basic":
        retriever = BasicRetriever(
            persist_directory=chroma_db_path,
            k=k_final,
            collection_name=collection_name,
        )
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
    print("✓ Retriever initialized")
    
    # Initialize language model
    print(f"\n[2/3] Loading language model: {lm_model_path}...")
    lm = get_local_lm(model_name_or_path=lm_model_path)
    print("✓ Language model loaded")
    
    # Initialize RAG generator
    print(f"\n[3/3] Initializing RAG generator...")
    generator = RAGGenerator(
        lm=lm,
        retriever=retriever,
        k=k_final,
        max_retrieval_rounds=max_retrieval_rounds,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
    )
    print("✓ RAG generator initialized")
    
    print("\n" + "=" * 80)
    print("Pipeline ready!")
    print("=" * 80)
    
    return retriever, generator


def process_query(
    query: str,
    retriever,
    generator: RAGGenerator,
    verbose: bool = True,
) -> dict:
    """
    Process a single query through the RAG pipeline.
    
    Args:
        query: Query string
        retriever: Initialized retriever
        generator: Initialized RAG generator
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with answer, score, status, and metadata
    """
    if verbose:
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print("=" * 80)
    
    # Get initial context from retriever
    if verbose:
        print("\n[Retrieval] Fetching initial context...")
    retriever_results = retriever.retrieve(query)
    initial_context = retriever_results_to_context_chunks(retriever_results)
    
    if verbose:
        print(f"Retrieved {len(initial_context)} chunks")
        if initial_context:
            print(f"Top chunk score: {initial_context[0].score:.4f}")
    
    # Generate answer
    if verbose:
        print("\n[Generation] Generating answer...")
    result = generator.generate_answer(
        query=query,
        initial_context=initial_context,
    )
    
    # Format output
    output = {
        "query": query,
        "answer": result.answer,
        "score": result.score,
        "status": result.status.value,
        "num_retrieval_rounds": result.num_retrieval_rounds,
        "num_chunks_used": len(result.used_chunks),
        "reason": result.reason,
    }
    
    if verbose:
        print("\n" + "-" * 80)
        print("Result:")
        print(f"  Status: {result.status.value}")
        print(f"  Score: {result.score:.4f}")
        print(f"  Retrieval rounds: {result.num_retrieval_rounds}")
        print(f"  Chunks used: {len(result.used_chunks)}")
        print(f"  Reason: {result.reason}")
        print("\nAnswer:")
        print(f"  {result.answer}")
        print("-" * 80)
    
    return output


def main():
    """Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline for Swedish Legal Documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # ChromaDB configuration
    parser.add_argument(
        "--chroma-db",
        type=str,
        default="./chroma_db_test",
        help="Path to ChromaDB directory (default: ./chroma_db_test)",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="sfs_paragraphs",
        help="ChromaDB collection name (default: sfs_paragraphs)",
    )
    
    # Retriever configuration
    parser.add_argument(
        "--retriever-type",
        type=str,
        choices=["reranking", "basic"],
        default="reranking",
        help="Type of retriever to use (default: reranking)",
    )
    parser.add_argument(
        "--k-initial",
        type=int,
        default=50,
        help="Initial retrieval count for reranking retriever (default: 50)",
    )
    parser.add_argument(
        "--k-final",
        type=int,
        default=10,
        help="Final retrieval count after reranking (default: 10)",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="jinaai/jina-reranker-v2-base-multilingual",
        help="Reranker model name (default: jinaai/jina-reranker-v2-base-multilingual)",
    )
    
    # Generation configuration
    parser.add_argument(
        "--lm-model",
        type=str,
        default="gpt2",
        help="Language model path/name for generation (default: gpt2)",
    )
    parser.add_argument(
        "--max-retrieval-rounds",
        type=int,
        default=2,
        help="Maximum active retrieval rounds (default: 2)",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=0.75,
        help="High confidence threshold (default: 0.75)",
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=0.40,
        help="Low confidence threshold (default: 0.40)",
    )
    
    # Query input
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process",
    )
    parser.add_argument(
        "--query-file",
        type=str,
        help="File with queries (one per line)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (prompt for queries)",
    )
    
    # Output options
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON format)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.query and not args.query_file and not args.interactive:
        parser.error("Must specify --query, --query-file, or --interactive")
    
    # Initialize pipeline
    try:
        retriever, generator = initialize_pipeline(
            chroma_db_path=args.chroma_db,
            retriever_type=args.retriever_type,
            collection_name=args.collection_name,
            k_initial=args.k_initial,
            k_final=args.k_final,
            reranker_model=args.reranker_model,
            lm_model_path=args.lm_model,
            max_retrieval_rounds=args.max_retrieval_rounds,
            high_threshold=args.high_threshold,
            low_threshold=args.low_threshold,
        )
    except Exception as e:
        print(f"Error initializing pipeline: {e}", file=sys.stderr)
        return 1
    
    # Collect queries
    queries = []
    if args.query:
        queries.append(args.query)
    elif args.query_file:
        query_file = Path(args.query_file)
        if not query_file.exists():
            print(f"Error: Query file not found: {query_file}", file=sys.stderr)
            return 1
        with open(query_file, "r", encoding="utf-8") as f:
            queries = [line.strip() for line in f if line.strip()]
    
    # Process queries
    results = []
    verbose = not args.quiet
    
    if args.interactive:
        print("\n" + "=" * 80)
        print("Interactive Mode - Enter queries (type 'quit' or 'exit' to stop)")
        print("=" * 80)
        while True:
            try:
                query = input("\nQuery: ").strip()
                if not query or query.lower() in ("quit", "exit", "q"):
                    break
                result = process_query(query, retriever, generator, verbose=verbose)
                results.append(result)
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except EOFError:
                break
    else:
        for query in queries:
            try:
                result = process_query(query, retriever, generator, verbose=verbose)
                results.append(result)
            except Exception as e:
                print(f"Error processing query '{query}': {e}", file=sys.stderr)
                if verbose:
                    import traceback
                    traceback.print_exc()
    
    # Save results if requested
    if args.output:
        import json
        output_file = Path(args.output)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

