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
import json
import time
from pathlib import Path
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.retrieval.retriever import RerankingRetriever
from src.retrieval.retriever_basic import BasicRetriever
from src.generation.lm_wrapper import LocalHFModel, get_local_lm
from src.generation.genAI import RAGGenerator, ContextChunk
from src.generation.adapters import retriever_results_to_context_chunks
from src.config import DEFAULT_MODEL_CONFIG, DEFAULT_PIPELINE_CONFIG


def _log_gpu_memory(location: str, hypothesis_id: str):
    """Log GPU memory usage for debugging."""
    try:
        if torch.cuda.is_available():
            log_data = {
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': hypothesis_id,
                'location': location,
                'message': 'GPU memory check',
                'data': {}
            }
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                log_data['data'][f'gpu_{i}_allocated_gb'] = round(allocated, 2)
                log_data['data'][f'gpu_{i}_reserved_gb'] = round(reserved, 2)
            log_data['timestamp'] = time.time() * 1000
            with open('/data/users/spreitz/LagRag/.cursor/debug.log', 'a') as f:
                f.write(json.dumps(log_data) + '\n')
    except Exception:
        pass


def print_gpu_memory(stage: str = ""):
    """
    Print detailed GPU memory information to stdout.
    
    Args:
        stage: Description of the current stage (e.g., "Before loading models")
    """
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print("\n" + "=" * 80)
    if stage:
        print(f"GPU Memory Status: {stage}")
    else:
        print("GPU Memory Status")
    print("=" * 80)
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GiB
        reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GiB
        total = props.total_memory / (1024**3)  # GiB
        free = total - reserved
        
        # Get peak memory if available
        try:
            peak_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)
            peak_reserved = torch.cuda.max_memory_reserved(i) / (1024**3)
        except:
            peak_allocated = None
            peak_reserved = None
        
        print(f"\nGPU {i}: {props.name}")
        print(f"  Total Memory:     {total:8.2f} GiB")
        print(f"  Allocated:         {allocated:8.2f} GiB ({allocated/total*100:5.1f}%)")
        print(f"  Reserved:          {reserved:8.2f} GiB ({reserved/total*100:5.1f}%)")
        print(f"  Free:              {free:8.2f} GiB ({free/total*100:5.1f}%)")
        
        if peak_allocated is not None:
            print(f"  Peak Allocated:    {peak_allocated:8.2f} GiB")
        if peak_reserved is not None:
            print(f"  Peak Reserved:     {peak_reserved:8.2f} GiB")
        
        # Check for potential issues
        if free < 1.0:
            print(f"  WARNING: Less than 1 GiB free!")
        if reserved > total * 0.95:
            print(f"  WARNING: GPU {i} is >95% full!")
    
    print("=" * 80 + "\n")


def initialize_pipeline(
    chroma_db_path: Optional[str] = None,
    retriever_type: Optional[str] = None,
    collection_name: Optional[str] = None,
    k_initial: Optional[int] = None,
    k_final: Optional[int] = None,
    reranker_model: Optional[str] = None,
    lm_model_path: Optional[str] = None,
    max_retrieval_rounds: Optional[int] = None,
    high_threshold: Optional[float] = None,
    low_threshold: Optional[float] = None,
):
    """
    Initialize the complete RAG pipeline.
    
    All arguments are optional and will use defaults from DEFAULT_MODEL_CONFIG
    and DEFAULT_PIPELINE_CONFIG if None.
    
    Args:
        chroma_db_path: Path to ChromaDB directory (default: from DEFAULT_PIPELINE_CONFIG)
        retriever_type: "reranking" or "basic" (default: from DEFAULT_PIPELINE_CONFIG)
        collection_name: ChromaDB collection name (default: from DEFAULT_PIPELINE_CONFIG)
        k_initial: Initial retrieval count for reranking retriever (default: from DEFAULT_PIPELINE_CONFIG)
        k_final: Final retrieval count after reranking (default: from DEFAULT_PIPELINE_CONFIG)
        reranker_model: Model name for reranker (default: from DEFAULT_MODEL_CONFIG)
        lm_model_path: Path to language model for generation (default: from DEFAULT_MODEL_CONFIG)
        max_retrieval_rounds: Maximum active retrieval rounds (default: from DEFAULT_PIPELINE_CONFIG)
        high_threshold: High confidence threshold for answers (default: from DEFAULT_PIPELINE_CONFIG)
        low_threshold: Low confidence threshold for answers (default: from DEFAULT_PIPELINE_CONFIG)
        
    Returns:
        Tuple of (retriever, generator)
    """
    # Use config defaults if arguments are None
    chroma_db_path = chroma_db_path or DEFAULT_PIPELINE_CONFIG.chroma_db_path
    retriever_type = retriever_type or DEFAULT_PIPELINE_CONFIG.retriever_type
    collection_name = collection_name or DEFAULT_PIPELINE_CONFIG.collection_name
    k_initial = k_initial if k_initial is not None else DEFAULT_PIPELINE_CONFIG.k_initial
    k_final = k_final if k_final is not None else DEFAULT_PIPELINE_CONFIG.k_final
    reranker_model = reranker_model or DEFAULT_MODEL_CONFIG.reranker_model
    lm_model_path = lm_model_path or DEFAULT_MODEL_CONFIG.generation_model
    max_retrieval_rounds = max_retrieval_rounds if max_retrieval_rounds is not None else DEFAULT_PIPELINE_CONFIG.max_retrieval_rounds
    high_threshold = high_threshold if high_threshold is not None else DEFAULT_PIPELINE_CONFIG.high_threshold
    low_threshold = low_threshold if low_threshold is not None else DEFAULT_PIPELINE_CONFIG.low_threshold
    print("=" * 80)
    print("Initializing RAG Pipeline")
    print("=" * 80)
    
    # Clear GPU memory and reset stats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            for i in range(num_gpus):
                try:
                    # Try with device index
                    torch.cuda.reset_peak_memory_stats(i)
                except (RuntimeError, ValueError) as e:
                    # Some GPUs might not support peak memory stats or might not be accessible
                    # This is not critical, so we just continue
                    pass
        print("GPU memory cache cleared")
    
    # Print initial GPU memory status
    print_gpu_memory("Before loading any models")
    
    # #region agent log
    _log_gpu_memory('pipeline.py:71', 'A,B,C,D,E')
    try:
        if torch.cuda.is_available():
            default_device = torch.cuda.current_device()
            log_data = {'sessionId':'debug-session','runId':'run1','hypothesisId':'D','location':'pipeline.py:72','message':'Default CUDA device check','data':{'default_device':default_device,'gpu_count':torch.cuda.device_count()},'timestamp':time.time()*1000}
            with open('/data/users/spreitz/LagRag/.cursor/debug.log', 'a') as f:
                f.write(json.dumps(log_data) + '\n')
    except: pass
    # #endregion
    
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
    print("Retriever initialized")
    
    # Print GPU memory after embedding/reranker loaded
    print_gpu_memory("After loading embedding model and reranker")
    
    # #region agent log
    _log_gpu_memory('pipeline.py:93', 'A,B,C,D,E')
    # #endregion
    
   # Initialize language model
    print(f"\n[2/3] Loading language model: {lm_model_path}...")
    
    # Print GPU memory before loading generation model
    print_gpu_memory("Before loading generation model")
    
    # #region agent log
    _log_gpu_memory('pipeline.py:99', 'A,B,C,D,E')
    # #endregion
    lm = get_local_lm(
        model_name_or_path=lm_model_path,
        device=DEFAULT_MODEL_CONFIG.generation_device  # Use GPU 1
    )
    print(f"Language model loaded on {DEFAULT_MODEL_CONFIG.generation_device}")
    
    # Print GPU memory after generation model loaded
    print_gpu_memory("After loading generation model")
    
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
    
    # Final GPU memory status
    print_gpu_memory("Pipeline initialization complete")
    
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
        Dictionary with answer, score, status, metadata, and chunks
    """
    if verbose:
        print("\n" + "=" * 80)
        print(f"Query: {query}")
        print("=" * 80)
    
    # Get initial context from retriever
    if verbose:
        print("\n[Retrieval] Fetching initial context...")
    retriever_results_dict = retriever.retrieve(query)
    initial_candidates = retriever_results_dict["initial_candidates"]
    retriever_results = retriever_results_dict["reranked_results"]  # Extract reranked results
    initial_context = retriever_results_to_context_chunks(retriever_results)
    
    if verbose:
        print(f"Retrieved {len(initial_candidates)} initial candidates (before reranking)")
        print(f"After reranking: {len(initial_context)} chunks")
        if initial_context:
            print(f"Top reranked chunk score: {initial_context[0].score:.4f}")
        
        # Print initial candidates for debugging
        print("\n" + "=" * 80)
        print("Initial Candidates (before reranking):")
        print("=" * 80)
        for i, candidate in enumerate(initial_candidates[:49], 1):  # Show top 10
            score_str = f"{candidate.get('score_retrieval', 'N/A'):.4f}" if candidate.get('score_retrieval') is not None else "N/A"
            title = candidate.get("titel") or candidate.get("title") or "N/A"
            paragraf = candidate.get("paragraf", "N/A")
            print(f"\nInitial Candidate {i}:")
            print(f"  Title: {title}")
            print(f"  Paragraph: {paragraf}")
            print(f"  Retrieval Score: {score_str}")
            text_preview = candidate.get("text", "")[:200] + "..." if len(candidate.get("text", "")) > 200 else candidate.get("text", "")
            print(f"  Text: {text_preview}")
        if len(initial_candidates) > 10:
            print(f"\n... and {len(initial_candidates) - 10} more initial candidates")
        print("=" * 80)
    
    # Generate answer
    if verbose:
        print("\n[Generation] Generating answer...")
    result = generator.generate_answer(
        query=query,
        initial_context=initial_context,
    )
    
    # Convert chunks to dictionaries for JSON serialization
    chunks_data = []
    for chunk in result.used_chunks:
        chunk_dict = {
            "id": chunk.id,
            "text": chunk.text,
            "score": chunk.score,
            "metadata": chunk.metadata,
        }
        chunks_data.append(chunk_dict)
    
    # Convert initial candidates to dictionaries for JSON serialization
    initial_candidates_data = []
    for candidate in initial_candidates:
        candidate_dict = {
            "text": candidate.get("text", ""),
            "score_retrieval": candidate.get("score_retrieval"),
            "score_rerank": candidate.get("score_rerank"),
            "metadata": {k: v for k, v in candidate.items() 
                        if k not in ("text", "score_retrieval", "score_rerank")}
        }
        initial_candidates_data.append(candidate_dict)
    
    # Format output
    output = {
        "query": query,
        "answer": result.answer,
        "score": result.score,
        "status": result.status.value,
        "num_retrieval_rounds": result.num_retrieval_rounds,
        "num_chunks_used": len(result.used_chunks),
        "reason": result.reason,
        "chunks": chunks_data,  # Reranked chunks used in generation
        "initial_candidates": initial_candidates_data,  # All initial candidates (50)
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
        print("\n" + "-" * 80)
        print("Source Chunks:")
        print("-" * 80)
        for i, chunk in enumerate(result.used_chunks, 1):
            score_str = f"{chunk.score:.4f}" if chunk.score is not None else "N/A"
            print(f"\nChunk {i} (Score: {score_str}):")
            if chunk.metadata:
                metadata_str = ", ".join([f"{k}={v}" for k, v in chunk.metadata.items() if k not in ("text",)])
                if metadata_str:
                    print(f"  Metadata: {metadata_str}")
            # Print first 200 characters of chunk text
            chunk_preview = chunk.text
            print(f"  Text: {chunk_preview}")
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
        default=None,
        help=f"Path to ChromaDB directory (default: {DEFAULT_PIPELINE_CONFIG.chroma_db_path})",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=None,
        help=f"ChromaDB collection name (default: {DEFAULT_PIPELINE_CONFIG.collection_name})",
    )
    
    # Retriever configuration
    parser.add_argument(
        "--retriever-type",
        type=str,
        choices=["reranking", "basic"],
        default=None,
        help=f"Type of retriever to use (default: {DEFAULT_PIPELINE_CONFIG.retriever_type})",
    )
    parser.add_argument(
        "--k-initial",
        type=int,
        default=None,
        help=f"Initial retrieval count for reranking retriever (default: {DEFAULT_PIPELINE_CONFIG.k_initial})",
    )
    parser.add_argument(
        "--k-final",
        type=int,
        default=None,
        help=f"Final retrieval count after reranking (default: {DEFAULT_PIPELINE_CONFIG.k_final})",
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default=None,
        help=f"Reranker model name (default: {DEFAULT_MODEL_CONFIG.reranker_model})",
    )
    
    # Generation configuration
    parser.add_argument(
        "--lm-model",
        type=str,
        default=None,
        help=f"Language model path/name for generation (default: {DEFAULT_MODEL_CONFIG.generation_model})",
    )
    parser.add_argument(
        "--max-retrieval-rounds",
        type=int,
        default=None,
        help=f"Maximum active retrieval rounds (default: {DEFAULT_PIPELINE_CONFIG.max_retrieval_rounds})",
    )
    parser.add_argument(
        "--high-threshold",
        type=float,
        default=None,
        help=f"High confidence threshold (default: {DEFAULT_PIPELINE_CONFIG.high_threshold})",
    )
    parser.add_argument(
        "--low-threshold",
        type=float,
        default=None,
        help=f"Low confidence threshold (default: {DEFAULT_PIPELINE_CONFIG.low_threshold})",
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
    # Pass None for arguments not provided (will use config defaults)
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
        import traceback
        print(f"Error initializing pipeline: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
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

