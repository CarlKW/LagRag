import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.retrieval.retriever import RerankingRetriever
from src.retrieval.retriever_basic import BasicRetriever
from src.generation.lm_wrapper import LocalHFModel, get_local_lm
from src.generation.genAI import RAGGenerator, ContextChunk
from src.generation.adapters import retriever_results_to_context_chunks
from src.config import DEFAULT_MODEL_CONFIG, DEFAULT_PIPELINE_CONFIG


def print_gpu_memory(stage: str = ""):
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    print(f"\nGPU Memory Status: {stage}" if stage else "\nGPU Memory Status")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        total = props.total_memory / (1024**3)
        free = total - reserved
        
        try:
            peak_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)
            peak_reserved = torch.cuda.max_memory_reserved(i) / (1024**3)
        except:
            peak_allocated = None
            peak_reserved = None
        
        print(f"\nGPU {i}: {props.name}")
        print(f"  Total: {total:.2f} GiB | Allocated: {allocated:.2f} GiB | Reserved: {reserved:.2f} GiB | Free: {free:.2f} GiB")
        
        if peak_allocated is not None:
            print(f"  Peak Allocated: {peak_allocated:.2f} GiB | Peak Reserved: {peak_reserved:.2f} GiB")
        
        if free < 1.0:
            print(f"  WARNING: Less than 1 GiB free!")
        if reserved > total * 0.95:
            print(f"  WARNING: GPU {i} is >95% full!")
    print()


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
    
    print("Initializing RAG Pipeline\n")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        num_gpus = torch.cuda.device_count()
        if num_gpus > 0:
            for i in range(num_gpus):
                try:
                    torch.cuda.reset_peak_memory_stats(i)
                except (RuntimeError, ValueError):
                    pass
        print("GPU memory cache cleared")
    
    print_gpu_memory("Before loading any models")
    
    print(f"[1/3] Initializing {retriever_type} retriever...")
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
    
    print_gpu_memory("After loading embedding model and reranker")
    
    print(f"\n[2/3] Loading language model: {lm_model_path}...")
    print_gpu_memory("Before loading generation model")
    
    lm = get_local_lm(
        model_name_or_path=lm_model_path,
        device=DEFAULT_MODEL_CONFIG.generation_device
    )
    print(f"Language model loaded on {DEFAULT_MODEL_CONFIG.generation_device}")
    
    print_gpu_memory("After loading generation model")
    
    print(f"\n[3/3] Initializing RAG generator...")
    generator = RAGGenerator(
        lm=lm,
        retriever=retriever,
        k=k_final,
        max_retrieval_rounds=max_retrieval_rounds,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
    )
    print("RAG generator initialized")
    
    print_gpu_memory("Pipeline initialization complete")
    print("\nPipeline ready!\n")
    
    return retriever, generator


def process_query(query: str, retriever, generator: RAGGenerator, verbose: bool = True) -> dict:
    if verbose:
        print(f"\nQuery: {query}\n")
    
    if verbose:
        print("[Retrieval] Fetching initial context...")
    retriever_results_dict = retriever.retrieve(query)
    initial_candidates = retriever_results_dict["initial_candidates"]
    retriever_results = retriever_results_dict["reranked_results"]
    initial_context = retriever_results_to_context_chunks(retriever_results)
    
    if verbose:
        print("[Generation] Generating answer...")
    result = generator.generate_answer(
        query=query,
        initial_context=initial_context,
    )
    
    chunks_data = []
    for chunk in result.used_chunks:
        chunks_data.append({
            "id": chunk.id,
            "text": chunk.text,
            "score": chunk.score,
            "metadata": chunk.metadata,
        })
    
    initial_candidates_data = []
    for candidate in initial_candidates:
        initial_candidates_data.append({
            "text": candidate.get("text", ""),
            "score_retrieval": candidate.get("score_retrieval"),
            "score_rerank": candidate.get("score_rerank"),
            "metadata": {k: v for k, v in candidate.items() 
                        if k not in ("text", "score_retrieval", "score_rerank")}
        })
    
    output = {
        "query": query,
        "answer": result.answer,
        "score": result.score,
        "status": result.status.value,
        "num_retrieval_rounds": result.num_retrieval_rounds,
        "num_chunks_used": len(result.used_chunks),
        "reason": result.reason,
        "chunks": chunks_data,
        "initial_candidates": initial_candidates_data,
    }
    
    if verbose:
        print("\nResult:")
        print(f"  Status: {result.status.value}")
        print(f"  Score: {result.score:.4f}")
        print(f"  Retrieval rounds: {result.num_retrieval_rounds}")
        print(f"  Chunks used: {len(result.used_chunks)}")
        print(f"  Reason: {result.reason}")
        print(f"\nAnswer: {result.answer}")
        print("\nSource Chunks:")
        for i, chunk in enumerate(result.used_chunks, 1):
            score_str = f"{chunk.score:.4f}" if chunk.score is not None else "N/A"
            print(f"\nChunk {i} (Score: {score_str}):")
            if chunk.metadata:
                metadata_str = ", ".join([f"{k}={v}" for k, v in chunk.metadata.items() if k not in ("text",)])
                if metadata_str:
                    print(f"  Metadata: {metadata_str}")
            print(f"  Text: {chunk.text}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline for Swedish Legal Documents")
    
    parser.add_argument("--chroma-db", type=str, default=None,
                       help=f"Path to ChromaDB directory (default: {DEFAULT_PIPELINE_CONFIG.chroma_db_path})")
    parser.add_argument("--collection-name", type=str, default=None,
                       help=f"ChromaDB collection name (default: {DEFAULT_PIPELINE_CONFIG.collection_name})")
    
    parser.add_argument("--retriever-type", type=str, choices=["reranking", "basic"], default=None,
                       help=f"Type of retriever (default: {DEFAULT_PIPELINE_CONFIG.retriever_type})")
    parser.add_argument("--k-initial", type=int, default=None,
                       help=f"Initial retrieval count (default: {DEFAULT_PIPELINE_CONFIG.k_initial})")
    parser.add_argument("--k-final", type=int, default=None,
                       help=f"Final retrieval count (default: {DEFAULT_PIPELINE_CONFIG.k_final})")
    parser.add_argument("--reranker-model", type=str, default=None,
                       help=f"Reranker model name (default: {DEFAULT_MODEL_CONFIG.reranker_model})")
    
    parser.add_argument("--lm-model", type=str, default=None,
                       help=f"Language model (default: {DEFAULT_MODEL_CONFIG.generation_model})")
    parser.add_argument("--max-retrieval-rounds", type=int, default=None,
                       help=f"Max retrieval rounds (default: {DEFAULT_PIPELINE_CONFIG.max_retrieval_rounds})")
    parser.add_argument("--high-threshold", type=float, default=None,
                       help=f"High confidence threshold (default: {DEFAULT_PIPELINE_CONFIG.high_threshold})")
    parser.add_argument("--low-threshold", type=float, default=None,
                       help=f"Low confidence threshold (default: {DEFAULT_PIPELINE_CONFIG.low_threshold})")
    
    parser.add_argument("--query", type=str, help="Single query to process")
    parser.add_argument("--query-file", type=str, help="File with queries (one per line)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--output", type=str, help="Output file for results (JSON format)")
    
    args = parser.parse_args()
    
    if not args.query and not args.query_file and not args.interactive:
        parser.error("Must specify --query, --query-file, or --interactive")
    
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
    
    results = []
    verbose = not args.quiet
    
    if args.interactive:
        print("\nInteractive Mode - Enter queries (type 'quit' or 'exit' to stop)\n")
        while True:
            try:
                query = input("Query: ").strip()
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
    
    if args.output:
        output_file = Path(args.output)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
