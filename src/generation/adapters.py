"""
Adapter functions to convert between retriever output formats and generator input formats.
"""
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from src.generation.genAI import ContextChunk
else:
    # Import at runtime to avoid circular imports
    ContextChunk = None


def retriever_results_to_context_chunks(
    retriever_results: List[Dict[str, Any]]
) -> List["ContextChunk"]:
    """
    Convert retriever dict output to ContextChunk objects.
    
    The retriever returns dictionaries with keys:
    - text: chunk text content
    - score_retrieval: cosine distance (lower is better)
    - score_rerank: reranker score (higher is better)
    - metadata fields: sfs_nr, paragraf, titel, etc.
    
    Args:
        retriever_results: List of dictionaries from retriever.retrieve()
        
    Returns:
        List of ContextChunk objects with:
        - id: unique identifier constructed from metadata
        - text: chunk text
        - score: rerank score (higher is better)
        - metadata: all original metadata fields
    """
    # Import here to avoid circular imports
    from src.generation.genAI import ContextChunk
    
    context_chunks = []
    
    for i, result in enumerate(retriever_results):
        # Extract text
        text = result.get("text", "")
        
        # Use rerank score as primary score (higher is better)
        score = result.get("score_rerank")
        if score is None:
            # Fallback to retrieval score if rerank score not available
            # Convert distance to similarity (lower distance = higher similarity)
            retrieval_score = result.get("score_retrieval")
            if retrieval_score is not None:
                # Invert distance to get similarity (1 - distance)
                score = 1.0 - float(retrieval_score)
        
        # Create unique ID from metadata
        sfs_nr = result.get("sfs_nr", "unknown")
        paragraf = result.get("paragraf", "unknown")
        subchunk_index = result.get("subchunk_index")
        
        if subchunk_index is not None:
            chunk_id = f"{sfs_nr}_{paragraf}_{subchunk_index}"
        else:
            chunk_id = f"{sfs_nr}_{paragraf}"
        
        # If ID is not unique enough, add index
        # (This shouldn't happen often, but ensures uniqueness)
        if any(cc.id == chunk_id for cc in context_chunks):
            chunk_id = f"{chunk_id}_{i}"
        
        # Extract metadata (exclude fields we've already used)
        metadata = {k: v for k, v in result.items() 
                   if k not in ("text", "score_retrieval", "score_rerank")}
        
        # Create ContextChunk
        chunk = ContextChunk(
            id=chunk_id,
            text=text,
            score=float(score) if score is not None else None,
            metadata=metadata
        )
        
        context_chunks.append(chunk)
    
    return context_chunks

