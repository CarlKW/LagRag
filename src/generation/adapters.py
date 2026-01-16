
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from src.generation.genAI import ContextChunk
else:
    # Import at runtime to avoid circular imports
    ContextChunk = None


def retriever_results_to_context_chunks(
    retriever_results: List[Dict[str, Any]]
) -> List["ContextChunk"]:
    
    #Convert retriever dict output to ContextChunk objects.

    # Import here to avoid circular imports
    from src.generation.genAI import ContextChunk
    
    context_chunks = []
    
    for i, result in enumerate(retriever_results):
        # Extract text
        text = result.get("text", "")
        
        # rerank score(higher is better)
        score = result.get("score_rerank")
        if score is None:
            # fallback to retrieval 
            # (lower distance = higher similarity)
            retrieval_score = result.get("score_retrieval")
            if retrieval_score is not None:
                score = 1.0 - float(retrieval_score)
        
        # unique ID from metadata
        sfs_nr = result.get("sfs_nr", "unknown")
        paragraf = result.get("paragraf", "unknown")
        subchunk_index = result.get("subchunk_index")
        
        if subchunk_index is not None:
            chunk_id = f"{sfs_nr}_{paragraf}_{subchunk_index}"
        else:
            chunk_id = f"{sfs_nr}_{paragraf}"
        
        # If ID is not unique enough, add index
        if any(cc.id == chunk_id for cc in context_chunks):
            chunk_id = f"{chunk_id}_{i}"
        
        metadata = {k: v for k, v in result.items() 
                   if k not in ("text", "score_retrieval", "score_rerank")}
        
        chunk = ContextChunk(
            id=chunk_id,
            text=text,
            score=float(score) if score is not None else None,
            metadata=metadata
        )
        
        context_chunks.append(chunk)
    
    return context_chunks

