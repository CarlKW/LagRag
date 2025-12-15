"""
Retriever module that loads a persisted Chroma DB, runs an initial similarity
search with the same embedding model used for indexing, and reranks the
candidate chunks with a cross-encoder (jinaai/jina-reranker-v2-base-multilingual).
"""

from typing import Any, Dict, List, Tuple
from pathlib import Path

import torch
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

from src.indexing.embedder import get_embedding_function


class RerankingRetriever:
    """
    Retrieves chunks from a Chroma vector store and reranks them with a
    cross-encoder reranker.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db_test",
        k_initial: int = 50,
        k_final: int = 5,
        reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual",
        batch_size: int = 16,
    ):
        self.persist_directory = persist_directory
        self.k_initial = k_initial
        self.k_final = k_final
        self.batch_size = batch_size
        self.reranker_model = reranker_model

        # Load the same embedding model used for chunk creation.
        self.embeddings = get_embedding_function()

        # Connect to the existing Chroma DB.
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

        # Load reranker once; it is a cross-encoder that scores (query, doc) pairs.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker = CrossEncoder(self.reranker_model, device=device)

    def _initial_retrieve(self, query: str) -> List[Tuple[Any, float]]:
        """Run initial similarity search in Chroma and return (doc, score) tuples."""
        return self.vectorstore.similarity_search_with_score(query, k=self.k_initial)

    def _rerank(self, query: str, candidates: List[Tuple[Any, float]]) -> List[float]:
        """Score candidate chunks with the cross-encoder reranker."""
        if not candidates:
            return []
        pairs = [(query, doc.page_content) for doc, _ in candidates]
        # CrossEncoder.predict supports batching internally.
        scores = self.reranker.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return scores.tolist() if hasattr(scores, "tolist") else scores

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve and rerank chunks for the given query.

        Returns a list of dictionaries with text, retrieval score, rerank score,
        and the original metadata fields.
        """
        candidates = self._initial_retrieve(query)
        if not candidates:
            return []

        rerank_scores = self._rerank(query, candidates)

        # Combine scores with documents.
        enriched = []
        for (doc, retrieval_score), rerank_score in zip(candidates, rerank_scores):
            result = {
                "text": doc.page_content,
                "score_retrieval": float(retrieval_score),
                "score_rerank": float(rerank_score),
            }
            # Include all metadata fields if present.
            if doc.metadata:
                result.update(doc.metadata)
            enriched.append(result)

        # Sort by rerank score descending and take top k_final.
        enriched.sort(key=lambda x: x["score_rerank"], reverse=True)
        return enriched[: self.k_final]


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    default_db = project_root / "chroma_db_test"

    print(f"Loading retriever with DB at: {default_db}")
    retriever = RerankingRetriever(persist_directory=str(default_db))

    sample_query = "När får bidraget betalas ut för investeringsbidrag?"
    print(f"\nRunning sample query: {sample_query!r}")
    results = retriever.retrieve(sample_query)

    if not results:
        print("No results found. Ensure the Chroma DB has been created by the pipeline.")
    else:
        print("\nTop reranked chunks:")
        for i, item in enumerate(results, 1):
            title = item.get("titel") or item.get("title") or "N/A"
            paragraf = item.get("paragraf", "N/A")
            print("=" * 80)
            print(f"Result {i}")
            print(f"Title: {title}")
            print(f"Paragraph: {paragraf}")
            print(f"Retrieval score: {item['score_retrieval']:.4f}")
            print(f"Rerank score: {item['score_rerank']:.4f}")
            print("Content preview:")
            print(item["text"][:400] + ("..." if len(item["text"]) > 400 else ""))
            print()

