"""
Retriever module that loads a persisted Chroma DB, runs an initial similarity
search with the same embedding model used for indexing, and reranks the
candidate chunks with a cross-encoder (jinaai/jina-reranker-v2-base-multilingual).
"""
import sys
from typing import Any, Dict, List, Tuple
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
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
        self.reranker = CrossEncoder(
            self.reranker_model,
            device=device,
            trust_remote_code=True

        )

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

    # Define a list of queries to test.
    queries = [
        "Vilken lag snackar om att narkotika bara får säljas för medicinsk användning?",

        #: "Lag (1992:860)
        #2 § Narkotika får föras in till eller ut från landet,\r\ntillverkas, bjudas ut till försäljning,
        # överlåtas eller innehas\r\nendast för\r\n\r\n1. medicinskt ändamål\r\n\r\n2. vetenskapligt 
        #ändamål\r\n\r\n3. annat samhällsnyttigt ändamål som är särskilt angeläget,\r\neller\r\n\r\n4.
        # industriellt ändamål\r\n\r\na) i de fall regeringen särskilt föreskriver det, eller\r\n\r\nb)
        # om undantag från kravet på tillstånd har meddelats enligt\r\n12 § fjärde stycket.\r\n\r\nVid
        #tillämpning av denna lag ska en vara anses ha förts in till\r\neller ut från landet när den har
        # förts över gränsen för svenskt\r\nterritorium. Lag (2011:114).\r\n\r\nInförsel och utförsel\r\
        #n\r\n

        "vad gäller för en kommun ska få bidrag för att ta in untlänningar?",

        #"Lag (1992:1068) o
        #3 § Ett villkor för att introduktionsersättning skall få beviljas är 
        #att\r\nutlänningen förbinder sig att följa en introduktionsplan som 
        #fastställts\r\nav kommunen efter samråd med utlänningen.\r\n\r\n

        # "Vilka krav gäller för stöd till energieffektivisering?",
    ]

    for query in queries:
        print("\n" + "#" * 80)
        print(f"Running query: {query!r}")
        results = retriever.retrieve(query)

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
                print(item["text"][:700] + ("..." if len(item["text"]) > 400 else ""))
                print()
