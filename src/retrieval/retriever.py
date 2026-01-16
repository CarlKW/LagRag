
import sys
import src.config
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import torch
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

from src.config import DEFAULT_MODEL_CONFIG, DEFAULT_PIPELINE_CONFIG

from src.indexing.embedder import get_embedding_function

# Retrieves chunks from a Chroma vector store and reranks them with a cross-encoder reranker
class RerankingRetriever:
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        k_initial: Optional[int] = None,
        k_final: Optional[int] = None,
        reranker_model: Optional[str] = None,
        batch_size: Optional[int] = None,
        collection_name: Optional[str] = None,
    ):
        # use config defaults
        self.persist_directory = persist_directory or DEFAULT_PIPELINE_CONFIG.chroma_db_path
        self.k_initial = k_initial if k_initial is not None else DEFAULT_PIPELINE_CONFIG.k_initial
        self.k_final = k_final if k_final is not None else DEFAULT_PIPELINE_CONFIG.k_final
        self.batch_size = batch_size if batch_size is not None else DEFAULT_MODEL_CONFIG.reranker_batch_size
        self.reranker_model = reranker_model or DEFAULT_MODEL_CONFIG.reranker_model
        self.collection_name = collection_name or DEFAULT_PIPELINE_CONFIG.collection_name


        self.embeddings = get_embedding_function()

        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )
        
        # santy check: log collection count
        try:
            collection_count = self.vectorstore._collection.count()
            print(f"loaded Chroma  '{self.collection_name}' with {collection_count} documents")
        except Exception as e:
            print(f"WARNINGGG: Could not retrieve collection count: {e}")


        device = DEFAULT_MODEL_CONFIG.reranker_device
        self.reranker = CrossEncoder(
            self.reranker_model,
            device=device,
            trust_remote_code=True
        )

    def _initial_retrieve(self, query: str) -> List[Tuple[Any, float]]:
        return self.vectorstore.similarity_search_with_score(query, k=self.k_initial)

    def _rerank(self, query: str, candidates: List[Tuple[Any, float]]) -> List[float]:
        if not candidates:
            return []
        pairs = [(query, doc.page_content) for doc, _ in candidates]
        # CrossEncoder.predict supports batching internally.
        scores = self.reranker.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return scores.tolist() if hasattr(scores, "tolist") else scores

        
    # Retrieve and rerank chunks for the given query
    def retrieve(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        candidates = self._initial_retrieve(query)
        if not candidates:
            return {
                "initial_candidates": [],
                "reranked_results": []
            }

        rerank_scores = self._rerank(query, candidates)

        # Combine scores with documents.
        enriched = []
        for (doc, retrieval_score), rerank_score in zip(candidates, rerank_scores):
            result = {
                "text": doc.page_content,
                "score_retrieval": float(retrieval_score),  # Cosine distance (lower is better)
                "score_rerank": float(rerank_score),  # Reranker score (higher is better)
            }
            if doc.metadata:
                result.update(doc.metadata)
            enriched.append(result)

        initial_candidates = sorted(enriched, key=lambda x: x["score_retrieval"])
        
        enriched.sort(key=lambda x: x["score_rerank"], reverse=True)
        reranked_results = enriched[: self.k_final]
        
        return {
            "initial_candidates": initial_candidates,
            "reranked_results": reranked_results
        }

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    default_db = project_root / "chroma_db_test"

    print(f"Loading retriever with DB at: {default_db}")
    retriever = RerankingRetriever(persist_directory=str(default_db))

    # queries to test.
    queries = [
        "Narkotika får föras in till eller ut från landet",

        #: "Lag (1992:860)
        #2 § Narkotika får föras in till eller ut från landet,\r\ntillverkas, bjudas ut till försäljning,
        # överlåtas eller innehas\r\nendast för\r\n\r\n1. medicinskt ändamål\r\n\r\n2. vetenskapligt 
        #ändamål\r\n\r\n3. annat samhällsnyttigt ändamål som är särskilt angeläget,\r\neller\r\n\r\n4.
        # industriellt ändamål\r\n\r\na) i de fall regeringen särskilt föreskriver det, eller\r\n\r\nb)
        # om undantag från kravet på tillstånd har meddelats enligt\r\n12 § fjärde stycket.\r\n\r\nVid
        #tillämpning av denna lag ska en vara anses ha förts in till\r\neller ut från landet när den har
        # förts över gränsen för svenskt\r\nterritorium. Lag (2011:114).\r\n\r\nInförsel och utförsel\r\
        #n\r\n

        "Ett villkor för att introduktionsersättning",

        #"Lag (1992:1068) o
        #3 § Ett villkor för att introduktionsersättning skall få beviljas är 
        #att\r\nutlänningen förbinder sig att följa en introduktionsplan som 
        #fastställts\r\nav kommunen efter samråd med utlänningen.\r\n\r\n

        # "Vilka krav gäller för stöd till energieffektivisering?",
    ]

    for query in queries:
        print("\n" + "#" * 80)
        print(f"Running query: {query!r}")

        base_results = retriever.vectorstore.similarity_search_with_score(query, k=1)
        if base_results:
            base_doc, base_score = base_results[0]
            print("\n[Bästa basträff från Chroma (embedding-likhet)]")
            print("=" * 80)
            print(f"Score (Chroma): {base_score:.4f}")
            print(f"Title: {base_doc.metadata.get('titel') or base_doc.metadata.get('title') or 'N/A'}")
            print(f"Paragraph: {base_doc.metadata.get('paragraf', 'N/A')}")
            print("Content preview:")
            text = base_doc.page_content
            print(text[:700] + ("..." if len(text) > 700 else ""))

        results = retriever.retrieve(query)

        if not results:
            print("No results found. Ensure the Chroma DB has been created by the pipeline.")
        else:
            print("\nTop reranked chunks:")
            for i, item in enumerate(results, 1):
                title = item.get("titel") or item.get("title") or "N/A"
                paragraf = item.get("paragraf", "N/A")
                print(f"Result {i}")
                print(f"Title: {title}")
                print(f"Paragraph: {paragraf}")
                print(f"Retrieval score (Chroma): {item['score_retrieval']:.4f}")
                print(f"Rerank score (Jina): {item['score_rerank']:.4f}")
                print("Content preview:")
                print(item["text"][:700] + ("..." if len(item["text"]) > 700 else ""))
                print()
