"""
Basic retriever module that loads a persisted Chroma DB and retrieves chunks
using similarity search with cosine similarity. No reranking is performed.
"""
import sys
from typing import Any, Dict, List
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from langchain_chroma import Chroma

from src.indexing.embedder import get_embedding_function


class BasicRetriever:
    """
    Retrieves chunks from a Chroma vector store using similarity search.
    Returns results sorted by cosine similarity (higher is better).
    Uses the same collection name and embedding function as the indexing pipeline.
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db_test",
        k: int = 5,
        collection_name: str = "sfs_paragraphs",
    ):
        """
        Initialize the basic retriever.

        Args:
            persist_directory: Path to the persisted Chroma database
            k: Number of top results to return
            collection_name: Name of the Chroma collection (must match indexing)
        """
        self.persist_directory = persist_directory
        self.k = k
        self.collection_name = collection_name

        # Check GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Warning: GPU not available, using CPU")

        # Load the same embedding model used for chunk creation.
        # The embedding model automatically uses GPU if available.
        self.embeddings = get_embedding_function()

        # Connect to the existing Chroma DB with the same collection name.
        # Chroma will use cosine similarity as configured during indexing.
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )
        
        # Sanity check: log collection count
        try:
            collection_count = self.vectorstore._collection.count()
            print(f"Loaded Chroma collection '{self.collection_name}' with {collection_count} documents")
        except Exception as e:
            print(f"Warning: Could not retrieve collection count: {e}")

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve chunks for the given query using similarity search.

        Args:
            query: Query string to search for

        Returns:
            List of dictionaries with text, retrieval score (cosine distance),
            and the original metadata fields. Results are sorted by cosine similarity
            (descending, so higher is better). Note: Chroma returns distance (1 - similarity),
            so lower distance = higher similarity.
        """
        # Run similarity search with scores
        # With cosine similarity, Chroma returns distance (1 - cosine_similarity)
        # Lower distance = higher similarity
        results = self.vectorstore.similarity_search_with_score(query, k=self.k)

        if not results:
            return []

        # Convert to dictionary format
        enriched = []
        for doc, retrieval_score in results:
            # retrieval_score is cosine distance (1 - cosine_similarity)
            # Lower distance = higher similarity
            result = {
                "text": doc.page_content,
                "score_retrieval": float(retrieval_score),  # Cosine distance (lower is better)
            }
            # Include all metadata fields if present.
            if doc.metadata:
                result.update(doc.metadata)
            enriched.append(result)

        # Results are already sorted by Chroma (best first, lowest cosine distance)
        return enriched


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    default_db = project_root / "chroma_db_test"

    print(f"Loading basic retriever with DB at: {default_db}")
    retriever = BasicRetriever(persist_directory=str(default_db), k=5)

    # Define a list of queries to test.
    queries = [
        "Narkotika får föras in till eller ut från landet",
        "Ett villkor för att introduktionsersättning",

        "begäran om att skicka ett viktigt meddelande till allmänheten"
        #"Lag (2023:407) om viktigt meddelande till allmänheten", 
        #§ En begäran om sändning av ett viktigt meddelande till \nallmänheten ska göras till samhällets alarmeringstjänst. 
        # \nSamhällets alarmeringstjänst ska ta emot och vidareförmedla 
        # \nbegäran till den eller dem som ansvarar för sändning av 
        # \nmeddelandet.\n\nEtt viktigt meddelande till allmänheten om
    ]

    for query in queries:
        print("\n" + "#" * 80)
        print(f"Running query: {query!r}")

        results = retriever.retrieve(query)

        if not results:
            print("No results found. Ensure the Chroma DB has been created by the pipeline.")
        else:
            print(f"\nTop 5 chunks (sorted by cosine similarity, lower distance = higher similarity):")
            for i, item in enumerate(results[:5], 1):
                title = item.get("titel") or item.get("title") or "N/A"
                paragraf = item.get("paragraf", "N/A")
                print("=" * 80)
                print(f"Result {i}")
                print(f"Title: {title}")
                print(f"Paragraph: {paragraf}")
                print(f"Cosine distance score: {item['score_retrieval']:.4f} (lower = more similar)")
                print("Content preview:")
                print(item["text"][:700] + ("..." if len(item["text"]) > 700 else ""))
                print()
        
        # Sanity check: exact-match test (commented out by default)
        # Uncomment to test: querying with an exact chunk text should return itself as top-1
        # if results:
        #     # Get the text of the top result
        #     top_result_text = results[0]["text"]
        #     # Query with the exact same text
        #     exact_results = retriever.retrieve(top_result_text)
        #     if exact_results:
        #         exact_top_text = exact_results[0]["text"]
        #         if exact_top_text == top_result_text:
        #             print("✓ Exact-match test passed: querying with chunk text returns itself as top-1")
        #         else:
        #             print("⚠ Exact-match test failed: querying with chunk text did not return itself as top-1")
        #             print(f"  Expected: {top_result_text[:100]}...")
        #             print(f"  Got: {exact_top_text[:100]}...")

