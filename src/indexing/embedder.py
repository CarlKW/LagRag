"""
Embedder for creating embeddings using the TTC-L2V-supervised-2 model.
"""

from typing import List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer


class TTCEmbeddings(Embeddings):
    """
    LangChain-compatible embedding class using TTC-L2V-supervised-2 model.
    """
    
    def __init__(self, model_name: str = "TTC-L2V-supervised-2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the HuggingFace/sentence-transformers model
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding.tolist()


def get_embedding_function(model_name: str = "TTC-L2V-supervised-2") -> Embeddings:
    """
    Get a LangChain-compatible embedding function.
    
    Args:
        model_name: Name of the model (default: "TTC-L2V-supervised-2")
        
    Returns:
        LangChain Embeddings object that can be passed to vector stores
    """
    return TTCEmbeddings(model_name=model_name)


if __name__ == "__main__":
    # Example usage
    print("Loading embedding model...")
    embeddings = get_embedding_function()
    
    # Test embedding
    test_texts = [
        "Detta är en testtext för att kontrollera att embedding-modellen fungerar.",
        "This is a test text to verify the embedding model works."
    ]
    
    print("Creating embeddings...")
    embedded = embeddings.embed_documents(test_texts)
    
    print(f"Embedded {len(embedded)} texts")
    print(f"Embedding dimension: {len(embedded[0])}")

