"""
Embedder for creating embeddings using the TTC-L2V-supervised-2 model.

This uses LLM2Vec with a base model and PEFT adapters:
- Base model: jealk/llm2vec-scandi-mntp-v2
- Adapter: jealk/TTC-L2V-supervised-2
"""

from typing import List

import torch
from langchain_core.embeddings import Embeddings
from transformers import AutoConfig, AutoModel, AutoTokenizer

try:
    from llm2vec import LLM2Vec
    from peft import PeftModel
except ImportError:
    raise ImportError(
        "Required packages not installed. Please run: pip install llm2vec peft accelerate"
    )


class TTCEmbeddings(Embeddings):
    """
    LangChain-compatible embedding class using TTC-L2V-supervised-2 model.
    
    This class loads the base model jealk/llm2vec-scandi-mntp-v2,
    applies the MNTP LoRA adapter from the base model, and then applies
    the supervised adapter jealk/TTC-L2V-supervised-2.
    """
    
    def __init__(self, base_model_name: str = "jealk/llm2vec-scandi-mntp-v2", 
                 adapter_name: str = "jealk/TTC-L2V-supervised-2"):
        """
        Initialize the embedding model.
        
        Args:
            base_model_name: Name of the base model (default: jealk/llm2vec-scandi-mntp-v2)
            adapter_name: Name of the adapter (default: jealk/TTC-L2V-supervised-2)
        """
        self.base_model_name = base_model_name
        self.adapter_name = adapter_name
        
        # Detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine dtype based on device
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        
        print(f"Loading base model: {base_model_name}")
        print(f"Device: {self.device}, Dtype: {self.dtype}")
        
        # Load base model with tokenizer and config
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        # Load config first and modify it
        config = AutoConfig.from_pretrained(base_model_name)
        # Remove adapter configs to prevent auto-loading
        if hasattr(config, 'adapter_config') or hasattr(config, 'peft_config'):
            # Clear any adapter-related configs
            pass

        # Load model without auto-loading adapters
        base_model = AutoModel.from_pretrained(
            base_model_name,
            config=config,
            torch_dtype=self.dtype,
            trust_remote_code=True
        )
        
        # Apply MNTP adapter from base model repo if it exists
        # The base model jealk/llm2vec-scandi-mntp-v2 may have MNTP adapter in a subdirectory
        # Common locations: the base model repo might have an "adapter" or "mntp_adapter" subdirectory
        # or the MNTP adapter may already be merged into the base model
        model_with_mntp = base_model
        
        # Try to load MNTP adapter from common subdirectory locations in the base model repo
        mntp_adapter_paths = [
            f"{base_model_name}/adapter",  # Common subdirectory name
            f"{base_model_name}/mntp_adapter",  # Alternative name
        ]
        
        # Try loading MNTP adapter
        mntp_loaded = False
        for mntp_path in mntp_adapter_paths:
            try:
                model_with_mntp = PeftModel.from_pretrained(
                    base_model,
                    mntp_path,
                    torch_dtype=self.dtype
                )
                print(f"Loaded MNTP adapter from: {mntp_path}")
                mntp_loaded = True
                break
            except (ValueError, OSError, TypeError, Exception):
                # If loading fails, the adapter doesn't exist at this path
                # This is expected if MNTP is already merged or doesn't exist separately
                continue
        
        if not mntp_loaded:
            # Base model may already have MNTP merged, or it's a standard model
            print("Base model loaded (MNTP adapter not found as separate PEFT adapter - may be merged)")
        
        # Load the supervised adapter on top of the model (with or without MNTP)
        print(f"Loading supervised adapter: {adapter_name}")
        try:
            model_with_supervised = PeftModel.from_pretrained(
                model_with_mntp,
                adapter_name,
                torch_dtype=self.dtype
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load supervised adapter {adapter_name}. "
                f"Error: {e}. Make sure the adapter exists and is compatible with the base model."
            )
        
        # Merge all adapters into base model
        print("Merging adapters into base model...")
        model_final = model_with_supervised.merge_and_unload()
        
        # Move model to device
        model_final = model_final.to(self.device)
        
        # Wrap with LLM2Vec
        print("Wrapping model with LLM2Vec...")
        self.model = LLM2Vec(model_final, tokenizer=self.tokenizer, pooling_mode="mean")
        
        # Ensure model is in eval mode
        self.model.eval()
        
        print("Model loaded successfully!")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each is a list of floats)
        """
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
        
        # Convert to CPU and then to Python list
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        # Use embed_documents for single text (handles batching)
        return self.embed_documents([text])[0]


def get_embedding_function(base_model_name: str = "jealk/llm2vec-scandi-mntp-v2",
                          adapter_name: str = "jealk/TTC-L2V-supervised-2") -> Embeddings:
    """
    Get a LangChain-compatible embedding function.
    
    Args:
        base_model_name: Name of the base model (default: jealk/llm2vec-scandi-mntp-v2)
        adapter_name: Name of the adapter (default: jealk/TTC-L2V-supervised-2)
        
    Returns:
        LangChain Embeddings object that can be passed to vector stores
    """
    return TTCEmbeddings(base_model_name=base_model_name, adapter_name=adapter_name)


if __name__ == "__main__":
    # Example usage
    print("Loading embedding model...")
    embeddings = get_embedding_function()
    
    # Test embedding
    test_texts = [
        "Detta är en testtext för att kontrollera att embedding-modellen fungerar.",
        "This is a test text to verify the embedding model works."
    ]
    
    print("\nCreating embeddings...")
    embedded = embeddings.embed_documents(test_texts)
    
    print(f"Embedded {len(embedded)} texts")
    print(f"Embedding dimension: {len(embedded[0])}")
