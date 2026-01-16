"""
Embedder for creating embeddings using the TTC-L2V-supervised-2 model.

This uses LLM2Vec with a base model and PEFT adapters:
- Base model: jealk/llm2vec-scandi-mntp-v2
- Adapter: jealk/TTC-L2V-supervised-2
"""

from typing import List, Optional
import numpy as np
import torch
from langchain_core.embeddings import Embeddings
from transformers import AutoConfig, AutoModel, AutoTokenizer
# from src.config import DEFAULT_MODEL_CONFIG  potentisl circular import
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
    
    This class loads the base model jealk/llm2vec-scandi-mntp-v2
    (which already has MNTP merged) and applies the supervised adapter
    jealk/TTC-L2V-supervised-2.
    """
    
    def __init__(self, base_model_name: str = "jealk/llm2vec-scandi-mntp-v2", 
                 adapter_name: str = "jealk/TTC-L2V-supervised-2", 
                 device: Optional[str] = None):
        """
        Initialize the embedding model.
        
        Args:
            base_model_name: Name of the base model (default: jealk/llm2vec-scandi-mntp-v2)
            adapter_name: Name of the adapter (default: jealk/TTC-L2V-supervised-2)
        """
        # #region agent log
        import json, sys; open('/data/users/spreitz/LagRag/.cursor/debug.log', 'a').write(json.dumps({'sessionId':'debug-session','runId':'run1','hypothesisId':'B','location':'embedder.py:45','message':'TTCEmbeddings.__init__ entry, about to import DEFAULT_MODEL_CONFIG','data':{'config_in_sys_modules':'src.config' in sys.modules},'timestamp':__import__('time').time()*1000})+'\n')
        # #endregion
        from src.config import DEFAULT_MODEL_CONFIG
        # #region agent log
        try: open('/data/users/spreitz/LagRag/.cursor/debug.log', 'a').write(json.dumps({'sessionId':'debug-session','runId':'run1','hypothesisId':'B','location':'embedder.py:46','message':'DEFAULT_MODEL_CONFIG imported in __init__','data':{'type':type(DEFAULT_MODEL_CONFIG).__name__ if 'DEFAULT_MODEL_CONFIG' in locals() else 'NOT_IN_LOCALS','in_locals':'DEFAULT_MODEL_CONFIG' in locals()},'timestamp':__import__('time').time()*1000})+'\n')
        except Exception as e: open('/data/users/spreitz/LagRag/.cursor/debug.log', 'a').write(json.dumps({'sessionId':'debug-session','runId':'run1','hypothesisId':'B','location':'embedder.py:46','message':'ERROR after import DEFAULT_MODEL_CONFIG','data':{'error':str(e)},'timestamp':__import__('time').time()*1000})+'\n')
        # #endregion

        self.base_model_name = base_model_name
        self.adapter_name = adapter_name
        

        # Use specified device
        if device is None:
            # #region agent log
            try: open('/data/users/spreitz/LagRag/.cursor/debug.log', 'a').write(json.dumps({'sessionId':'debug-session','runId':'run1','hypothesisId':'A','location':'embedder.py:53','message':'About to access DEFAULT_MODEL_CONFIG.embedding_device','data':{'in_locals':'DEFAULT_MODEL_CONFIG' in locals(),'in_globals':'DEFAULT_MODEL_CONFIG' in globals()},'timestamp':__import__('time').time()*1000})+'\n')
            except Exception as e: open('/data/users/spreitz/LagRag/.cursor/debug.log', 'a').write(json.dumps({'sessionId':'debug-session','runId':'run1','hypothesisId':'A','location':'embedder.py:53','message':'ERROR before accessing DEFAULT_MODEL_CONFIG','data':{'error':str(e)},'timestamp':__import__('time').time()*1000})+'\n')
            # #endregion
            device = DEFAULT_MODEL_CONFIG.embedding_device
        
        self.device = device
        
        # Early GPU memory check
        if torch.cuda.is_available() and "cuda" in str(self.device):
            device_idx = int(self.device.split(":")[1]) if ":" in str(self.device) else 0
            allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)  # GB
            total = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)  # GB
            free = total - reserved
            
            print(f"\n[Embedder Init] GPU {device_idx} Memory: {free:.2f} GB free / {total:.2f} GB total")
            if free < 2.0:  # Less than 2GB free
                print(f"WARNING: Very little GPU memory free ({free:.2f} GB). Model loading may fail.")
        self.device = torch.device(device)
        
        # Early GPU memory check
        if torch.cuda.is_available() and self.device.type == "cuda":
            # Get device index from torch.device object
            device_idx = self.device.index if self.device.index is not None else 0
            allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)  # GB
            total = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)  # GB
            free = total - reserved
            
            print(f"\n[Embedder Init] GPU {device_idx} Memory Check:")
            print(f"  Total: {total:.2f} GB | Reserved: {reserved:.2f} GB | Free: {free:.2f} GB")
            if free < 2.0:  # Less than 2GB free
                print(f"  ⚠️  WARNING: Very little GPU memory free ({free:.2f} GB). Model loading may fail.")
                print(f"  💡 Check other processes: nvidia-smi")

        # Determine dtype based on device
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        
        print(f"Loading base model: {base_model_name}")
        print(f"Device: {self.device}, Dtype: {self.dtype}")
        
        # Load base model with tokenizer and config
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        # Load config first and modify it
        config = AutoConfig.from_pretrained(base_model_name)


        # # Remove adapter configs to prevent auto-loading
        # if hasattr(config, 'adapter_config') or hasattr(config, 'peft_config'):
        #     # Clear any adapter-related configs
        #     pass

        # Load base model (MNTP is already merged into jealk/llm2vec-scandi-mntp-v2)
        base_model = AutoModel.from_pretrained(
            base_model_name,
            config=config,
            torch_dtype=self.dtype,
            trust_remote_code=True
        )

        # If the model object has a peft_config attribute, remove it to avoid stacked adapters warnings
        if hasattr(base_model, "peft_config"):
            print("WARNING: base_model already has peft_config; deleting to avoid stacked adapters")
            try:
                delattr(base_model, "peft_config")
            except Exception as e:
                print(f"WARNING: could not delete peft_config: {e}")
        
        # Load the supervised adapter on top of the base model
        # Note: We skip trying to load MNTP separately to avoid double-PEFT issues.
        # The base model jealk/llm2vec-scandi-mntp-v2 already has MNTP merged.
        print(f"Loading supervised adapter: {adapter_name}")
        try:
            model_with_supervised = PeftModel.from_pretrained(
                base_model,
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
        # Check GPU memory before moving
        if torch.cuda.is_available() and self.device.type == "cuda":
            # Get device index from torch.device object
            device_idx = self.device.index if self.device.index is not None else 0
            allocated = torch.cuda.memory_allocated(device_idx) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(device_idx) / (1024**3)  # GB
            total = torch.cuda.get_device_properties(device_idx).total_memory / (1024**3)  # GB
            free = total - reserved
            
            print(f"\n{'='*60}")
            print(f"GPU Memory Status (Device {device_idx}):")
            print(f"  Total: {total:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Free: {free:.2f} GB")
            print(f"{'='*60}")
            
            # Estimate model size (rough estimate for Llama-3-8B in bfloat16: ~16GB)
            estimated_model_size_gb = 16.0
            if free < estimated_model_size_gb:
                print(f"WARNING: Only {free:.2f} GB free, but model needs ~{estimated_model_size_gb} GB")
                print(f"This will likely fail. Consider:")
                print(f"  1. Kill other processes using GPU: nvidia-smi")
                print(f"  2. Use CPU instead: device='cpu'")
                print(f"  3. Request exclusive GPU access in SLURM")
        
        # #region agent log
        import json, time; 
        try:
            log_data = {'sessionId':'debug-session','runId':'run1','hypothesisId':'A,B','location':'embedder.py:121','message':'About to move embedding model to device','data':{'target_device':str(self.device),'gpu_count':torch.cuda.device_count() if torch.cuda.is_available() else 0},'timestamp':time.time()*1000}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    log_data['data'][f'gpu_{i}_before_mb'] = round(torch.cuda.memory_allocated(i) / (1024**2), 2)
            open('/data/users/spreitz/LagRag/.cursor/debug.log', 'a').write(json.dumps(log_data) + '\n')
        except: pass
        # #endregion
        
        print(f"Moving model to {self.device}...")
        try:
            model_final = model_final.to(self.device)
            print(f"✓ Model successfully moved to {self.device}")
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n{'='*60}")
            print(f"ERROR: CUDA Out of Memory!")
            print(f"{'='*60}")
            print(f"Failed to move model to {self.device}")
            print(f"Error: {e}")
            print(f"\nSolutions:")
            print(f"1. Check what's using GPU: nvidia-smi")
            print(f"2. Kill other processes: nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs kill -9")
            print(f"3. Use CPU instead (slower): Set device='cpu' in config")
            print(f"4. Request exclusive GPU in SLURM: #SBATCH --gres=gpu:L40s:1")
            raise
        # #region agent log
        try:
            log_data = {'sessionId':'debug-session','runId':'run1','hypothesisId':'A,B','location':'embedder.py:124','message':'After moving embedding model to device','data':{'target_device':str(self.device),'model_device':str(next(model_final.parameters()).device)},'timestamp':time.time()*1000}
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    log_data['data'][f'gpu_{i}_after_mb'] = round(torch.cuda.memory_allocated(i) / (1024**2), 2)
            open('/data/users/spreitz/LagRag/.cursor/debug.log', 'a').write(json.dumps(log_data) + '\n')
        except: pass
        # #endregion
        
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
            List of embedding vectors (each is a list of float32 values)
        """
        total = len(texts)
        print(f"Embedding {total} documents in batches of 32...")
        
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=True,  # Enable progress bar
                convert_to_numpy=True
            )
        
        # Convert to numpy array if not already
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        # Ensure float32 dtype
        embeddings = embeddings.astype(np.float32)
        
        # Clean any NaN or Inf values BEFORE normalization (replace with zeros)
        # This prevents NaN/Inf from propagating through normalization
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Safe L2 normalization: divide by (norm + epsilon) to avoid division by zero
        # For cosine similarity, embeddings must be L2-normalized
        # Expected norm after normalization: ≈ 1.0 (within numerical precision)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        epsilon = 1e-12
        embeddings = embeddings / (norms + epsilon)
        
        # Clean NaN/Inf values AGAIN after normalization (defensive programming)
        # This guarantees no NaN/Inf can ever reach Chroma
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Sanity check: verify all norms are finite after normalization
        final_norms = np.linalg.norm(embeddings, axis=1)
        assert np.all(np.isfinite(final_norms)), "Embeddings contain non-finite values after normalization"
        # Expected norm ≈ 1.0 (within numerical precision due to epsilon)
        # In practice, norms will be slightly less than 1.0 due to epsilon in denominator
        
        # Ensure still float32 after normalization and cleaning
        embeddings = embeddings.astype(np.float32)
        
        # Convert to plain Python list of floats (float32)
        # This ensures Chroma receives clean, normalized, float32 embeddings
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


def get_embedding_function(
    base_model_name: Optional[str] = None,
    adapter_name: Optional[str] = None,
    device: Optional[str] = None
) -> Embeddings:
    """Get a LangChain-compatible embedding function."""
    # #region agent log
    import json, sys; open('/data/users/spreitz/LagRag/.cursor/debug.log', 'a').write(json.dumps({'sessionId':'debug-session','runId':'run1','hypothesisId':'D','location':'embedder.py:194','message':'get_embedding_function entry, about to import DEFAULT_MODEL_CONFIG','data':{'config_in_sys_modules':'src.config' in sys.modules},'timestamp':__import__('time').time()*1000})+'\n')
    # #endregion
    from src.config import DEFAULT_MODEL_CONFIG
    # #region agent log
    try: open('/data/users/spreitz/LagRag/.cursor/debug.log', 'a').write(json.dumps({'sessionId':'debug-session','runId':'run1','hypothesisId':'D','location':'embedder.py:195','message':'DEFAULT_MODEL_CONFIG imported in get_embedding_function','data':{'in_locals':'DEFAULT_MODEL_CONFIG' in locals()},'timestamp':__import__('time').time()*1000})+'\n')
    except Exception as e: open('/data/users/spreitz/LagRag/.cursor/debug.log', 'a').write(json.dumps({'sessionId':'debug-session','runId':'run1','hypothesisId':'D','location':'embedder.py:195','message':'ERROR after import in get_embedding_function','data':{'error':str(e)},'timestamp':__import__('time').time()*1000})+'\n')
    # #endregion
    
    if base_model_name is None:
        base_model_name = DEFAULT_MODEL_CONFIG.embedding_base_model
    if adapter_name is None:
        adapter_name = DEFAULT_MODEL_CONFIG.embedding_adapter
    if device is None:
        device = DEFAULT_MODEL_CONFIG.embedding_device
    
    return TTCEmbeddings(base_model_name=base_model_name, adapter_name=adapter_name, device=device)


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
