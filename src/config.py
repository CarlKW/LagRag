from dataclasses import dataclass
from typing import Optional
"""
One main configuration file for changing models or number of retrieved chunks.
"""

@dataclass
class ModelConfig:
    generation_model: str = "microsoft/Phi-3.5-mini-instruct"
    
    embedding_device: str = "cuda:0"
    generation_device: str = "cuda:0"
    reranker_device: str = "cuda:0"
    
    embedding_base_model: str = "jealk/llm2vec-scandi-mntp-v2"
    embedding_adapter: str = "jealk/TTC-L2V-supervised-2"
    
    reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    
    max_new_tokens: int = 256
    temperature: float = 0.1
    
    reranker_batch_size: int = 16


@dataclass
class PipelineConfig:
    chroma_db_path: str = "./chroma_db_pipeline"
    collection_name: str = "sfs_paragraphs"
    
    k_initial: int = 50
    k_final: int = 3
    max_retrieval_rounds: int = 2
    
    high_threshold: float = 0.75
    low_threshold: float = 0.40
    
    retriever_type: str = "reranking"


DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_PIPELINE_CONFIG = PipelineConfig()
