"""
Centralized configuration for all models and pipeline settings.

This module provides a single source of truth for all model names, paths,
and other configuration values used throughout the pipeline.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for all models used in the pipeline."""
    
    # Generation model (LLM for answer generation)
    generation_model: str = "AI-Sweden-Models/Llama-3-8B-instruct"
    
    # Device configuration
    embedding_device: str = "cuda:0"  # Embedding model on GPU 0
    generation_device: str = "cuda:1"  # Generation model on GPU 1
    reranker_device: str = "cuda:0"    # Reranker on GPU 0 (smaller, can share)
    

    # Embedding model configuration
    embedding_base_model: str = "jealk/llm2vec-scandi-mntp-v2"
    embedding_adapter: str = "jealk/TTC-L2V-supervised-2"
    
    # Reranker model
    reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual"
    
    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 0.1
    
    # Reranker batch size
    reranker_batch_size: int = 16


@dataclass
class PipelineConfig:
    """Configuration for pipeline behavior."""
    
    # ChromaDB settings
    chroma_db_path: str = "./chroma_db_pipeline"
    collection_name: str = "sfs_paragraphs"
    
    # Retrieval settings
    k_initial: int = 50
    k_final: int = 10
    max_retrieval_rounds: int = 2
    
    # Thresholds
    high_threshold: float = 0.75
    low_threshold: float = 0.40
    
    # Retriever type
    retriever_type: str = "reranking"  # "reranking" or "basic"


# Global default configuration instances
# Override these or pass custom config to functions as needed
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_PIPELINE_CONFIG = PipelineConfig()


def get_model_config(
    generation_model: Optional[str] = None,
    embedding_base_model: Optional[str] = None,
    embedding_adapter: Optional[str] = None,
    reranker_model: Optional[str] = None,
    **kwargs
) -> ModelConfig:
    """
    Get a ModelConfig instance with optional overrides.
    
    Args:
        generation_model: Override generation model name
        embedding_base_model: Override embedding base model
        embedding_adapter: Override embedding adapter
        reranker_model: Override reranker model
        **kwargs: Additional overrides for other ModelConfig fields
        
    Returns:
        ModelConfig instance with specified overrides applied
    """
    config = ModelConfig()
    
    if generation_model is not None:
        config.generation_model = generation_model
    if embedding_base_model is not None:
        config.embedding_base_model = embedding_base_model
    if embedding_adapter is not None:
        config.embedding_adapter = embedding_adapter
    if reranker_model is not None:
        config.reranker_model = reranker_model
    
    # Apply any other overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def get_pipeline_config(**kwargs) -> PipelineConfig:
    """
    Get a PipelineConfig instance with optional overrides.
    
    Args:
        **kwargs: Overrides for PipelineConfig fields
        
    Returns:
        PipelineConfig instance with specified overrides applied
    """
    config = PipelineConfig()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config
