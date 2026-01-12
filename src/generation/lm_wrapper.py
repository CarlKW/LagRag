"""
Wrapper and singleton accessor for the local generation LM.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import DEFAULT_MODEL_CONFIG

logger = logging.getLogger(__name__)


class LocalHFModel:
    """
    Thin wrapper around a local HuggingFace causal language model.

    Usage:
        lm = LocalHFModel("AI-Sweden-Models/Llama-3-8B-instruct")
        out = lm.generate("Hello", max_new_tokens=32)
        
    Or use get_local_lm() which uses DEFAULT_MODEL_CONFIG.generation_model:
        lm = get_local_lm()
        out = lm.generate("Hello", max_new_tokens=32)
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        **default_generate_kwargs: Any,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        logger.info("Loading generation model %s on %s", model_name_or_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Use device_map to load directly on target device, avoiding temporary GPU memory usage
        # device_map={"": str(self.device)} loads all layers on the specified device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map={"": str(self.device)},  # Load directly on target device
        )
        self.model.eval()

        # Default generate kwargs (e.g. top_p, repetition_penalty, etc.)
        self.default_generate_kwargs: Dict[str, Any] = default_generate_kwargs

        # Ensure pad_token_id is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        **overrides: Any,
    ) -> str:
        """
        Generate a completion for the given prompt.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.device)

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": max(temperature, 0.0),
            "do_sample": temperature > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            **self.default_generate_kwargs,
            **overrides,
        }

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **generate_kwargs,
            )

        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Strip prompt if model echoes it
        if full_text.startswith(prompt):
            generated = full_text[len(prompt) :]
        else:
            generated = full_text

        return generated.strip()


# -----------------------------------------------------------------------------
# Singleton accessor (one LM instance for the whole process)
# -----------------------------------------------------------------------------

_LM_INSTANCE: Optional[LocalHFModel] = None


def get_local_lm(
    model_name_or_path: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    **default_generate_kwargs: Any,
) -> LocalHFModel:
    """
    Lazily create and return a singleton LocalHFModel.

    If model_name_or_path is None, uses DEFAULT_MODEL_CONFIG.generation_model.
    Call this once at startup or wherever you wire the pipeline together.
    """
    global _LM_INSTANCE
    if _LM_INSTANCE is None:
        if model_name_or_path is None:
            model_name_or_path = DEFAULT_MODEL_CONFIG.generation_model
        _LM_INSTANCE = LocalHFModel(
            model_name_or_path=model_name_or_path,
            device=device,
            dtype=dtype,
            **default_generate_kwargs,
        )
    return _LM_INSTANCE