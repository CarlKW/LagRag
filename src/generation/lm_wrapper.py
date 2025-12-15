"""
Wrapper and singleton accessor for the local generation LM.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LocalHFModel:
    """
    Thin wrapper around a local HuggingFace causal language model.

    Usage:
        lm = LocalHFModel("gpt2")
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

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
        ).to(self.device)
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
    model_name_or_path: str = "gpt2",  # TODO: set your real model
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    **default_generate_kwargs: Any,
) -> LocalHFModel:
    """
    Lazily create and return a singleton LocalHFModel.

    Call this once at startup or wherever you wire the pipeline together.
    """
    global _LM_INSTANCE
    if _LM_INSTANCE is None:
        _LM_INSTANCE = LocalHFModel(
            model_name_or_path=model_name_or_path,
            device=device,
            dtype=dtype,
            **default_generate_kwargs,
        )
    return _LM_INSTANCE