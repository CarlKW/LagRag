"""
Generation module for the RAG pipeline with active retrieval.

This module provides:

- `ContextChunk`: small container for retrieved text snippets.
- `GenerationResult`: structured output for the generation step.
- `LocalHFModel`: thin wrapper around a local HuggingFace causal LM.
- Scoring utilities that use the same LM as a judge.
- `RAGGenerator`: orchestration class that performs:
    1) answer generation from query + context,
    2) answer scoring,
    3) optional active retrieval if the answer is not good enough,
    4) a safe fallback when the question cannot be answered.

Typical usage (pseudocode):

    from generation.genAI import (
        ContextChunk,
        LocalHFModel,
        RAGGenerator,
    )

    lm = LocalHFModel(model_name_or_path="gpt2")  # example model
    generator = RAGGenerator(lm=lm, retriever=my_retriever)
    retriever = RerankingRetriever(database)

    retriver = RerankingRetriever(persistent dir)
    chunks = my_retriever("What is RAG?", k=10)
    result = generator.generate_answer(query="What is RAG?", initial_context=chunks)

    print(result.status)
    print(result.answer)

The behaviour is controlled by thresholds:
- If the LM judge score is high, the answer is returned to the user.
- If the score is medium or low, the generator can perform active retrieval
  (if a retriever is supplied) up to a configurable number of rounds.
- If, after several attempts, scores remain low, a fixed "cannot answer"
  message is returned instead of hallucinating.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional
from src.generation.lm_wrapper import LocalHFModel, get_local_lm
from src.generator.retriever import RerankingRetriever
from src.generation.adapters import retriever_results_to_context_chunks 


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures and interfaces
# ---------------------------------------------------------------------------


@dataclass
class ContextChunk:
    """
    Small container for a retrieved context snippet.

    Attributes
    ----------
    id:
        Identifier for the chunk (e.g. document ID + offset).
    text:
        Plain-text content for this chunk.
    score:
        Optional relevance score from the retriever (higher is better).
    metadata:
        Optional metadata associated with the chunk (e.g. source file, page).
    """

    id: Any
    text: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnswerStatus(str, Enum):
    """
    High-level status of the generation outcome.

    Values
    ------
    ANSWERED:
        A high-confidence answer that can be shown to the user.
    NEED_MORE_CONTEXT:
        The answer is mediocre; more retrieval may improve it.
    CANNOT_ANSWER:
        The model estimates that the question cannot be answered
        from the available knowledge.
    CANNOT_ANSWER_CANDIDATE:
        Internal label during evaluation before we exhaust retrieval.
    """

    ANSWERED = "answered"
    NEED_MORE_CONTEXT = "need_more_context"
    CANNOT_ANSWER = "cannot_answer"
    CANNOT_ANSWER_CANDIDATE = "cannot_answer_candidate"


@dataclass
class GenerationResult:
    """
    Structured result of the RAG generation step.
    """

    answer: str
    score: float
    status: AnswerStatus
    used_chunks: List[ContextChunk] = field(default_factory=list)
    num_retrieval_rounds: int = 0
    reason: str = ""



# ---------------------------------------------------------------------------
# Local HuggingFace model wrapper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Prompt construction helpers
# ---------------------------------------------------------------------------


def _format_context_chunks(context_chunks: Iterable[ContextChunk]) -> str:
    """
    Format context chunks as a numbered list of snippets.
    """

    lines: List[str] = []
    for i, chunk in enumerate(context_chunks):
        lines.append(f"[{i}] {chunk.text.strip()}")
    return "\n\n".join(lines) if lines else "No context provided."


def build_answer_prompt(
    query: str,
    context_chunks: List[ContextChunk],
) -> str:
    """
    Build a prompt instructing the LM to answer using only the context.
    """

    context_text = _format_context_chunks(context_chunks)

    prompt = (
        "You are a precise assistant. Use ONLY the information in the context to answer.\n"
        "If the answer is not fully supported by the context, say exactly:\n"
        "\"I'm sorry, I can't answer that based on my knowledge.\"\n\n"
        "Context:\n"
        f"{context_text}\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Answer:\n"
    )
    return prompt


def build_scoring_prompt(
    query: str,
    context_chunks: List[ContextChunk],
    answer: str,
) -> str:
    """
    Build a prompt for using the LM as a judge of answer quality.
    """

    context_text = _format_context_chunks(context_chunks)
    prompt = (
        "You are evaluating an answer given some context and a question.\n\n"
        "Context:\n"
        f"{context_text}\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Answer:\n"
        f"{answer}\n\n"
        "Evaluate how well the answer is supported by the CONTEXT ONLY.\n"
        "- 1.0 = fully supported, precise, no hallucinations\n"
        "- 0.5 = partially supported or somewhat vague\n"
        "- 0.0 = not supported or clearly hallucinated\n\n"
        "Respond with ONLY a single number between 0.0 and 1.0.\n"
    )
    return prompt


# ---------------------------------------------------------------------------
# Scoring and evaluation
# ---------------------------------------------------------------------------


def generate_raw_answer(
    query: str,
    context_chunks: List[ContextChunk],
    lm: LocalHFModel,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """
    Generate a candidate answer for the query using the provided context.
    """

    prompt = build_answer_prompt(query=query, context_chunks=context_chunks)
    answer = lm.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return answer.strip()


_FLOAT_REGEX = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")   ## !!!Look this up 


def _parse_score(text: str, default: float = 0.0) -> float:
    """
    Parse the first floating-point number from the model output and clamp it to [0, 1].
    """

    match = _FLOAT_REGEX.search(text)
    if not match:
        logger.warning("Could not parse score from LM output: %r", text)
        return default

    try:
        value = float(match.group(0))
    except ValueError:
        logger.warning("Failed to convert parsed score %r to float", match.group(0))
        return default

    return max(0.0, min(1.0, value))


def score_answer(
    query: str,
    context_chunks: List[ContextChunk],
    answer: str,
    lm: LocalHFModel,
    max_new_tokens: int = 16,
) -> float:
    """
    Score a candidate answer using the LM as a judge.

    Returns a float in [0.0, 1.0].
    """

    prompt = build_scoring_prompt(query=query, context_chunks=context_chunks, answer=answer)
    raw_score = lm.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    score = _parse_score(raw_score, default=0.0)
    return score


def evaluate_answer(
    query: str,
    context_chunks: List[ContextChunk],
    answer: str,
    lm: LocalHFModel,
    high_threshold: float = 0.75,
    low_threshold: float = 0.40,
) -> tuple[AnswerStatus, float]:
    """
    Evaluate the answer and map the score to a status.

    Returns
    -------
    status:
        One of AnswerStatus.ANSWERED, AnswerStatus.NEED_MORE_CONTEXT,
        AnswerStatus.CANNOT_ANSWER_CANDIDATE.
    score:
        The underlying numeric score in [0.0, 1.0].
    """

    score = score_answer(
        query=query,
        context_chunks=context_chunks,
        answer=answer,
        lm=lm,
    )

    if score >= high_threshold:
        status = AnswerStatus.ANSWERED
    elif score >= low_threshold:
        status = AnswerStatus.NEED_MORE_CONTEXT
    else:
        status = AnswerStatus.CANNOT_ANSWER_CANDIDATE

    return status, score


# ---------------------------------------------------------------------------
# RAG generator with active retrieval
# ---------------------------------------------------------------------------


class RAGGenerator:
    """
    Orchestrates RAG generation with optional active retrieval.

    The main entry point is :meth:`generate_answer`.
    """

    def __init__(
        self,
        lm: LocalHFModel,   ### Our downloaded model
        retriever: Optional[RerankingRetriever] = None,   # The reetriever from Carl 
        k: int = 10,  # tune
        max_retrieval_rounds: int = 2,  # tune
        high_threshold: float = 0.75,  # tune
        low_threshold: float = 0.40,  # tune 
        canonical_cannot_answer_text: str = (
            "I'm sorry I dont have information on that."
        ),
    ) -> None:
        """
        Parameters
        ----------
        lm:
            Local language model used for both answering and scoring.
        retriever:
            Optional retrieval callback used for active retrieval.
        k:
            Number of chunks to request per retrieval call.
        max_retrieval_rounds:
            Maximum number of additional retrieval rounds.
        high_threshold:
            Score threshold above which answers are accepted as final.
        low_threshold:
            Score below which answers are considered bad and will either
            trigger retrieval (if possible) or produce a cannot-answer result.
        canonical_cannot_answer_text:
            Message returned to the user when the system determines that the
            question cannot be answered from its knowledge base.
        """

        self.lm = lm
        self.retriever = retriever
        self.k = k
        self.max_retrieval_rounds = max_retrieval_rounds
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.canonical_cannot_answer_text = canonical_cannot_answer_text

    def generate_answer(
        self,
        query: str,
        initial_context: List[ContextChunk],
    ) -> GenerationResult:
        """
        Generate an answer for the query, performing active retrieval as needed.
        """

        context = initial_context
        rounds = 0
        best_candidate: Optional[GenerationResult] = None

        while True:
            answer = generate_raw_answer(
                query=query,
                context_chunks=context,
                lm=self.lm,
            )

            status, score = evaluate_answer(
                query=query,
                context_chunks=context,
                answer=answer,
                lm=self.lm,
                high_threshold=self.high_threshold,
                low_threshold=self.low_threshold,
            )

            candidate = GenerationResult(
                answer=answer,
                score=score,
                status=status,
                used_chunks=list(context),
                num_retrieval_rounds=rounds,
                reason="intermediate_candidate",  ## !!!Look this up 
            )

            if best_candidate is None or candidate.score > best_candidate.score:
                best_candidate = candidate

            # High-confidence answer: return to user.
            if status is AnswerStatus.ANSWERED:
                candidate.status = AnswerStatus.ANSWERED
                candidate.reason = "score_above_high_threshold"
                return candidate

            # If we reach here, status is either NEED_MORE_CONTEXT or CANNOT_ANSWER_CANDIDATE.
            can_retrieve_more = self.retriever is not None and rounds < self.max_retrieval_rounds

            if can_retrieve_more:
                # Perform another retrieval round to try to improve the context.
                rounds += 1
                logger.info(
                    "Active retrieval round %d for query %r (status=%s, score=%.3f)",
                    rounds,
                    query,
                    status.value,
                    score,
                )
                # Retrieve results (retriever.retrieve() only takes query, k is set in constructor)
                retriever_results = self.retriever.retrieve(query)
                # Convert dict results to ContextChunk format
                context = retriever_results_to_context_chunks(retriever_results)
                continue

            # No more retrieval possible; decide final outcome.
            if best_candidate and best_candidate.score >= self.low_threshold:
                # Use the best medium-quality answer we have.
                best_candidate.status = AnswerStatus.ANSWERED
                best_candidate.reason = "medium_score_no_more_retrieval"
                return best_candidate

            # All candidates are poor and we cannot retrieve more: fallback.
            return GenerationResult(
                answer=self.canonical_cannot_answer_text,
                score=best_candidate.score if best_candidate else 0.0,
                status=AnswerStatus.CANNOT_ANSWER,
                used_chunks=list(context),
                num_retrieval_rounds=rounds,
                reason="low_score_after_max_rounds",
            )


__all__ = [
    "ContextChunk",
    "AnswerStatus",
    "GenerationResult",
    "generate_raw_answer",
    "score_answer",
    "evaluate_answer",
    "RAGGenerator",
    "LocalHFModel",
]


