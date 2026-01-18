
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional
from src.generation.lm_wrapper import LocalHFModel, get_local_lm
from src.retrieval.retriever import RerankingRetriever
from src.generation.adapters import retriever_results_to_context_chunks 


logger = logging.getLogger(__name__)

# skip self eval: if True, score_answer() returns 1.0 without calling the LM
SKIP_SELF_EVALUATION = True


@dataclass
class ContextChunk:

    id: Any
    text: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    NEED_MORE_CONTEXT = "need_more_context"
    CANNOT_ANSWER = "cannot_answer"
    CANNOT_ANSWER_CANDIDATE = "cannot_answer_candidate"


@dataclass
class GenerationResult:
    answer: str
    score: float
    status: AnswerStatus
    used_chunks: List[ContextChunk] = field(default_factory=list)
    num_retrieval_rounds: int = 0
    reason: str = ""


# Format context chunks as a numbered list of snippets
def _format_context_chunks(context_chunks: Iterable[ContextChunk]) -> str:
    lines: List[str] = []
    for i, chunk in enumerate(context_chunks):
        lines.append(f"[{i}] {chunk.text.strip()}")
    return "\n\n".join(lines) if lines else "No context provided."


def build_answer_prompt(
    query: str,
    context_chunks: List[ContextChunk],
) -> str:
    context_text = _format_context_chunks(context_chunks)

    prompt = (
        "Du är en precis assistent. Använd ENDAST informationen i kontexten för att svara.\n"
        "Om svaret inte stöds fullt ut av kontexten, säg exakt:\n"
        "\"Jag är ledsen, jag kan inte svara på det baserat på min kunskap.\"\n\n"
        "Kontext:\n"
        f"{context_text}\n\n"
        "Fråga:\n"
        f"{query}\n\n"
        "Svar:\n"
    )
    return prompt


# prompt for using the LM as a judge of answer quality
def build_scoring_prompt(
    query: str,
    context_chunks: List[ContextChunk],
    answer: str,
) -> str:
    context_text = _format_context_chunks(context_chunks)
    prompt = (
        "Du utvärderar ett svar givet kontext och en fråga. Utgå ifrån från den givna kontexten för att svara på frågan \n\n"
        "Kontext:\n"
        f"{context_text}\n\n"
        "Fråga:\n"
        f"{query}\n\n"
        "Svar:\n"
        f"{answer}\n\n"
        "Utvärdera hur väl svaret stöds av ENDAST KONTEXTEN.\n"
        "- 1.0 = fullt stött, precist, inga hallucinationer\n"
        "- 0.5 = delvis stött eller något otydligt\n"
        "- 0.0 = inte stött eller tydligt hallucinerat\n\n"
        "Svara med ENDAST ett enda tal mellan 0.0 och 1.0.\n"
    )
    return prompt



# make a candidate answer for the query using the provided context
def generate_raw_answer(
    query: str,
    context_chunks: List[ContextChunk],
    lm: LocalHFModel,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    prompt = build_answer_prompt(query=query, context_chunks=context_chunks)
    answer = lm.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return answer.strip()


_FLOAT_REGEX = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")   ## !!!Look this up 


# parse first floatingpoint number from the model output and clamp it to [0, 1]
def _parse_score(text: str, default: float = 0.0) -> float:
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


# Score a candidate answer using the LM as a judge, returns float in [0.0, 1.0]
def score_answer(
    query: str,
    context_chunks: List[ContextChunk],
    answer: str,
    lm: LocalHFModel,
    max_new_tokens: int = 16,
) -> float:
    if SKIP_SELF_EVALUATION:
        return 1.0

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


class RAGGenerator:
    def __init__(
        self,
        lm: LocalHFModel,   ### Our downloaded model
        retriever: Optional[RerankingRetriever] = None,   # The reetriever from our pipeline
        k: int = 10,  # tune
        max_retrieval_rounds: int = 0,  # tune
        high_threshold: float = 0.75,  # tune
        low_threshold: float = 0.40,  # tune 
        canonical_cannot_answer_text: str = (
            "Jag är ledsen, jag har ingen information om det."
        ),
    ) -> None:
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
                # another retrieval round to try to improve the context.
                # PART OF SELF EVAL, SKIPPED IN FINAL VERSION
                rounds += 1
                logger.info(
                    "Active retrieval round %d for query %r (status=%s, score=%.3f)",
                    rounds,
                    query,
                    status.value,
                    score,
                )
                retriever_results = self.retriever.retrieve(query)
                context = retriever_results_to_context_chunks(retriever_results)
                continue

            # No more retrieval possible; decide final outcome.
            if best_candidate and best_candidate.score >= self.low_threshold:
                best_candidate.status = AnswerStatus.ANSWERED
                best_candidate.reason = "medium_score_no_more_retrieval"
                return best_candidate

            # All candidates are shit and we cannot retrieve more: fallback.
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


