"""
LLM Evaluation Toolkit
=======================
Benchmark and compare LLM responses using automated scoring metrics.
Evaluate response quality, factual consistency, relevance, and safety
without needing human annotators.

Metrics implemented:
    - Semantic similarity   (sentence-transformers cosine similarity)
    - ROUGE-L               (longest common subsequence overlap)
    - Faithfulness score    (does response stay grounded in context?)
    - Relevance score       (does response address the question?)
    - Toxicity flag         (simple keyword-based safety check)
    - Latency               (wall-clock response time)

Supports multiple LLM backends:
    - HuggingFace (free, local)
    - OpenAI (if OPENAI_API_KEY set)
    - Mock (deterministic, for testing)

Author: Nishtha Mehta
"""

import os
import re
import json
import time
import logging
import hashlib
import asyncio
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from datetime import datetime
from collections import defaultdict

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline as hf_pipeline
except ImportError:
    os.system("pip install sentence-transformers transformers torch -q")
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline as hf_pipeline


# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("llm_eval")

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"
TOXIC_KEYWORDS = {
    "hate", "kill", "violence", "harm", "illegal", "dangerous",
    "weapon", "bomb", "exploit", "attack", "threat"
}


# ── Data Models ───────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    """A single evaluation test case."""
    id: str
    question: str
    reference_answer: str           # ground truth
    context: str = ""               # optional grounding context
    category: str = "general"       # for grouped reporting


@dataclass
class ModelResponse:
    """One model's response to one eval case."""
    case_id: str
    model_name: str
    response: str
    latency_ms: float
    tokens_approx: int = 0          # rough token count
    error: Optional[str] = None


@dataclass
class EvalScore:
    """All scores for one (model, case) pair."""
    case_id: str
    model_name: str
    question: str
    reference: str
    response: str

    # Scores (all in [0, 1])
    semantic_similarity: float = 0.0
    rouge_l: float = 0.0
    faithfulness: float = 0.0
    relevance: float = 0.0
    is_toxic: bool = False
    latency_ms: float = 0.0

    @property
    def composite_score(self) -> float:
        """Weighted composite score."""
        if self.is_toxic:
            return 0.0
        return (
            0.35 * self.semantic_similarity +
            0.20 * self.rouge_l +
            0.25 * self.faithfulness +
            0.20 * self.relevance
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["composite_score"] = round(self.composite_score, 4)
        return d


@dataclass
class ModelReport:
    """Aggregated evaluation results for one model."""
    model_name: str
    scores: list[EvalScore] = field(default_factory=list)

    @property
    def avg_composite(self) -> float:
        return np.mean([s.composite_score for s in self.scores]) if self.scores else 0.0

    @property
    def avg_semantic(self) -> float:
        return np.mean([s.semantic_similarity for s in self.scores]) if self.scores else 0.0

    @property
    def avg_rouge(self) -> float:
        return np.mean([s.rouge_l for s in self.scores]) if self.scores else 0.0

    @property
    def avg_faithfulness(self) -> float:
        return np.mean([s.faithfulness for s in self.scores]) if self.scores else 0.0

    @property
    def avg_latency(self) -> float:
        return np.mean([s.latency_ms for s in self.scores]) if self.scores else 0.0

    @property
    def toxicity_rate(self) -> float:
        toxic = sum(1 for s in self.scores if s.is_toxic)
        return toxic / len(self.scores) if self.scores else 0.0

    def by_category(self) -> dict:
        cats = defaultdict(list)
        for s in self.scores:
            cats[s.question].append(s.composite_score)
        return {cat: round(np.mean(scores), 4) for cat, scores in cats.items()}

    def summary(self) -> dict:
        return {
            "model": self.model_name,
            "cases_evaluated": len(self.scores),
            "avg_composite_score": round(self.avg_composite, 4),
            "avg_semantic_similarity": round(self.avg_semantic, 4),
            "avg_rouge_l": round(self.avg_rouge, 4),
            "avg_faithfulness": round(self.avg_faithfulness, 4),
            "avg_latency_ms": round(self.avg_latency, 1),
            "toxicity_rate": round(self.toxicity_rate, 4),
        }


# ── Metrics ───────────────────────────────────────────────────────────────────

class MetricsEngine:
    """
    Compute all evaluation metrics.
    Uses sentence-transformers for semantic scores (free, local).
    """

    def __init__(self):
        log.info("Loading embedding model for metrics...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        log.info("✓ Metrics engine ready")

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / (norms + 1e-9)

    def semantic_similarity(self, response: str, reference: str) -> float:
        """Cosine similarity between response and reference embeddings."""
        if not response.strip() or not reference.strip():
            return 0.0
        vecs = self._embed([response, reference])
        score = float(np.dot(vecs[0], vecs[1]))
        return max(0.0, min(1.0, (score + 1) / 2))  # remap [-1,1] → [0,1]

    def rouge_l(self, response: str, reference: str) -> float:
        """
        ROUGE-L: Longest Common Subsequence (LCS) based F1 score.
        Measures sequence-level recall — useful for extractive tasks.
        """
        def lcs_length(a: list, b: list) -> int:
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]

        ref_tokens = reference.lower().split()
        resp_tokens = response.lower().split()

        if not ref_tokens or not resp_tokens:
            return 0.0

        lcs = lcs_length(resp_tokens, ref_tokens)
        precision = lcs / len(resp_tokens)
        recall = lcs / len(ref_tokens)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def faithfulness(self, response: str, context: str) -> float:
        """
        Faithfulness: how well is the response grounded in the provided context?
        Measures semantic overlap between response and context.
        A score of 0 = response is entirely off-context (hallucination risk).
        """
        if not context.strip():
            return 0.5  # no context provided — neutral score
        return self.semantic_similarity(response, context)

    def relevance(self, response: str, question: str) -> float:
        """
        Relevance: does the response actually address the question?
        High relevance = response is semantically close to the question's topic.
        """
        return self.semantic_similarity(response, question)

    def is_toxic(self, text: str) -> bool:
        """Simple keyword-based toxicity check."""
        words = set(re.findall(r'\w+', text.lower()))
        return bool(words & TOXIC_KEYWORDS)

    def score(self, response: ModelResponse, case: EvalCase) -> EvalScore:
        """Compute all metrics for a single (response, case) pair."""
        resp_text = response.response or ""

        return EvalScore(
            case_id=case.id,
            model_name=response.model_name,
            question=case.question,
            reference=case.reference_answer,
            response=resp_text,
            semantic_similarity=self.semantic_similarity(resp_text, case.reference_answer),
            rouge_l=self.rouge_l(resp_text, case.reference_answer),
            faithfulness=self.faithfulness(resp_text, case.context),
            relevance=self.relevance(resp_text, case.question),
            is_toxic=self.is_toxic(resp_text),
            latency_ms=response.latency_ms,
        )


# ── LLM Backends ──────────────────────────────────────────────────────────────

class MockLLM:
    """
    Deterministic mock LLM for testing without any model.
    Returns a snippet of the reference answer (simulates partial recall).
    """
    name = "mock_llm"

    def generate(self, question: str, context: str = "") -> tuple[str, float]:
        time.sleep(0.01)  # simulate latency
        # Return first 60% of the reference-like text from context
        source = context or question
        words = source.split()
        response = " ".join(words[:max(1, len(words) * 6 // 10)])
        latency = random_latency(10, 30)
        return response, latency


class HuggingFaceLLM:
    """Free HuggingFace QA model as LLM backend."""
    name = "hf_roberta"

    def __init__(self):
        log.info("Loading HuggingFace QA model...")
        self._pipe = hf_pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2"
        )
        log.info("✓ HuggingFace model loaded")

    def generate(self, question: str, context: str = "") -> tuple[str, float]:
        if not context.strip():
            context = question  # self-referential fallback
        t0 = time.time()
        try:
            result = self._pipe(question=question, context=context, max_answer_len=100)
            response = result["answer"]
        except Exception:
            response = ""
        latency = (time.time() - t0) * 1000
        return response, latency


def random_latency(low: float = 50, high: float = 400) -> float:
    return round(np.random.uniform(low, high), 1)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class LLMEvaluator:
    """
    Runs evaluation benchmarks across multiple models and test cases.

    Usage:
        evaluator = LLMEvaluator()
        evaluator.add_model(MockLLM())
        evaluator.add_model(HuggingFaceLLM())
        report = evaluator.run(eval_cases)
        evaluator.save_report(report)
    """

    def __init__(self):
        self.models: list = []
        self.metrics = MetricsEngine()

    def add_model(self, model):
        self.models.append(model)
        log.info(f"  + Added model: {model.name}")

    def _run_model(self, model, cases: list[EvalCase]) -> ModelReport:
        report = ModelReport(model_name=model.name)

        for case in cases:
            context = case.context or case.reference_answer
            response_text, latency = model.generate(case.question, context)

            response = ModelResponse(
                case_id=case.id,
                model_name=model.name,
                response=response_text,
                latency_ms=latency,
                tokens_approx=len(response_text.split()),
            )

            score = self.metrics.score(response, case)
            report.scores.append(score)

        return report

    def run(self, cases: list[EvalCase]) -> dict[str, ModelReport]:
        """Evaluate all models on all cases."""
        log.info(f"\n🧪 Running evaluation: {len(self.models)} models × {len(cases)} cases")
        reports = {}
        for model in self.models:
            log.info(f"  Evaluating {model.name}...")
            reports[model.name] = self._run_model(model, cases)
            log.info(f"  ✓ {model.name}: composite={reports[model.name].avg_composite:.3f}")
        return reports

    def save_report(self, reports: dict[str, ModelReport], path: str = "eval_report.json"):
        """Save full evaluation report to JSON."""
        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "models": [m.name for m in self.models],
                "total_cases": sum(len(r.scores) for r in reports.values()),
            },
            "summaries": [r.summary() for r in reports.values()],
            "detailed_scores": {
                name: [s.to_dict() for s in report.scores]
                for name, report in reports.items()
            },
        }
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        log.info(f"  ✓ Report saved to {path}")

    def print_leaderboard(self, reports: dict[str, ModelReport]):
        """Print a ranked leaderboard table."""
        ranked = sorted(reports.values(), key=lambda r: r.avg_composite, reverse=True)

        print("\n" + "=" * 70)
        print("LLM EVALUATION LEADERBOARD")
        print("=" * 70)
        print(f"{'Rank':<6}{'Model':<20}{'Composite':<12}{'Semantic':<12}"
              f"{'ROUGE-L':<10}{'Latency':<12}{'Toxic%'}")
        print("-" * 70)

        for rank, report in enumerate(ranked, 1):
            toxic_pct = f"{report.toxicity_rate * 100:.1f}%"
            print(
                f"{rank:<6}{report.model_name:<20}"
                f"{report.avg_composite:<12.4f}"
                f"{report.avg_semantic:<12.4f}"
                f"{report.avg_rouge:<10.4f}"
                f"{report.avg_latency:<12.1f}"
                f"{toxic_pct}"
            )
        print("=" * 70)


# ── Built-in Eval Dataset ─────────────────────────────────────────────────────

def get_default_eval_cases() -> list[EvalCase]:
    """
    Built-in evaluation cases covering AI/ML topics.
    In production: load from JSON / Hugging Face Datasets.
    """
    return [
        EvalCase(
            id="rag_001",
            question="What is RAG in the context of LLMs?",
            reference_answer="RAG stands for Retrieval-Augmented Generation. It combines a retrieval system that fetches relevant documents with a language model that generates answers grounded in those documents.",
            context="RAG (Retrieval-Augmented Generation) is a technique that improves LLM responses by first retrieving relevant documents from a knowledge base using vector similarity search, then using those documents as context for the language model to generate factually grounded answers.",
            category="llm_concepts",
        ),
        EvalCase(
            id="faiss_001",
            question="What is FAISS and what is it used for?",
            reference_answer="FAISS is Facebook AI Similarity Search, a library for efficient similarity search and clustering of dense vectors. It is widely used in RAG pipelines to index document embeddings and retrieve the most semantically similar chunks.",
            context="FAISS (Facebook AI Similarity Search) is an open-source library developed by Meta. It provides efficient algorithms for similarity search and clustering of high-dimensional dense vectors. It supports both exact and approximate nearest neighbor search.",
            category="vector_db",
        ),
        EvalCase(
            id="async_001",
            question="What is asyncio and why is it used in Python?",
            reference_answer="asyncio is Python's library for writing concurrent code using async/await syntax. It uses a single-threaded event loop to handle many I/O-bound tasks simultaneously without blocking.",
            context="Python's asyncio library enables asynchronous programming using coroutines. It runs an event loop that can handle thousands of concurrent I/O-bound operations efficiently without needing multiple threads.",
            category="python_backend",
        ),
        EvalCase(
            id="redis_001",
            question="Why use Redis for caching in LLM applications?",
            reference_answer="Redis is used for caching LLM responses because LLM API calls are expensive and slow. Caching identical prompts saves cost and reduces latency from hundreds of milliseconds to microseconds.",
            context="Redis is an in-memory data structure store used as a cache, message broker, and queue. In LLM applications, caching repeated prompt-response pairs in Redis avoids redundant API calls, reducing both latency and cost.",
            category="backend",
        ),
        EvalCase(
            id="embed_001",
            question="What are sentence embeddings?",
            reference_answer="Sentence embeddings are dense vector representations of text where semantically similar sentences have similar vectors. They are produced by models like sentence-transformers and used in semantic search and RAG pipelines.",
            context="Sentence transformers convert sentences into fixed-size dense vectors. Similar sentences have vectors with high cosine similarity. These embeddings are used for semantic search, clustering, and as the retrieval component in RAG systems.",
            category="llm_concepts",
        ),
        EvalCase(
            id="fastapi_001",
            question="What makes FastAPI good for LLM services?",
            reference_answer="FastAPI is built on asyncio and supports async endpoints natively, making it ideal for LLM services where multiple requests can be handled concurrently without blocking while waiting for LLM responses.",
            context="FastAPI is a modern Python web framework built on Starlette and asyncio. It supports native async/await syntax, automatic OpenAPI documentation, and Pydantic request validation, making it popular for building high-performance AI API services.",
            category="backend",
        ),
    ]


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    random.seed(42)
    np.random.seed(42)

    print("\n" + "=" * 70)
    print("LLM Evaluation Toolkit")
    print("=" * 70)

    evaluator = LLMEvaluator()

    # Add models (Mock is instant, HF takes ~1 min first run)
    evaluator.add_model(MockLLM())

    # Uncomment to also evaluate the real HF model:
    # evaluator.add_model(HuggingFaceLLM())

    cases = get_default_eval_cases()
    print(f"\nEval dataset: {len(cases)} cases across {len(set(c.category for c in cases))} categories\n")

    reports = evaluator.run(cases)
    evaluator.print_leaderboard(reports)
    evaluator.save_report(reports)

    print("\n✅ Done. Results saved to eval_report.json")
