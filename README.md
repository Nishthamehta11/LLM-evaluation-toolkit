LLM Evaluation Toolkit

**Benchmark and compare LLM responses automatically** — semantic similarity, ROUGE-L, faithfulness, relevance, toxicity, and latency. No human annotators needed.

Built to evaluate RAG pipelines and LLM services the way production teams do.

---

## Why This Exists

Shipping an LLM feature without evaluation is like deploying code without tests. This toolkit gives you:
- Quantitative scores to compare model versions or prompt strategies
- Faithfulness scoring to detect hallucination risk
- Latency benchmarking to make cost/quality trade-offs
- A leaderboard to rank models objectively

---

## Metrics

| Metric | What It Measures | Weight |
|--------|-----------------|--------|
| Semantic similarity | Cosine similarity of response vs reference embeddings | 35% |
| ROUGE-L | Longest Common Subsequence overlap | 20% |
| Faithfulness | Is the response grounded in the context? | 25% |
| Relevance | Does the response address the question? | 20% |
| Toxicity | Keyword-based safety flag | Disqualifier |
| Latency | Wall-clock response time (ms) | Reported |

**Composite score** = weighted average of the first four metrics (0 if toxic).

---

## Quick Start

```bash
pip install sentence-transformers transformers torch numpy
python llm_evaluator.py
```

### Output
```
======================================================================
LLM EVALUATION LEADERBOARD
======================================================================
Rank  Model               Composite   Semantic    ROUGE-L   Latency     Toxic%
----------------------------------------------------------------------
1     hf_roberta          0.7231      0.8102      0.5843    312.4       0.0%
2     mock_llm            0.5814      0.6920      0.4901    18.2        0.0%
======================================================================
```

---

## Architecture

```
EvalCase (question + reference + context)
         ↓
    LLM Backend (Mock / HuggingFace / OpenAI)
         ↓
    ModelResponse (text + latency)
         ↓
    MetricsEngine
     ├── semantic_similarity()   → SentenceTransformer cosine
     ├── rouge_l()               → LCS-based F1
     ├── faithfulness()          → response ↔ context similarity
     ├── relevance()             → response ↔ question similarity
     └── is_toxic()              → keyword scan
         ↓
    EvalScore → ModelReport → Leaderboard + JSON report
```

---

## Add Your Own Model

```python
class MyCustomLLM:
    name = "my_model"
    
    def generate(self, question: str, context: str = "") -> tuple[str, float]:
        t0 = time.time()
        response = your_llm_call(question, context)
        latency_ms = (time.time() - t0) * 1000
        return response, latency_ms

evaluator = LLMEvaluator()
evaluator.add_model(MyCustomLLM())
evaluator.add_model(HuggingFaceLLM())
reports = evaluator.run(get_default_eval_cases())
evaluator.print_leaderboard(reports)
```

---

## Add Your Own Eval Cases

```python
cases = [
    EvalCase(
        id="custom_001",
        question="What is the capital of France?",
        reference_answer="Paris is the capital of France.",
        context="France is a country in Western Europe. Its capital city is Paris.",
        category="geography",
    )
]
reports = evaluator.run(cases)
```

---

## Extending to OpenAI

```python
import openai

class OpenAILLM:
    name = "gpt-4o-mini"
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def generate(self, question, context=""):
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Context: {context}"},
                {"role": "user", "content": question}
            ]
        )
        return resp.choices[0].message.content, (time.time()-t0)*1000
```

---

## Files

```
llm-evaluation-toolkit/
├── llm_evaluator.py        # Full implementation
├── requirements.txt
└── eval_report.json        # Auto-generated: detailed scores per case
```

---

## Requirements

```
sentence-transformers>=2.2.2
transformers>=4.30.0
torch>=2.0.0
numpy>=1.24.0
```

---

*Python 3.10+ · sentence-transformers · No API keys required for default run*
