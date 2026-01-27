# Evaluation Metrics

## Why Evaluation Matters

You can't improve what you don't measure. AI systems feel magical when they work and mystifying when they fail. Systematic evaluation replaces intuition with data, letting you:

- Compare different approaches objectively
- Track improvements over time
- Catch regressions before production
- Build confidence in your system

## The Evaluation Challenge

Evaluating AI systems is hard because:

1. **Outputs are open-ended.** There's no single correct answer to "explain quantum computing."

2. **Quality is subjective.** What's "good enough" depends on context and users.

3. **Edge cases matter.** Systems that work 95% of the time can fail catastrophically on the other 5%.

4. **Manual evaluation doesn't scale.** You can't have humans review every response.

## Evaluation Approaches

### Vibe Checks

Informal, qualitative evaluation. Run your system with test queries, read the outputs, assess whether they're good.

```python
test_queries = [
    "What are embeddings?",
    "How does RAG work?",
    "Explain the difference between agents and workflows",
]

for query in test_queries:
    response = system.invoke(query)
    print(f"Q: {query}")
    print(f"A: {response}")
    print("---")
    # Human reads and assesses
```

**Pros:** Quick, catches obvious issues, good for early development.
**Cons:** Subjective, doesn't scale, can't track over time.

### Reference-Based Evaluation

Compare outputs to known good answers. Works when you have ground truth.

```python
test_cases = [
    {"query": "What is 2+2?", "expected": "4"},
    {"query": "Capital of France?", "expected": "Paris"},
]

def exact_match(response: str, expected: str) -> bool:
    return expected.lower() in response.lower()

scores = []
for case in test_cases:
    response = system.invoke(case["query"])
    score = exact_match(response, case["expected"])
    scores.append(score)

accuracy = sum(scores) / len(scores)
```

**Pros:** Objective, reproducible, automated.
**Cons:** Requires ground truth, doesn't handle open-ended queries.

### LLM-as-Judge

Use a language model to evaluate outputs. The evaluator LLM assesses quality based on criteria you define.

```python
from openai import OpenAI

client = OpenAI()

def llm_judge(query: str, response: str, criteria: str) -> dict:
    eval_prompt = f"""Evaluate the following response.
    
Query: {query}
Response: {response}

Criteria: {criteria}

Rate from 1-5 and explain your rating.
Format: {{"score": N, "explanation": "..."}}
"""
    
    result = client.responses.create(
        model="gpt-4o-mini",
        input=eval_prompt,
        response_format={"type": "json_object"}
    )
    
    return json.loads(result.output_text)

# Example usage
result = llm_judge(
    query="Explain machine learning",
    response="Machine learning is a type of AI...",
    criteria="Is the explanation clear, accurate, and appropriate for a beginner?"
)
```

**Pros:** Handles open-ended outputs, captures nuance.
**Cons:** Expensive, evaluator can be wrong, adds latency.

### Human Evaluation

Gold standard but expensive. Have humans rate outputs on defined criteria.

```python
# Typically done via annotation platform or simple UI
evaluation_task = {
    "query": "Explain transformers",
    "response": system.invoke("Explain transformers"),
    "criteria": [
        {"name": "accuracy", "description": "Is the information correct?"},
        {"name": "clarity", "description": "Is it easy to understand?"},
        {"name": "completeness", "description": "Does it cover key concepts?"}
    ],
    "scale": "1-5"
}
```

**Pros:** Most reliable, catches subtle issues.
**Cons:** Expensive, slow, doesn't scale.

## RAG Evaluation Metrics

### Retrieval Metrics

**Context Precision:** What fraction of retrieved chunks are actually relevant?

```python
def context_precision(retrieved_chunks: list, relevant_chunks: list) -> float:
    relevant_retrieved = set(retrieved_chunks) & set(relevant_chunks)
    return len(relevant_retrieved) / len(retrieved_chunks)
```

**Context Recall:** What fraction of relevant chunks were retrieved?

```python
def context_recall(retrieved_chunks: list, relevant_chunks: list) -> float:
    relevant_retrieved = set(retrieved_chunks) & set(relevant_chunks)
    return len(relevant_retrieved) / len(relevant_chunks)
```

**Mean Reciprocal Rank (MRR):** Where does the first relevant result appear?

```python
def mrr(retrieved_chunks: list, relevant_chunks: list) -> float:
    for i, chunk in enumerate(retrieved_chunks):
        if chunk in relevant_chunks:
            return 1.0 / (i + 1)
    return 0.0
```

### Generation Metrics

**Faithfulness:** Is the answer grounded in the retrieved context? Does it only claim things supported by the sources?

```python
def faithfulness_check(answer: str, context: str) -> float:
    """Use LLM to check if answer is supported by context."""
    prompt = f"""Given this context:
{context}

And this answer:
{answer}

Rate 0-1: Is every claim in the answer supported by the context?
"""
    # LLM evaluation
    return llm_evaluate(prompt)
```

**Answer Relevance:** Does the answer actually address the question?

```python
def answer_relevance(query: str, answer: str) -> float:
    """Use LLM to check if answer addresses the question."""
    prompt = f"""Question: {query}
Answer: {answer}

Rate 0-1: Does this answer address what was asked?
"""
    return llm_evaluate(prompt)
```

**Answer Correctness:** Is the answer factually accurate?

Requires ground truth or external verification. Often domain-specific.

## The RAGAS Framework

RAGAS (RAG Assessment) provides a comprehensive evaluation suite:

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# Prepare evaluation dataset
eval_data = {
    "question": ["What is RAG?", "How do embeddings work?"],
    "answer": [system.invoke(q) for q in questions],
    "contexts": [retriever.get_context(q) for q in questions],
    "ground_truth": ["RAG is...", "Embeddings are..."]  # Optional
}

# Run evaluation
results = evaluate(
    dataset=eval_data,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

print(results)
# faithfulness: 0.85
# answer_relevancy: 0.92
# context_precision: 0.78
# context_recall: 0.81
```

## Agent Evaluation Metrics

### Task Completion

Did the agent accomplish the goal?

```python
def task_completion_rate(test_cases: list) -> float:
    completed = 0
    for case in test_cases:
        result = agent.invoke(case["input"])
        if case["success_criteria"](result):
            completed += 1
    return completed / len(test_cases)
```

### Tool Usage

Did the agent use tools appropriately?

```python
def tool_usage_analysis(trace: list) -> dict:
    tool_calls = [step for step in trace if step["type"] == "tool_call"]
    
    return {
        "total_tool_calls": len(tool_calls),
        "unique_tools_used": len(set(t["tool"] for t in tool_calls)),
        "unnecessary_calls": count_unnecessary(tool_calls),
        "failed_calls": count_failures(tool_calls)
    }
```

### Reasoning Quality

Is the agent's reasoning sound?

```python
def evaluate_reasoning(trace: list) -> float:
    """LLM evaluates the agent's reasoning trace."""
    reasoning_steps = extract_thoughts(trace)
    
    prompt = f"""Evaluate this reasoning trace:
{reasoning_steps}

Criteria:
1. Are the steps logical?
2. Does each step follow from the previous?
3. Are conclusions justified?

Rate 0-1 for reasoning quality.
"""
    return llm_evaluate(prompt)
```

### Efficiency

How efficiently did the agent complete the task?

```python
def efficiency_metrics(trace: list) -> dict:
    return {
        "total_steps": len(trace),
        "llm_calls": count_llm_calls(trace),
        "tool_calls": count_tool_calls(trace),
        "total_tokens": sum_tokens(trace),
        "wall_time": trace[-1]["timestamp"] - trace[0]["timestamp"]
    }
```

## Building Evaluation Datasets

### Manual Creation

Create test cases by hand for critical scenarios:

```python
test_dataset = [
    {
        "query": "What is the ReAct pattern?",
        "expected_answer": "ReAct combines reasoning and acting...",
        "expected_tool_calls": ["search_knowledge_base"],
        "category": "concept_explanation"
    },
    # More test cases...
]
```

### Synthetic Generation

Use LLMs to generate test cases:

```python
def generate_test_cases(documents: list, n: int = 50) -> list:
    """Generate questions from documents."""
    test_cases = []
    
    for doc in documents:
        prompt = f"""Given this document:
{doc.page_content}

Generate 3 questions that could be answered from this content.
For each question, provide the expected answer.

Format: [{{"question": "...", "answer": "..."}}, ...]
"""
        cases = llm.invoke(prompt)
        test_cases.extend(json.loads(cases))
    
    return test_cases[:n]
```

### Production Sampling

Sample real user queries (with privacy considerations):

```python
def sample_production_queries(n: int = 100) -> list:
    """Sample queries from production logs."""
    queries = load_production_logs()
    
    # Filter sensitive content
    queries = [q for q in queries if not contains_pii(q)]
    
    # Sample diverse queries
    return stratified_sample(queries, n)
```

## Continuous Evaluation

### Regression Testing

Run evaluations on every change:

```python
# In CI/CD pipeline
def test_rag_quality():
    results = evaluate_rag_system(test_dataset)
    
    assert results["faithfulness"] >= 0.8, "Faithfulness regression"
    assert results["answer_relevancy"] >= 0.85, "Relevancy regression"
    assert results["context_precision"] >= 0.7, "Precision regression"
```

### Production Monitoring

Track quality metrics in production:

```python
def log_response_quality(query: str, response: str, context: list):
    # Sample responses for evaluation (don't evaluate everything)
    if random.random() < 0.01:  # 1% sample
        scores = {
            "faithfulness": evaluate_faithfulness(response, context),
            "relevancy": evaluate_relevancy(query, response)
        }
        metrics.log("response_quality", scores)
```

### A/B Testing

Compare system versions with real users:

```python
def ab_test_response(query: str, user_id: str):
    variant = get_variant(user_id)  # "A" or "B"
    
    if variant == "A":
        response = system_v1.invoke(query)
    else:
        response = system_v2.invoke(query)
    
    # Collect user feedback
    log_for_analysis(query, response, variant)
    
    return response
```

## Best Practices

1. **Start with vibe checks.** Understand your system qualitatively before quantifying.

2. **Build diverse test sets.** Cover different query types, edge cases, failure modes.

3. **Combine metrics.** No single metric captures quality. Use retrieval + generation + task metrics.

4. **Establish baselines.** Know your starting point before trying to improve.

5. **Automate ruthlessly.** Manual evaluation is for calibration; automation is for scale.

6. **Track over time.** Quality dashboards catch regressions and show trends.

7. **Include failure cases.** Test that your system fails gracefully on bad inputs.

## Related Concepts

- **RAGAS**: Framework for RAG evaluation
- **LangSmith**: Platform for evaluation and monitoring
- **Synthetic Data**: Generating test cases automatically
- **Metrics-Driven Development**: Using evaluation to guide improvements
