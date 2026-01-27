# RAGAS: Automated Evaluation of RAG (2023)

## Paper Details

**Title:** RAGAS: Automated Evaluation of Retrieval Augmented Generation

**Authors:** Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert

**Published:** September 2023

**Link:** https://arxiv.org/abs/2309.15217

## Key Insight

Evaluating RAG systems requires measuring both retrieval quality and generation quality. RAGAS provides automated metrics that assess faithfulness, relevance, and correctness without requiring ground truth answers for every test case.

## The Problem

RAG systems can fail in multiple ways:
- Retrieval finds wrong documents
- Retrieval finds right documents but wrong sections
- Generation ignores retrieved context
- Generation hallucinates beyond context
- Answer is correct but doesn't address the question

Traditional metrics (BLEU, ROUGE) don't capture these failure modes. Human evaluation is expensive and slow.

## The RAGAS Framework

RAGAS defines four core metrics:

### 1. Faithfulness

Does the answer stick to the retrieved context?

**Definition:** Proportion of claims in the answer that can be inferred from the context.

```
Faithfulness = (Claims supported by context) / (Total claims in answer)
```

**Example:**
- Context: "Paris is the capital of France. It has a population of 2.1 million."
- Answer: "Paris is the capital of France with 2.1 million people and beautiful architecture."
- Faithfulness: 2/3 = 0.67 ("beautiful architecture" not in context)

### 2. Answer Relevance

Does the answer address the question?

**Definition:** How well the answer addresses what was actually asked.

```
Answer Relevance = Semantic similarity between question and answer
```

**Example:**
- Question: "What is the capital of France?"
- Answer: "France is a country in Western Europe with rich history."
- Relevance: Low (doesn't answer the question)

### 3. Context Precision

Are the retrieved documents relevant and well-ranked?

**Definition:** Proportion of relevant items in retrieved context, weighted by rank.

```
Context Precision = Weighted average of relevance at each position
```

**Example:**
- Retrieved: [Doc about Paris (relevant), Doc about London (irrelevant), Doc about French government (relevant)]
- Precision considers that irrelevant doc at position 2 hurts more than at position 3

### 4. Context Recall

Does the retrieved context cover what's needed to answer?

**Definition:** Proportion of the ground truth answer that can be attributed to retrieved context.

```
Context Recall = (Ground truth claims in context) / (Total ground truth claims)
```

**Example:**
- Ground truth: "Paris is capital, population 2.1M, on Seine River"
- Context mentions: Paris is capital, population 2.1M
- Recall: 2/3 = 0.67 (missing Seine River)

## Implementation

### Installation

```bash
uv pip install ragas
```

### Basic Evaluation

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset

# Prepare evaluation data
data = {
    "question": ["What is RAG?", "How do embeddings work?"],
    "answer": ["RAG is Retrieval-Augmented Generation...", "Embeddings are..."],
    "contexts": [
        ["RAG combines retrieval with generation..."],
        ["Embeddings represent text as vectors..."]
    ],
    "ground_truth": ["RAG is a technique that...", "Embeddings are vector..."]
}
dataset = Dataset.from_dict(data)

# Run evaluation
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

print(results)
```

### Evaluating a RAG Pipeline

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

def evaluate_rag_pipeline(questions: list[str], rag_chain) -> dict:
    """Evaluate a RAG pipeline on test questions."""
    
    # Run pipeline on questions
    results_data = {
        "question": [],
        "answer": [],
        "contexts": []
    }
    
    for question in questions:
        # Get answer and contexts from your RAG pipeline
        response = rag_chain.invoke(question)
        
        results_data["question"].append(question)
        results_data["answer"].append(response["answer"])
        results_data["contexts"].append(response["contexts"])
    
    dataset = Dataset.from_dict(results_data)
    
    # Evaluate (no ground truth needed for faithfulness/relevancy)
    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy]
    )
    
    return scores
```

### Custom Evaluation with Ground Truth

```python
from ragas.metrics import context_recall, answer_correctness

# When you have ground truth answers
data_with_truth = {
    "question": questions,
    "answer": generated_answers,
    "contexts": retrieved_contexts,
    "ground_truth": expected_answers  # Required for recall/correctness
}

dataset = Dataset.from_dict(data_with_truth)

results = evaluate(
    dataset,
    metrics=[context_recall, answer_correctness]
)
```

## Synthetic Test Data Generation

RAGAS can generate test data from your documents:

```python
from ragas.testset import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Setup generator
generator = TestsetGenerator.from_langchain(
    generator_llm=ChatOpenAI(model="gpt-4o-mini"),
    critic_llm=ChatOpenAI(model="gpt-4o-mini"),
    embeddings=OpenAIEmbeddings()
)

# Generate test set from documents
testset = generator.generate_with_langchain_docs(
    documents=your_documents,
    test_size=50,
    distributions={
        simple: 0.5,      # Simple factual questions
        reasoning: 0.3,   # Multi-step reasoning
        multi_context: 0.2  # Requires multiple docs
    }
)

# Convert to dataset
test_df = testset.to_pandas()
```

## Integration with LangSmith

```python
from langsmith import Client
from ragas.integrations.langsmith import evaluate as ragas_evaluate

client = Client()

# Evaluate runs from LangSmith
results = ragas_evaluate(
    dataset_name="my-rag-dataset",
    llm=ChatOpenAI(model="gpt-4o-mini"),
    metrics=[faithfulness, answer_relevancy]
)
```

## Interpreting Results

### Score Ranges

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Faithfulness | > 0.9 | 0.7-0.9 | < 0.7 |
| Answer Relevancy | > 0.8 | 0.6-0.8 | < 0.6 |
| Context Precision | > 0.8 | 0.6-0.8 | < 0.6 |
| Context Recall | > 0.8 | 0.6-0.8 | < 0.6 |

### Diagnosing Issues

**Low Faithfulness:**
- Model hallucinating beyond context
- Fix: Stronger grounding instructions, better prompts

**Low Answer Relevancy:**
- Not addressing the question
- Fix: Better prompt engineering, check for question understanding

**Low Context Precision:**
- Retrieving irrelevant documents
- Fix: Better embeddings, improved chunking, reranking

**Low Context Recall:**
- Missing relevant information in retrieval
- Fix: More documents (higher k), better search, query expansion

## Advanced Usage

### Component-Level Analysis

```python
from ragas.metrics import (
    context_entity_recall,
    noise_sensitivity,
    answer_similarity
)

# More granular metrics
detailed_results = evaluate(
    dataset,
    metrics=[
        context_entity_recall,  # Are key entities retrieved?
        noise_sensitivity,      # Does noise in context hurt?
        answer_similarity       # Semantic similarity to ground truth
    ]
)
```

### Batch Evaluation

```python
def evaluate_pipeline_versions(pipelines: dict, test_data: Dataset):
    """Compare multiple RAG pipeline versions."""
    
    results = {}
    for name, pipeline in pipelines.items():
        # Generate answers with this pipeline
        answers = [pipeline.invoke(q) for q in test_data["question"]]
        
        eval_data = Dataset.from_dict({
            "question": test_data["question"],
            "answer": [a["answer"] for a in answers],
            "contexts": [a["contexts"] for a in answers],
            "ground_truth": test_data["ground_truth"]
        })
        
        scores = evaluate(eval_data, metrics=[faithfulness, context_recall])
        results[name] = scores
    
    return results
```

## Best Practices

1. **Start with faithfulness and relevancy.** Don't need ground truth.

2. **Build ground truth incrementally.** Use production data.

3. **Track metrics over time.** Catch regressions.

4. **Analyze failures qualitatively.** Numbers tell you what, not why.

5. **Generate diverse test cases.** Cover edge cases.

## Limitations

- LLM-based metrics can be inconsistent
- Requires good LLM for evaluation (cost)
- Context recall needs ground truth
- May not catch subtle errors

## Comparison with Other Approaches

| Approach | Pros | Cons |
|----------|------|------|
| RAGAS | Automated, multi-dimensional | LLM cost, can miss nuance |
| Human Eval | Gold standard | Slow, expensive |
| BLEU/ROUGE | Fast, cheap | Poor for semantic quality |
| Embedding Similarity | Fast | Misses factual errors |

## Why This Matters

RAGAS provides a practical framework for:
- Automated testing in CI/CD
- Comparing pipeline versions
- Identifying failure modes
- Tracking quality over time

It bridges the gap between expensive human evaluation and inadequate traditional metrics.

## Related Concepts

- **Evaluation Metrics**: Broader evaluation landscape
- **RAG Fundamentals**: What RAGAS evaluates
- **LangSmith**: Integration for tracing + evaluation
