# LangSmith Quick Reference

## Overview

LangSmith is Anthropic's platform for debugging, testing, evaluating, and monitoring LLM applications. It provides tracing, evaluation datasets, prompt management, and deployment capabilities. LangSmith Studio v2 (2025) added web-based debugging with time travel.

## Setup

```bash
uv pip install langsmith
```

```python
import os

os.environ["LANGSMITH_API_KEY"] = "your-api-key"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "my-project"
```

## Tracing

### Automatic Tracing

LangChain and LangGraph automatically trace when environment variables are set:

```python
from langchain_openai import ChatOpenAI

# This call is automatically traced
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("Hello")
```

### Manual Tracing

```python
from langsmith import traceable

@traceable
def my_function(input_text: str) -> str:
    """This function is traced."""
    result = process(input_text)
    return result

@traceable(name="Custom Name", tags=["production"])
def another_function(data: dict) -> dict:
    """Traced with custom name and tags."""
    return transform(data)
```

### Trace Context

```python
from langsmith import trace

with trace("my-operation", inputs={"query": "test"}) as t:
    result = do_something()
    t.outputs = {"result": result}
    t.metadata = {"model": "gpt-4o-mini"}
```

### Nested Traces

```python
@traceable
def parent_function(query: str):
    # Child traces are automatically nested
    result1 = child_function_1(query)
    result2 = child_function_2(result1)
    return result2

@traceable
def child_function_1(input: str):
    return process_step_1(input)

@traceable
def child_function_2(input: str):
    return process_step_2(input)
```

## Evaluation

### Creating Datasets

```python
from langsmith import Client

client = Client()

# Create dataset
dataset = client.create_dataset(
    dataset_name="qa-evaluation",
    description="Question-answer pairs for RAG evaluation"
)

# Add examples
client.create_examples(
    inputs=[
        {"question": "What is RAG?"},
        {"question": "How do embeddings work?"}
    ],
    outputs=[
        {"answer": "RAG is Retrieval-Augmented Generation..."},
        {"answer": "Embeddings are vector representations..."}
    ],
    dataset_id=dataset.id
)
```

### Running Evaluations

```python
from langsmith.evaluation import evaluate

def my_app(inputs: dict) -> dict:
    """The application being evaluated."""
    question = inputs["question"]
    answer = rag_chain.invoke(question)
    return {"answer": answer}

def correctness_evaluator(run, example) -> dict:
    """Custom evaluator."""
    predicted = run.outputs["answer"]
    expected = example.outputs["answer"]
    
    # Use LLM to judge correctness
    score = llm_judge(predicted, expected)
    
    return {"key": "correctness", "score": score}

# Run evaluation
results = evaluate(
    my_app,
    data="qa-evaluation",  # Dataset name
    evaluators=[correctness_evaluator],
    experiment_prefix="rag-v1"
)
```

### Built-in Evaluators

```python
from langsmith.evaluation import LangChainStringEvaluator

# Correctness evaluator
correctness = LangChainStringEvaluator("correctness")

# Helpfulness evaluator  
helpfulness = LangChainStringEvaluator("helpfulness")

# Custom criteria
custom = LangChainStringEvaluator(
    "criteria",
    config={"criteria": {"conciseness": "Is the response concise?"}}
)

results = evaluate(
    my_app,
    data="qa-evaluation",
    evaluators=[correctness, helpfulness, custom]
)
```

### Comparing Experiments

```python
from langsmith import Client

client = Client()

# Get experiment results
experiment1 = client.read_project(project_name="rag-v1")
experiment2 = client.read_project(project_name="rag-v2")

# Compare metrics
for run in client.list_runs(project_name="rag-v1"):
    print(f"Run: {run.id}, Feedback: {run.feedback_stats}")
```

## LangSmith Studio v2

### Web-Based Debugging

Access at: `https://smith.langchain.com`

Features:
- **Graph View**: Visualize agent execution flow
- **Chat Mode**: Interactive testing with your agents
- **Time Travel**: Step backward/forward through execution
- **State Inspection**: Examine state at any point

### Playground

Test prompts and chains interactively:

```python
# Traces appear in LangSmith automatically
# Use the Playground to:
# - Modify prompts and re-run
# - Compare different model outputs
# - Save successful prompts
```

## Feedback and Annotation

### Programmatic Feedback

```python
from langsmith import Client

client = Client()

# Add feedback to a run
client.create_feedback(
    run_id="run-uuid",
    key="user-rating",
    score=0.8,
    comment="Good response but could be more detailed"
)

# Feedback with correction
client.create_feedback(
    run_id="run-uuid",
    key="correction",
    correction={"answer": "The correct answer is..."}
)
```

### User Feedback Collection

```python
from langsmith import traceable

@traceable
def chat(message: str) -> tuple[str, str]:
    response = llm.invoke(message)
    run_id = get_current_run_id()  # Get for feedback later
    return response, run_id

# In your app, after user rates response:
def submit_feedback(run_id: str, rating: int):
    client.create_feedback(
        run_id=run_id,
        key="user-rating",
        score=rating / 5.0
    )
```

## Prompt Management

### Hub

```python
from langsmith import hub

# Pull prompt from hub
prompt = hub.pull("my-org/rag-prompt")

# Use in chain
chain = prompt | llm | StrOutputParser()

# Push updated prompt
hub.push("my-org/rag-prompt-v2", prompt)
```

### Versioning

```python
# Pull specific version
prompt_v1 = hub.pull("my-org/rag-prompt:v1")
prompt_latest = hub.pull("my-org/rag-prompt:latest")

# Compare versions in Studio
```

## Monitoring

### Production Monitoring

```python
import os

# Set project for production
os.environ["LANGSMITH_PROJECT"] = "production"

# All traces go to production project
# Monitor in LangSmith dashboard
```

### Alerts and Dashboards

Configure in LangSmith UI:
- Error rate alerts
- Latency thresholds
- Cost monitoring
- Custom metric dashboards

### Sampling

```python
import random

# Sample 10% of production traffic for tracing
os.environ["LANGSMITH_TRACING"] = "true" if random.random() < 0.1 else "false"
```

## Deployment (LangGraph Platform)

### Deploying Agents

```python
# langgraph.json configuration
{
    "graphs": {
        "my_agent": "./agent.py:graph"
    },
    "dependencies": ["langchain", "langchain-openai"]
}
```

```bash
# Deploy to LangSmith
langgraph deploy --project my-agent
```

### API Access

```python
from langgraph_sdk import get_client

client = get_client(url="https://your-deployment.langsmith.com")

# Invoke deployed agent
result = await client.runs.create(
    assistant_id="my_agent",
    input={"messages": [{"role": "user", "content": "Hello"}]}
)
```

## Best Practices

### Trace Organization

```python
# Use projects to organize
os.environ["LANGSMITH_PROJECT"] = "feature-development"

# Use tags for filtering
@traceable(tags=["rag", "production"])
def rag_pipeline(query: str):
    pass

# Use metadata for context
@traceable(metadata={"user_id": "123", "session": "abc"})
def user_request(query: str):
    pass
```

### Evaluation Strategy

1. **Create golden datasets** from production examples
2. **Run evaluations on PRs** before merging
3. **Compare experiments** to track improvements
4. **Monitor production** for regressions

### Cost Management

```python
# Track token usage
@traceable
def tracked_call(prompt: str):
    response = llm.invoke(prompt)
    # Token counts appear in trace
    return response

# Filter expensive traces
# Use sampling in production
```

## Debugging Tips

### Finding Issues

1. **Filter by error**: Find failed runs
2. **Sort by latency**: Find slow runs
3. **Search by input**: Find specific queries
4. **Compare runs**: See what changed

### Time Travel Debugging

1. Open trace in Studio
2. Click on any node
3. See state at that point
4. Modify and re-run from there

### Common Issues

- **Missing traces**: Check env vars are set
- **Nested traces broken**: Ensure @traceable on all functions
- **High latency**: Check trace for slow nodes
- **Errors not visible**: Add try/except with trace logging

## Related Concepts

- **LangChain**: Automatically traces to LangSmith
- **LangGraph**: Visualization in Studio
- **Evaluation**: Core LangSmith capability
- **Observability**: Production monitoring
