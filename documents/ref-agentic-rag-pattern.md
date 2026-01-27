# Agentic RAG Pattern

## Overview

Agentic RAG extends basic RAG by adding reasoning capabilities. Instead of always retrieving before generating, an agentic RAG system decides when to retrieve, what to retrieve, and whether retrieved results are sufficient. The agent can ask clarifying questions, search multiple sources, and validate its own answers.

## When to Use

**Use agentic RAG when:**
- Query complexity varies (some need retrieval, some don't)
- Multiple retrieval strategies might apply
- The system should handle ambiguous queries
- Quality matters more than speed
- Retrieved results may need validation or expansion

**Use basic RAG when:**
- All queries need retrieval
- The retrieval strategy is fixed
- Speed is critical
- Simplicity is preferred

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Query Analysis                             │
│  • Classify query type                                       │
│  • Determine if retrieval needed                            │
│  • Plan retrieval strategy                                  │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
            ┌─────────────┴─────────────┐
            ▼                           ▼
┌───────────────────┐         ┌───────────────────┐
│  Direct Answer    │         │     Retrieve      │
│  (no retrieval)   │         │     Context       │
└───────────────────┘         └─────────┬─────────┘
                                        ▼
                              ┌───────────────────┐
                              │  Evaluate Results │
                              │  • Sufficient?    │
                              │  • Relevant?      │
                              └─────────┬─────────┘
                                        ▼
                          ┌─────────────┴─────────────┐
                          ▼                           ▼
                 ┌───────────────┐          ┌───────────────┐
                 │   Generate    │          │ Retrieve More │
                 │   Response    │          │  (iterate)    │
                 └───────────────┘          └───────────────┘
```

## Key Components

### Query Analyzer

Examines incoming queries to determine complexity and required information:

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def analyze_query(query: str) -> dict:
    """
    Analyze a user query to determine handling strategy.
    
    Returns classification and retrieval recommendation.
    """
    analysis_prompt = f"""Analyze this query and return JSON:
    
Query: {query}

Classify as:
- type: "factual" | "conceptual" | "procedural" | "comparison" | "opinion"
- complexity: "simple" | "moderate" | "complex"
- needs_retrieval: true | false
- search_queries: ["suggested", "search", "queries"] if retrieval needed
"""
    
    result = llm.invoke(analysis_prompt)
    return json.loads(result)
```

### Adaptive Retriever

Adjusts retrieval strategy based on query analysis:

```python
def adaptive_retrieve(query: str, analysis: dict) -> list:
    """Retrieve with strategy based on query analysis."""
    
    if not analysis["needs_retrieval"]:
        return []
    
    if analysis["complexity"] == "simple":
        # Single focused search
        return vector_store.similarity_search(query, k=3)
    
    elif analysis["complexity"] == "moderate":
        # Multiple searches with different angles
        results = []
        for search_query in analysis["search_queries"]:
            results.extend(vector_store.similarity_search(search_query, k=2))
        return deduplicate(results)
    
    else:  # complex
        # Comprehensive multi-query with reranking
        all_results = []
        for search_query in analysis["search_queries"]:
            all_results.extend(vector_store.similarity_search(search_query, k=5))
        
        return rerank(all_results, query, top_k=5)
```

### Result Evaluator

Assesses whether retrieved documents adequately answer the query:

```python
@tool
def evaluate_retrieval(query: str, documents: list) -> dict:
    """
    Evaluate if retrieved documents can answer the query.
    
    Returns assessment with recommendation.
    """
    context = "\n\n".join([doc.page_content for doc in documents])
    
    eval_prompt = f"""Given this query and retrieved context, assess:

Query: {query}

Retrieved Context:
{context}

Evaluate:
1. Relevance: Are the documents about the right topic? (0-1)
2. Completeness: Do they contain enough to answer fully? (0-1)
3. Confidence: How confident are you in answering? (0-1)

If confidence < 0.7, suggest additional searches.

Return JSON: {{"relevance": X, "completeness": X, "confidence": X, "additional_searches": [...]}}
"""
    
    return json.loads(llm.invoke(eval_prompt))
```

## Implementation with LangGraph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

class AgenticRAGState(TypedDict):
    query: str
    analysis: dict
    retrieved_docs: list
    evaluation: dict
    response: str
    iteration: int

def analyze_node(state: AgenticRAGState) -> AgenticRAGState:
    """Analyze the query and plan retrieval."""
    analysis = analyze_query(state["query"])
    return {"analysis": analysis}

def retrieve_node(state: AgenticRAGState) -> AgenticRAGState:
    """Retrieve relevant documents."""
    docs = adaptive_retrieve(state["query"], state["analysis"])
    return {"retrieved_docs": docs}

def evaluate_node(state: AgenticRAGState) -> AgenticRAGState:
    """Evaluate retrieval quality."""
    evaluation = evaluate_retrieval(state["query"], state["retrieved_docs"])
    return {"evaluation": evaluation, "iteration": state.get("iteration", 0) + 1}

def generate_node(state: AgenticRAGState) -> AgenticRAGState:
    """Generate final response."""
    context = format_documents(state["retrieved_docs"])
    response = generate_answer(state["query"], context)
    return {"response": response}

def direct_answer_node(state: AgenticRAGState) -> AgenticRAGState:
    """Answer without retrieval."""
    response = llm.invoke(f"Answer this question: {state['query']}")
    return {"response": response}

# Routing functions
def needs_retrieval(state: AgenticRAGState) -> Literal["retrieve", "direct"]:
    if state["analysis"]["needs_retrieval"]:
        return "retrieve"
    return "direct"

def retrieval_sufficient(state: AgenticRAGState) -> Literal["generate", "retrieve_more"]:
    if state["evaluation"]["confidence"] >= 0.7 or state["iteration"] >= 3:
        return "generate"
    return "retrieve_more"

# Build graph
graph = StateGraph(AgenticRAGState)

graph.add_node("analyze", analyze_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("evaluate", evaluate_node)
graph.add_node("generate", generate_node)
graph.add_node("direct_answer", direct_answer_node)

graph.set_entry_point("analyze")
graph.add_conditional_edges("analyze", needs_retrieval, {
    "retrieve": "retrieve",
    "direct": "direct_answer"
})
graph.add_edge("retrieve", "evaluate")
graph.add_conditional_edges("evaluate", retrieval_sufficient, {
    "generate": "generate",
    "retrieve_more": "retrieve"
})
graph.add_edge("generate", END)
graph.add_edge("direct_answer", END)

agentic_rag = graph.compile()
```

## Handling Edge Cases

### Ambiguous Queries

```python
def handle_ambiguity(query: str, analysis: dict) -> str:
    """Ask clarifying questions for ambiguous queries."""
    if analysis.get("ambiguous"):
        clarification = llm.invoke(f"""
This query is ambiguous: "{query}"

Generate 2-3 clarifying questions to better understand what the user needs.
""")
        return clarification
    return None
```

### No Results Found

```python
def handle_no_results(query: str) -> str:
    """Gracefully handle when retrieval finds nothing."""
    return llm.invoke(f"""
I searched but couldn't find specific information about: {query}

Based on my general knowledge, here's what I can tell you...
(Note: This is not from your documents)
""")
```

### Contradictory Information

```python
def handle_contradictions(docs: list) -> str:
    """Identify and handle contradictory retrieved information."""
    check_prompt = f"""
Review these documents for contradictions:
{format_documents(docs)}

Are there any contradictory claims? If so, identify them and explain how to resolve.
"""
    return llm.invoke(check_prompt)
```

## Trade-offs

**Pros:**
- More efficient (skips unnecessary retrieval)
- Better handling of complex queries
- Can ask clarifying questions
- Self-correcting through evaluation
- Adaptive to query complexity

**Cons:**
- More complex to implement
- Higher latency for multi-step queries
- Requires careful prompt engineering
- More LLM calls (cost)
- Harder to debug than basic RAG

## Comparison with Basic RAG

| Aspect | Basic RAG | Agentic RAG |
|--------|-----------|-------------|
| Retrieval | Always | Conditional |
| Strategy | Fixed | Adaptive |
| Iterations | Single | Multiple possible |
| Complexity handling | Same for all | Scales with query |
| Latency | Lower, predictable | Higher, variable |
| Cost | Lower | Higher |
| Quality | Good | Better for complex |

## Best Practices

1. **Start with query analysis.** Classification drives everything else.

2. **Set iteration limits.** Prevent infinite retrieval loops.

3. **Cache analysis results.** Similar queries can reuse strategies.

4. **Log decision points.** Understand why the agent chose each path.

5. **Fallback gracefully.** When agentic path fails, fall back to basic RAG.

6. **Tune confidence thresholds.** Balance quality against latency.

## Related Patterns

- **Basic RAG**: Simpler retrieve-then-generate
- **Multi-Query RAG**: Generates multiple search queries
- **Self-RAG**: Model decides when to retrieve during generation
- **Corrective RAG**: Verifies and corrects retrieved information
