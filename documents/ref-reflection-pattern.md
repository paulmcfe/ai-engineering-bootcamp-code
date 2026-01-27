# Reflection Pattern

## Overview

The reflection pattern adds self-critique to agent workflows. After generating a response, the agent evaluates its own output, identifies weaknesses, and revises if needed. This iterative self-improvement produces higher-quality results at the cost of additional latency.

## Why Reflection Works

Language models can often recognize problems in text better than they avoid them during generation. A model might generate a mediocre explanation, but when asked "Is this explanation clear and complete?" it can identify specific issues.

Reflection exploits this asymmetry: generate first, critique second, then improve based on the critique.

## Basic Reflection Loop

```
┌───────────────────────────────────────────────────────┐
│                    User Query                          │
└─────────────────────────┬─────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────┐
│                    Generate                            │
│              Initial Response                          │
└─────────────────────────┬─────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────┐
│                    Reflect                             │
│    Critique response against criteria                  │
└─────────────────────────┬─────────────────────────────┘
                          ▼
                   Satisfactory?
                   /          \
                 Yes           No
                 /              \
┌───────────────┐      ┌───────────────────────────────┐
│ Return Final  │      │         Revise                 │
│   Response    │      │  Improve based on critique     │
└───────────────┘      └───────────────┬───────────────┘
                                       │
                              Back to Reflect
```

## Implementation

### Simple Reflection

```python
from openai import OpenAI

client = OpenAI()

def reflect_and_improve(query: str, max_iterations: int = 3) -> str:
    """Generate response with reflection loop."""
    
    # Initial generation
    response = client.responses.create(
        model="gpt-4o-mini",
        input=query
    ).output_text
    
    for iteration in range(max_iterations):
        # Reflection step
        critique = client.responses.create(
            model="gpt-4o-mini",
            input=f"""Critique this response:

Query: {query}
Response: {response}

Evaluate:
1. Is it accurate?
2. Is it complete?
3. Is it clear?
4. What could be improved?

If it's good enough, respond with "SATISFACTORY".
Otherwise, list specific improvements needed."""
        ).output_text
        
        # Check if satisfactory
        if "SATISFACTORY" in critique.upper():
            break
        
        # Revision step
        response = client.responses.create(
            model="gpt-4o-mini",
            input=f"""Improve this response based on the critique:

Original query: {query}
Current response: {response}
Critique: {critique}

Write an improved response addressing the critique."""
        ).output_text
    
    return response
```

### Structured Reflection

```python
from pydantic import BaseModel

class Critique(BaseModel):
    accuracy_score: float  # 0-1
    completeness_score: float
    clarity_score: float
    issues: list[str]
    suggestions: list[str]
    satisfactory: bool

def structured_reflection(query: str, response: str) -> Critique:
    """Get structured critique of a response."""
    
    critique_prompt = f"""Analyze this response and return JSON:

Query: {query}
Response: {response}

Return:
{{
    "accuracy_score": 0.0-1.0,
    "completeness_score": 0.0-1.0,
    "clarity_score": 0.0-1.0,
    "issues": ["list", "of", "problems"],
    "suggestions": ["specific", "improvements"],
    "satisfactory": true/false (true if all scores >= 0.8)
}}"""
    
    result = client.responses.create(
        model="gpt-4o-mini",
        input=critique_prompt,
        response_format={"type": "json_object"}
    )
    
    return Critique.model_validate_json(result.output_text)
```

### Multi-Aspect Reflection

Critique different aspects separately:

```python
def multi_aspect_reflection(query: str, response: str) -> dict:
    """Critique response on multiple dimensions."""
    
    aspects = {
        "accuracy": "Is the information factually correct?",
        "completeness": "Does it fully address the query?",
        "clarity": "Is it easy to understand?",
        "conciseness": "Is it appropriately brief without being incomplete?",
        "tone": "Is the tone appropriate for the context?"
    }
    
    critiques = {}
    for aspect, question in aspects.items():
        critique = client.responses.create(
            model="gpt-4o-mini",
            input=f"""Evaluate this response:

Query: {query}
Response: {response}

Aspect: {aspect}
Question: {question}

Rate 1-5 and explain your rating in 1-2 sentences.
Format: Score: N | Explanation: ..."""
        ).output_text
        
        critiques[aspect] = parse_critique(critique)
    
    return critiques
```

## With LangGraph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class ReflectionState(TypedDict):
    query: str
    response: str
    critique: dict
    iteration: int
    final: bool

def generate_node(state: ReflectionState) -> ReflectionState:
    """Generate or revise response."""
    
    if state.get("critique"):
        # Revision based on critique
        prompt = f"""Improve based on critique:
Query: {state["query"]}
Current: {state["response"]}
Issues: {state["critique"]["issues"]}
"""
    else:
        # Initial generation
        prompt = state["query"]
    
    response = llm.invoke(prompt)
    return {"response": response}

def reflect_node(state: ReflectionState) -> ReflectionState:
    """Critique the current response."""
    critique = structured_reflection(state["query"], state["response"])
    return {
        "critique": critique.model_dump(),
        "iteration": state.get("iteration", 0) + 1
    }

def should_continue(state: ReflectionState) -> str:
    """Decide whether to continue reflecting."""
    if state["critique"]["satisfactory"]:
        return "end"
    if state["iteration"] >= 3:
        return "end"
    return "revise"

# Build graph
graph = StateGraph(ReflectionState)

graph.add_node("generate", generate_node)
graph.add_node("reflect", reflect_node)

graph.set_entry_point("generate")
graph.add_edge("generate", "reflect")
graph.add_conditional_edges("reflect", should_continue, {
    "revise": "generate",
    "end": END
})

reflective_agent = graph.compile()
```

## Reflection Prompts

### General Quality Critique

```python
GENERAL_CRITIQUE = """Evaluate this response:

Query: {query}
Response: {response}

Consider:
- Accuracy: Is information correct?
- Completeness: Does it fully answer the question?
- Clarity: Is it easy to understand?
- Relevance: Does it stay on topic?

List any issues found. If response is good, say "No significant issues."
"""
```

### Domain-Specific Critique

```python
CODE_CRITIQUE = """Review this code response:

Query: {query}
Code: {response}

Check for:
- Correctness: Will it work as intended?
- Best practices: Does it follow coding standards?
- Error handling: Are edge cases handled?
- Documentation: Is it well-documented?
- Security: Any vulnerabilities?

List issues by category.
"""

WRITING_CRITIQUE = """Review this written content:

Query: {query}
Content: {response}

Evaluate:
- Structure: Is it well-organized?
- Flow: Do ideas connect smoothly?
- Grammar: Any errors?
- Style: Is tone appropriate?
- Engagement: Is it interesting to read?

Provide specific feedback.
"""
```

### Comparative Critique

```python
COMPARATIVE_CRITIQUE = """Compare these two responses:

Query: {query}
Response A: {response_a}
Response B: {response_b}

Which is better and why? Consider accuracy, completeness, and clarity.
If A is better, say "PREFER A: [reason]"
If B is better, say "PREFER B: [reason]"
"""
```

## Advanced Patterns

### External Verification

Combine reflection with external checks:

```python
def reflect_with_verification(query: str, response: str) -> dict:
    """Reflect with external fact-checking."""
    
    # Extract claims from response
    claims = extract_claims(response)
    
    # Verify each claim
    verified = []
    for claim in claims:
        # Search for supporting evidence
        evidence = search_for_evidence(claim)
        verified.append({
            "claim": claim,
            "supported": bool(evidence),
            "evidence": evidence
        })
    
    # Critique based on verification
    critique = generate_critique_from_verification(verified)
    
    return {
        "claims": verified,
        "critique": critique,
        "accuracy_score": sum(v["supported"] for v in verified) / len(verified)
    }
```

### Self-Consistency Check

Generate multiple responses and compare:

```python
def self_consistency_reflection(query: str, n: int = 3) -> str:
    """Generate multiple responses and find consensus."""
    
    responses = []
    for _ in range(n):
        response = llm.invoke(query, temperature=0.7)
        responses.append(response)
    
    # Compare responses
    comparison = llm.invoke(f"""
Compare these {n} responses to the same query:

Query: {query}

{chr(10).join(f"Response {i+1}: {r}" for i, r in enumerate(responses))}

Identify:
1. Points where all responses agree (high confidence)
2. Points where responses differ (needs resolution)
3. The best overall response

Synthesize the most accurate answer combining the best parts.
""")
    
    return comparison
```

### Critic Agent

Separate agent dedicated to critique:

```python
critic_agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_facts, check_logic],
    system_prompt="""You are a critical reviewer.

Your job: Find problems in responses. Be thorough and specific.

Always look for:
- Factual errors
- Logical fallacies
- Missing information
- Unclear explanations
- Unsupported claims

Be constructive - identify problems AND suggest fixes."""
)

def critique_with_agent(query: str, response: str) -> str:
    """Use dedicated critic agent."""
    return critic_agent.invoke({
        "input": f"Critique this response:\nQuery: {query}\nResponse: {response}"
    })
```

## When to Use Reflection

**Good for:**
- High-stakes outputs (reports, code, analysis)
- Complex queries requiring accuracy
- Content that will be published or shared
- Tasks where quality trumps speed

**Skip reflection when:**
- Speed is critical
- Queries are simple/routine
- Initial response quality is consistently high
- Cost is a major constraint

## Tuning Reflection

### Iteration Limits

Too few iterations may not fix problems. Too many waste resources.

```python
# Adaptive iteration limit
def adaptive_max_iterations(critique: Critique) -> int:
    avg_score = (critique.accuracy_score + critique.completeness_score + critique.clarity_score) / 3
    
    if avg_score >= 0.9:
        return 1  # Already good
    elif avg_score >= 0.7:
        return 2  # Minor improvements
    else:
        return 3  # Needs more work
```

### Satisfaction Threshold

When is "good enough" good enough?

```python
def is_satisfactory(critique: Critique, threshold: float = 0.8) -> bool:
    """Check if response meets quality threshold."""
    return (
        critique.accuracy_score >= threshold and
        critique.completeness_score >= threshold and
        critique.clarity_score >= threshold
    )
```

### Critique Depth

Balance thoroughness with efficiency:

```python
# Quick critique for simple queries
quick_critique = "Rate 1-5: Is this response good? If not, what's the main issue?"

# Deep critique for complex queries
deep_critique = """Provide detailed analysis:
1. Accuracy (score 1-5, specific errors if any)
2. Completeness (score 1-5, what's missing)
3. Clarity (score 1-5, confusing parts)
4. Evidence (score 1-5, unsupported claims)
5. Overall assessment and top 3 improvements"""
```

## Best Practices

1. **Set clear criteria.** Vague critique prompts produce vague feedback.

2. **Limit iterations.** Diminishing returns after 2-3 rounds.

3. **Use structured output.** Parse critiques reliably.

4. **Log iterations.** Track what's being improved across rounds.

5. **Consider using different models.** Critique with a different/stronger model than generation.

6. **Don't over-reflect on simple tasks.** Match reflection depth to task importance.

## Related Patterns

- **ReAct**: Reasoning pattern that reflection builds on
- **Self-Consistency**: Generate multiple and compare
- **Chain-of-Verification**: Verify claims explicitly
- **Constitutional AI**: Reflection against principles
