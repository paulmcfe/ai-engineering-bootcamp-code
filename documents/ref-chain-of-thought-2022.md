# Chain-of-Thought Prompting (2022)

## Paper Details

**Title:** Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

**Authors:** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou (Google Research)

**Published:** January 2022

**Link:** https://arxiv.org/abs/2201.11903

## Key Insight

Asking language models to show their reasoning step-by-step dramatically improves performance on complex tasks. Instead of prompting for a direct answer, you prompt for a "chain of thought" leading to the answer.

## The Problem

Standard prompting struggles with multi-step reasoning:

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?

Standard prompting → Model often gets wrong answer
```

The model tries to jump directly to the answer without working through the logic.

## The Solution

Chain-of-thought prompting adds intermediate reasoning steps:

```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?

A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 
   6 tennis balls. 5 + 6 = 11. The answer is 11.
```

By generating the reasoning trace, the model maintains context for each step.

## How It Works

### Few-Shot Chain-of-Thought

Provide examples that demonstrate step-by-step reasoning:

```python
prompt = """
Q: A juggler can juggle 16 balls. Half the balls are golf balls,
   and half the golf balls are blue. How many blue golf balls?

A: The juggler has 16 balls. Half are golf balls: 16 / 2 = 8.
   Half the golf balls are blue: 8 / 2 = 4. The answer is 4.

Q: [Your actual question here]

A:
"""
```

The model learns to follow the same reasoning pattern.

### Zero-Shot Chain-of-Thought

Simply add "Let's think step by step" to the prompt:

```python
prompt = """
Q: [Your question]

A: Let's think step by step.
"""
```

This simple phrase triggers step-by-step reasoning without examples.

## Results

Performance improvements on reasoning benchmarks:

| Task | Standard | Chain-of-Thought |
|------|----------|------------------|
| GSM8K (math) | 17.9% | 58.1% |
| SVAMP (math) | 58.8% | 79.0% |
| StrategyQA | 65.4% | 73.0% |
| Date Understanding | 49.3% | 67.5% |

Improvements are most dramatic on complex, multi-step problems.

## Key Findings

### Emergent Ability

Chain-of-thought only helps large models. Small models (< 10B parameters) don't benefit and sometimes perform worse. The technique emerges as an ability in sufficiently large models.

### Task Complexity Matters

Simple tasks don't benefit from chain-of-thought. The technique helps most when:
- Multiple reasoning steps are required
- Intermediate results inform later steps
- The answer isn't immediately obvious

### Reasoning Quality

Models can produce coherent reasoning traces even when the final answer is wrong. The quality of reasoning correlates with but doesn't guarantee correctness.

## Implementation

### Basic Chain-of-Thought

```python
from openai import OpenAI

client = OpenAI()

def solve_with_cot(question: str) -> str:
    """Solve problem using chain-of-thought."""
    
    response = client.responses.create(
        model="gpt-4o-mini",
        input=f"{question}\n\nLet's think step by step."
    )
    
    return response.output_text
```

### Few-Shot Chain-of-Thought

```python
def solve_with_examples(question: str) -> str:
    """Solve using few-shot chain-of-thought examples."""
    
    examples = """
Example 1:
Q: If there are 3 cars in the parking lot and 2 more arrive, 
   how many cars are in the parking lot?
A: There are originally 3 cars. 2 more arrive. 3 + 2 = 5.
   The answer is 5.

Example 2:
Q: Olivia has $23. She bought 5 bagels for $3 each. 
   How much money does she have left?
A: Olivia had $23. 5 bagels at $3 each costs 5 * 3 = $15.
   She has $23 - $15 = $8 left. The answer is $8.
"""
    
    response = client.responses.create(
        model="gpt-4o-mini",
        input=f"{examples}\nQ: {question}\nA:"
    )
    
    return response.output_text
```

### Extracting Final Answer

```python
import re

def extract_answer(cot_response: str) -> str:
    """Extract final answer from chain-of-thought response."""
    
    # Look for "The answer is X" pattern
    match = re.search(r"[Tt]he answer is[:\s]*(.+?)\.?$", cot_response)
    if match:
        return match.group(1).strip()
    
    # Fallback: return last sentence
    sentences = cot_response.split(".")
    return sentences[-1].strip() if sentences else cot_response
```

## Variations

### Self-Consistency

Generate multiple chain-of-thought paths, take majority vote on answers:

```python
def solve_with_self_consistency(question: str, n: int = 5) -> str:
    """Generate multiple reasoning paths, vote on answer."""
    
    answers = []
    for _ in range(n):
        response = solve_with_cot(question)
        answer = extract_answer(response)
        answers.append(answer)
    
    # Majority vote
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common
```

### Least-to-Most Prompting

Decompose into subproblems, solve in order:

```python
def least_to_most(question: str) -> str:
    """Break into subproblems, solve from simplest to most complex."""
    
    # First: decompose the problem
    decompose_prompt = f"""
Break this problem into simpler subproblems:
{question}

List the subproblems from simplest to most complex:
"""
    subproblems = get_subproblems(decompose_prompt)
    
    # Solve each subproblem, building context
    context = ""
    for subproblem in subproblems:
        solve_prompt = f"""
Previous solutions:
{context}

Now solve: {subproblem}
"""
        solution = solve_with_cot(solve_prompt)
        context += f"\n{subproblem}: {solution}"
    
    return context
```

## Connection to Agents

Chain-of-thought is foundational to agent reasoning:

- **ReAct** interleaves chain-of-thought with actions
- **Reflection** uses chain-of-thought to critique outputs
- **Planning** uses chain-of-thought to generate plans

Modern agents use chain-of-thought implicitly through their reasoning traces.

## Practical Tips

1. **Use for complex tasks.** Don't bother for simple lookups or classifications.

2. **Match example difficulty.** Examples should be similar complexity to actual task.

3. **Be specific about format.** If you need structured output, show it in examples.

4. **Combine with self-consistency.** Multiple paths reduce variance in answers.

5. **Inspect reasoning.** The trace helps debug wrong answers.

## Limitations

- Adds tokens (cost and latency)
- Doesn't help small models
- Can produce plausible but wrong reasoning
- May overthink simple problems

## Related Papers

- **ReAct (2022)**: Adds actions to chain-of-thought
- **Self-Consistency (2022)**: Multiple reasoning paths
- **Least-to-Most (2022)**: Problem decomposition
- **Tree of Thoughts (2023)**: Branching reasoning paths

## Why This Matters for AI Engineering

Chain-of-thought is why modern prompting works. Understanding it helps you:
- Design better prompts for complex tasks
- Debug model reasoning when answers are wrong
- Build agents that show their work
- Improve reliability through self-consistency
