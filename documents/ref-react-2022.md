# Paper Summary: ReAct (Reasoning + Acting)

## Citation

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). **ReAct: Synergizing Reasoning and Acting in Language Models**. *arXiv preprint arXiv:2210.03629*.

## One-Sentence Summary

ReAct interleaves reasoning traces (thinking about what to do) with actions (doing things) to solve complex tasks more effectively than either reasoning or acting alone.

## The Problem

Prior approaches to using language models for tasks had limitations:

**Reasoning-only (Chain-of-Thought):**
- Models reason through problems step-by-step
- But they can't interact with the environment
- Errors in reasoning propagate without correction
- No access to external information

**Acting-only (Action models):**
- Models can take actions (search, use tools)
- But they don't explain their reasoning
- Difficult to debug or understand failures
- May take suboptimal action sequences

## The Solution

ReAct synergizes reasoning and acting by interleaving them:

```
Thought: I need to find information about X
Action: Search[X]
Observation: [search results]
Thought: The results mention Y, which helps answer the question
Action: Lookup[Y]
Observation: [lookup results]
Thought: Now I have enough information to answer
Action: Finish[answer]
```

The key insight: reasoning traces help the model:
1. Track progress toward the goal
2. Handle exceptions and adjust plans
3. Interface with external knowledge
4. Make decisions interpretable

## The Pattern

ReAct follows a simple loop:

1. **Thought**: Reason about current state and what to do next
2. **Action**: Execute an action (search, lookup, finish, etc.)
3. **Observation**: Receive result of action
4. Repeat until task complete

```
Query → Thought → Action → Observation → Thought → Action → ... → Answer
```

## Key Results

ReAct outperformed baseline approaches on knowledge-intensive tasks:

| Method | HotpotQA | FEVER |
|--------|----------|-------|
| Standard prompting | 28.7 | 57.1 |
| Chain-of-Thought | 29.4 | 56.3 |
| Act-only | 25.7 | 58.9 |
| **ReAct** | **34.2** | **60.9** |

More importantly, ReAct:
- Reduced hallucination by grounding in retrieved facts
- Made reasoning interpretable (you can see why it made decisions)
- Recovered from errors by observing and adjusting

## Example Trace

**Question**: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?

```
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector extends into, then find the elevation range of that area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building in Colorado...The eastern sector extends into the High Plains.
Thought 2: The eastern sector extends into the High Plains. I need to find the elevation range of the High Plains.
Action 2: Search[High Plains]
Observation 2: High Plains refers to one of two distinct land regions: the High Plains (United States)...
Thought 3: I need to search High Plains (United States) to find the elevation range.
Action 3: Search[High Plains (United States)]
Observation 3: The High Plains are a subregion of the Great Plains...elevation ranges from around 1,800 to 7,000 ft.
Thought 4: The elevation range for High Plains is 1,800 to 7,000 ft.
Action 4: Finish[1,800 to 7,000 ft]
```

## Why This Matters for AI Engineering

ReAct is the foundation for modern AI agents:

1. **Tool use pattern**: Thought → Action → Observation became standard
2. **Interpretability**: Reasoning traces show why agents do what they do
3. **Error recovery**: Observations enable course correction
4. **LangChain/LangGraph**: Built around the ReAct pattern

Every agent framework implements some variant of ReAct.

## Practical Implications

**What the paper got right:**
- Interleaving reasoning with acting improves both
- Explicit reasoning traces help debugging
- External actions ground responses in reality

**What's evolved since:**
- Modern agents use function calling rather than text-based actions
- Multi-step planning augments pure ReAct
- Reflection and self-critique enhance the pattern
- Tool schemas are more structured

## Implementation Notes

### Basic ReAct Prompt Structure

```
Answer the following question by reasoning and acting.

Available actions:
- Search[query]: Search for information
- Lookup[term]: Look up a term in retrieved content
- Finish[answer]: Provide final answer

Question: {question}

Thought 1:
```

### Modern Implementation (LangChain)

```python
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_tool, lookup_tool],
    system_prompt="Think step-by-step before using tools."
)

# The agent automatically follows ReAct pattern
response = agent.invoke({"messages": [{"role": "user", "content": question}]})
```

## Connection to StudyBuddy

StudyBuddy v3 implements ReAct through LangChain's `create_agent`:
- **Thought**: Agent reasons about whether to search study materials
- **Action**: Calls `search_materials` tool
- **Observation**: Receives relevant content
- **Answer**: Generates explanation based on observations

The reasoning trace is exposed in the UI so students can see how StudyBuddy arrived at its explanation.

## Key Quotes

> "ReAct prompting induces language models to generate both verbal reasoning traces and actions pertaining to a task in an interleaved manner."

> "The reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while the actions allow it to interface with external sources."

## Comparison with Related Work

| Approach | Reasoning | Acting | Interpretable |
|----------|-----------|--------|---------------|
| Standard LM | ❌ | ❌ | ❌ |
| Chain-of-Thought | ✓ | ❌ | ✓ |
| Act-only | ❌ | ✓ | ❌ |
| **ReAct** | ✓ | ✓ | ✓ |

## Further Reading

- **Chain-of-Thought Prompting** (Wei et al., 2022): Reasoning-only baseline
- **WebGPT** (Nakano et al., 2021): Acting with web browsing
- **Toolformer** (Schick et al., 2023): Learning when to use tools
- **Reflexion** (Shinn et al., 2023): Adding self-reflection to ReAct

## BibTeX

```bibtex
@article{yao2022react,
  title={ReAct: Synergizing Reasoning and Acting in Language Models},
  author={Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and Du, Nan and Shafran, Izhak and Narasimhan, Karthik and Cao, Yuan},
  journal={arXiv preprint arXiv:2210.03629},
  year={2022}
}
```
