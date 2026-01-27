# Agents and Agency

## What Is an Agent?

An agent is a system that operates autonomously to achieve goals. It perceives its environment, reasons about what to do, takes actions that affect the environment, and adapts based on results. The key characteristic is autonomy—you give an agent a goal and it figures out how to achieve it.

A simple chatbot isn't an agent. It takes input, generates output, done. No reasoning loop, no decisions about what to do next, no adaptation. An agent can plan multiple steps, use tools, reflect on results, and adjust strategy.

## Agents vs. Workflows

This distinction matters for system design.

**Workflows** execute predefined sequences. Receive input → step 1 → step 2 → step 3 → output. Each step is predetermined. The system follows a script. Workflows are deterministic and predictable—great for well-defined, repeatable processes.

**Agents** make decisions. Given the current situation, what's the best next action? The path emerges from reasoning, not prescription. Agents are adaptive—they handle novel situations and edge cases that workflows miss.

```
Workflow: input → fixed_step_1 → fixed_step_2 → fixed_step_3 → output

Agent: input → reason → (action → observe → reason)* → output
```

In practice, combine both. Use workflows for predictable parts, agents for parts requiring judgment.

## The Agent Loop

Every agent follows this fundamental cycle:

```
┌─────────────────────────────────────────────┐
│                  PERCEIVE                    │
│         Understand current situation         │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│                   REASON                     │
│           Decide what to do next            │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│                    ACT                       │
│        Execute chosen action/tool           │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│                  OBSERVE                     │
│         See results, update state           │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
              Done? ──────► Yes ──► Return result
                │
                No
                │
                └──────────► Back to PERCEIVE
```

This loop continues until the agent achieves its goal or determines it can't proceed.

## Core Components

### Perception

How the agent understands the current state. For conversational agents, this means reading messages and context. For code agents, reading files and error messages. Perception combines:

- Current input (user message, trigger event)
- Conversation/task history
- Available tools and their descriptions
- Goal or objective

### Reasoning

Where decisions happen. The agent considers:
- What am I trying to accomplish?
- What information do I have?
- What actions are available?
- Which action makes the most progress?

This is typically an LLM processing a carefully constructed prompt that includes all relevant context.

### Action

Executing the chosen step. Actions include:
- Calling external tools (search, APIs, databases)
- Generating content (text, code, summaries)
- Asking clarifying questions
- Delegating to sub-agents
- Declaring task complete

### Observation

Processing action results. Did the tool return useful information? Did an error occur? Is the result what we expected? Observations feed back into perception for the next iteration.

## When to Use Agents

**Good fit:**
- Multi-step tasks where steps depend on intermediate results
- Tasks requiring tool selection based on context
- Multiple valid approaches requiring judgment
- Error recovery and adaptation needed
- Solution path unknown in advance

**Poor fit:**
- Well-defined, predictable sequences
- Simple transformations (translate, classify, extract)
- Tasks with no external tools or actions
- Speed-critical operations (agents add latency)

## Agent Capabilities

### Tool Use

Agents interact with the world through tools—functions they can call to accomplish tasks.

```python
from langchain.tools import tool

@tool
def search_database(query: str) -> str:
    """Search the knowledge base for relevant information."""
    # Implementation
    results = db.search(query)
    return format_results(results)

@tool  
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the specified recipient."""
    # Implementation
    email_service.send(to, subject, body)
    return f"Email sent to {to}"
```

Tools give agents capabilities beyond text generation: searching, writing files, calling APIs, executing code.

### Multi-Step Reasoning

Agents can break complex tasks into steps:

```
User: "Find papers about RAG, summarize the top 3, and draft a literature review."

Agent thinking:
1. First, I need to search for RAG papers
2. [Calls search tool] → Gets 10 papers
3. Need to read and evaluate these to find top 3
4. [Reads paper 1] → Relevant, about retrieval methods
5. [Reads paper 2] → Less relevant, more about generation
... continues until task complete
```

### Adaptation

When things don't go as expected, agents can adjust:

```
Agent: [Searches "RAG papers 2024"]
Observation: No results found

Agent thinking: Search returned nothing. Let me try broader terms.
Agent: [Searches "retrieval augmented generation"]
Observation: Found 47 results

Agent thinking: That worked better. Now I can proceed.
```

## Creating Agents (LangChain 1.0)

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool
def search_docs(query: str) -> str:
    """Search documentation for information."""
    results = vector_store.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in results])

# Create agent
agent = create_agent(
    model="gpt-4o-mini",
    tools=[calculator, search_docs],
    system_prompt="You are a helpful assistant. Use tools when needed."
)

# Run agent
response = agent.invoke({
    "messages": [{"role": "user", "content": "What's 15% of 240?"}]
})
```

## Agent Architectures

### ReAct (Reasoning + Acting)

Interleaves thinking and acting. The agent explicitly generates reasoning traces before each action.

```
Thought: I need to find information about embeddings.
Action: search_docs("embeddings")
Observation: [search results]
Thought: This explains what embeddings are. Now I can answer.
Final Answer: Embeddings are...
```

### Plan-and-Execute

Creates a complete plan upfront, then executes each step.

```
Plan:
1. Search for embedding documentation
2. Search for vector database documentation  
3. Synthesize into explanation

Execute:
Step 1: [search] → results
Step 2: [search] → results
Step 3: [generate] → final answer
```

### Reflexion

Adds self-critique. After generating a response, the agent evaluates its own work and iterates if needed.

```
Response: [initial answer]
Reflection: This answer is incomplete—I didn't explain the distance metrics.
Revised response: [better answer with distance metrics]
```

## Common Pitfalls

### Infinite Loops

Agent keeps trying the same unsuccessful action. Solution: iteration limits and loop detection.

### Tool Misuse

Agent calls wrong tool or passes bad parameters. Solution: clear tool descriptions, parameter validation, examples in prompts.

### Context Overflow

Long agent runs fill the context window. Solution: summarization, selective history, external memory.

### Hallucinated Actions

Agent claims to have done something it didn't. Solution: verify tool calls actually executed, check return values.

## Best Practices

1. **Start simple.** Single-tool agents before multi-tool. Short tasks before long ones.

2. **Write clear tool descriptions.** The LLM only knows what you tell it. Descriptions should explain when and how to use each tool.

3. **Set iteration limits.** Prevent runaway agents with max_iterations parameter.

4. **Log everything.** Agent behavior is hard to debug without detailed traces.

5. **Test edge cases.** What happens when tools fail? When queries are ambiguous? When no information is found?

6. **Use appropriate models.** More capable models (GPT-4, Claude) make better agents than smaller models.

## Related Concepts

- **ReAct Pattern**: Specific reasoning-acting pattern for agents
- **Tool Use**: How agents interact with external systems
- **Multi-Agent Systems**: Multiple agents collaborating
- **Memory**: Persistent state across agent interactions
- **LangGraph**: Framework for complex agent architectures
