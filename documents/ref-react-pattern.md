# The ReAct Pattern

## What Is ReAct?

ReAct stands for Reasoning + Acting. It's a pattern for building agents that interleave thinking with doing. Before each action, the agent generates a reasoning trace explaining why it's taking that action. After observing results, it reasons about what those results mean before deciding the next step.

The pattern was introduced in the 2022 paper "ReAct: Synergizing Reasoning and Acting in Language Models" by Yao et al. It has become the foundation for most production agent systems.

## Why ReAct Works

Language models are trained to predict the next token. When you ask directly for an answer, the model must jump straight to a conclusion. Complex tasks require intermediate reasoning steps. By explicitly generating those steps, you give the model space to work through the problem.

ReAct also provides interpretability. You can see exactly why the agent chose each action. When things go wrong, you can diagnose where reasoning failed. This transparency is essential for debugging and improving agents.

## The ReAct Loop

```
┌────────────────────────────────────────────────────────┐
│ Input: User question or task                           │
└────────────────────────┬───────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────┐
│ THOUGHT: Reason about what to do                       │
│ "I need to find information about X..."                │
└────────────────────────┬───────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────┐
│ ACTION: Choose and execute a tool                      │
│ search_database("X")                                   │
└────────────────────────┬───────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────┐
│ OBSERVATION: Receive tool results                      │
│ "Found 3 documents about X..."                         │
└────────────────────────┬───────────────────────────────┘
                         ▼
                    Is task complete?
                    /              \
                  No               Yes
                  /                  \
         Back to THOUGHT      FINAL ANSWER
```

## ReAct Trace Format

A typical ReAct trace looks like this:

```
Question: What is the capital of France and what's its population?

Thought: This question has two parts. I know the capital of France is 
Paris from my training, but I should verify the current population 
since that changes over time. Let me search for Paris population data.

Action: search_web
Action Input: {"query": "Paris France population 2025"}

Observation: Paris has a population of approximately 2.1 million in 
the city proper, with the metropolitan area having about 12 million.

Thought: I now have both pieces of information. The capital is Paris, 
and I have current population figures. I can provide a complete answer.

Final Answer: The capital of France is Paris. The city proper has a 
population of approximately 2.1 million people, while the greater 
Paris metropolitan area is home to about 12 million residents.
```

## Implementing ReAct

### With LangChain create_agent

LangChain's `create_agent` implements ReAct internally. You provide tools and a system prompt; the framework handles the thought-action-observation loop.

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information about a topic."""
    results = vector_store.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in results])

@tool
def get_current_date() -> str:
    """Get today's date."""
    from datetime import date
    return date.today().isoformat()

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_knowledge_base, get_current_date],
    system_prompt="""You are a helpful research assistant. 
    Think step by step before taking actions.
    Always search the knowledge base before answering factual questions."""
)

# The agent will automatically use ReAct pattern
response = agent.invoke({
    "messages": [{"role": "user", "content": "What are embeddings?"}]
})
```

### Manual ReAct Implementation

For more control, you can implement ReAct manually:

```python
from openai import OpenAI

client = OpenAI()

REACT_PROMPT = """You are an assistant that thinks step by step.

Available tools:
- search(query): Search the knowledge base
- calculate(expression): Evaluate math expressions

Format your response as:
Thought: [your reasoning]
Action: [tool name]
Action Input: [tool parameters]

After receiving an Observation, continue with another Thought.
When you have enough information, respond with:
Thought: [final reasoning]
Final Answer: [your response to the user]

Question: {question}
"""

def run_react_agent(question: str, max_iterations: int = 5):
    conversation = REACT_PROMPT.format(question=question)
    
    for i in range(max_iterations):
        response = client.responses.create(
            model="gpt-4o-mini",
            input=conversation
        )
        
        assistant_message = response.output_text
        conversation += f"\n{assistant_message}"
        
        # Check for final answer
        if "Final Answer:" in assistant_message:
            return assistant_message.split("Final Answer:")[-1].strip()
        
        # Parse and execute action
        action, action_input = parse_action(assistant_message)
        observation = execute_tool(action, action_input)
        
        conversation += f"\nObservation: {observation}"
    
    return "Max iterations reached without final answer"
```

## ReAct vs. Other Patterns

### ReAct vs. Chain-of-Thought (CoT)

CoT generates reasoning before a single response. ReAct interleaves reasoning with actions across multiple steps.

```
CoT: Think → Answer (one shot)
ReAct: Think → Act → Observe → Think → Act → ... → Answer (iterative)
```

Use CoT for single-step reasoning. Use ReAct when you need to gather information or take actions.

### ReAct vs. Plan-and-Execute

Plan-and-Execute creates a full plan upfront, then executes each step. ReAct decides the next step based on the current state.

```
Plan-and-Execute: Plan[1,2,3,4] → Do 1 → Do 2 → Do 3 → Do 4
ReAct: Think → Do 1 → Observe → Think → Do 2 → Observe → ...
```

Plan-and-Execute is better when the task is well-defined and steps are predictable. ReAct is better when the path depends on what you discover along the way.

### ReAct vs. Reflexion

Reflexion adds self-critique after generating responses. The agent evaluates its own output and revises if needed.

```
ReAct: Think → Act → Observe → Answer
Reflexion: Think → Act → Observe → Answer → Critique → Revise
```

Reflexion builds on ReAct. Use it when response quality is critical and you can afford extra latency.

## Prompt Engineering for ReAct

The system prompt significantly affects ReAct behavior. Key elements:

### Tool Descriptions

Clear, specific descriptions help the agent choose correctly:

```python
@tool
def search_documents(query: str) -> str:
    """
    Search the indexed documents for information about a topic.
    
    Use this tool when you need to find specific information from 
    the document collection. Good for factual questions, definitions,
    and explanations that should be grounded in source material.
    
    Args:
        query: The search term or question to look up
    
    Returns:
        Relevant excerpts from matching documents with source citations
    """
```

### Thinking Instructions

Guide how the agent should reason:

```
Before taking any action, think through:
1. What information do I need?
2. Which tool is most appropriate?
3. What query/input will get the best results?

After each observation, consider:
1. Did I get the information I needed?
2. Is this information sufficient to answer the question?
3. Do I need to search for more details?
```

### Examples (Few-Shot)

Include example traces in the prompt:

```
Example:
User: What's the weather in Tokyo?
Thought: I need current weather data, which requires the weather API.
Action: get_weather
Action Input: {"city": "Tokyo"}
Observation: Tokyo: 18°C, partly cloudy, humidity 65%
Thought: I have the weather information. I can now answer.
Final Answer: The current weather in Tokyo is 18°C and partly cloudy 
with 65% humidity.
```

## Common Issues and Solutions

### Agent Doesn't Use Tools

**Problem:** Agent answers from training data instead of using tools.

**Solutions:**
- Strengthen system prompt: "You MUST use tools for factual questions"
- Add explicit instructions about when to use each tool
- Use examples showing tool use

### Agent Uses Wrong Tool

**Problem:** Agent calls inappropriate tool for the task.

**Solutions:**
- Improve tool descriptions with specific use cases
- Add "When to use" and "When NOT to use" guidance
- Reduce number of tools to minimize confusion

### Agent Loops Indefinitely

**Problem:** Agent keeps searching without reaching a conclusion.

**Solutions:**
- Set max_iterations limit
- Add "If you've searched 3 times without finding an answer, say so"
- Detect repeated identical queries

### Verbose/Unfocused Reasoning

**Problem:** Thought sections are too long, wasting tokens.

**Solutions:**
- Prompt for concise thinking: "Keep thoughts to 1-2 sentences"
- Provide examples of concise reasoning
- Use structured output format

## Observability

Understanding what your agent is doing requires good logging:

```python
def run_react_with_logging(question: str):
    steps = []
    
    for iteration, (thought, action, observation) in enumerate(agent_loop()):
        step = {
            "iteration": iteration,
            "thought": thought,
            "action": action.name if action else None,
            "action_input": action.input if action else None,
            "observation": observation,
            "timestamp": datetime.now().isoformat()
        }
        steps.append(step)
        
        # Log to monitoring system
        logger.info(f"Agent step {iteration}", extra=step)
    
    return {
        "answer": final_answer,
        "steps": steps,
        "total_iterations": len(steps)
    }
```

LangSmith provides built-in tracing for LangChain agents, showing the full thought-action-observation sequence.

## Best Practices

1. **Keep reasoning focused.** Short, purposeful thoughts are better than verbose explanations.

2. **Limit tools to what's needed.** More tools mean more chances for confusion.

3. **Set iteration bounds.** Prevent infinite loops with max_iterations.

4. **Log traces for debugging.** You can't improve what you can't see.

5. **Test with diverse queries.** Edge cases reveal prompt and tool description issues.

6. **Use stronger models for complex tasks.** ReAct requires good reasoning capabilities.

## Related Concepts

- **Agents**: Systems that use ReAct for decision-making
- **Tool Use**: How agents execute actions
- **Chain-of-Thought**: Single-step reasoning (vs. multi-step ReAct)
- **LangGraph**: Building complex agent flows beyond basic ReAct
- **Observability**: Monitoring agent behavior in production
