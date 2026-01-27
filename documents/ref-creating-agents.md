# Recipe: Creating Agents

## Goal

Build an agent that can reason, use tools, and accomplish tasks autonomously.

## Quick Start

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search for information about a topic."""
    return f"Results for: {query}"

@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    return str(eval(expression))

# Create agent
agent = create_agent(
    model="gpt-4o-mini",
    tools=[search, calculate],
    system_prompt="You are a helpful assistant."
)

# Run agent
response = agent.invoke({
    "messages": [{"role": "user", "content": "What is 15% of 240?"}]
})

print(response["messages"][-1].content)
```

## LangChain create_agent

### Basic Agent

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Implementation
    return f"Weather in {city}: 72°F, sunny"

@tool
def search_documents(query: str) -> str:
    """Search the knowledge base for information."""
    results = vector_store.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in results])

agent = create_agent(
    model="gpt-4o-mini",
    tools=[get_weather, search_documents],
    system_prompt="""You are a helpful assistant with access to weather and document search.
    
Use search_documents for questions about specific topics.
Use get_weather for weather-related questions.
Think step-by-step before using tools."""
)
```

### Running the Agent

```python
# Simple invocation
response = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]
})

# Extract final answer
final_message = response["messages"][-1]
print(final_message.content)

# Multi-turn conversation
messages = [{"role": "user", "content": "Hello"}]
response = agent.invoke({"messages": messages})

# Continue conversation
messages = response["messages"]
messages.append({"role": "user", "content": "Now search for RAG"})
response = agent.invoke({"messages": messages})
```

### Streaming

```python
for event in agent.stream({
    "messages": [{"role": "user", "content": "Search for machine learning"}]
}):
    print(event)
```

## LangGraph Agent (More Control)

### Basic ReAct Agent

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Found: {query} results"

tools = [search]
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def agent_node(state: AgentState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# Build graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

agent = graph.compile()

# Run
result = agent.invoke({
    "messages": [{"role": "user", "content": "Search for AI"}]
})
```

### With Memory (Checkpointing)

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
agent = graph.compile(checkpointer=checkpointer)

# Each thread_id maintains separate conversation
config = {"configurable": {"thread_id": "user-123"}}

# First message
result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "My name is Alice"}]},
    config
)

# Continues same conversation
result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    config
)
```

### With Custom State

```python
class CustomState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str
    iteration_count: int
    max_iterations: int

def agent_node(state: CustomState):
    # Access custom state
    context = state.get("context", "")
    
    messages = state["messages"]
    if context:
        # Inject context into system message
        messages = [{"role": "system", "content": f"Context: {context}"}] + messages
    
    response = model.invoke(messages)
    
    return {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def should_continue(state: CustomState):
    # Check iteration limit
    if state.get("iteration_count", 0) >= state.get("max_iterations", 10):
        return END
    
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END
```

## Agent Patterns

### Agentic RAG

```python
from langchain.tools import tool

@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information.
    Use this before answering questions about specific topics."""
    results = vector_store.similarity_search(query, k=3)
    if not results:
        return "No relevant information found."
    return "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc in results
    ])

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_knowledge_base],
    system_prompt="""You are a helpful assistant with access to a knowledge base.

IMPORTANT: Always search the knowledge base before answering questions about specific topics.
Cite your sources when providing information from the knowledge base."""
)
```

### Multi-Tool Agent

```python
@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    # Web search implementation
    pass

@tool
def search_documents(query: str) -> str:
    """Search internal documents for information."""
    pass

@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations."""
    return str(eval(expression))

@tool
def get_date() -> str:
    """Get the current date."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d")

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_web, search_documents, calculate, get_date],
    system_prompt="""You are a helpful assistant with multiple capabilities.

Choose the right tool for each task:
- search_web: For current events or general information
- search_documents: For internal/specific knowledge
- calculate: For math operations
- get_date: For current date

Think step-by-step about which tool to use."""
)
```

### Agent with Reflection

```python
from langgraph.graph import StateGraph, END

class ReflectiveState(TypedDict):
    messages: Annotated[list, add_messages]
    draft_response: str
    reflection: str
    iteration: int

def generate_node(state: ReflectiveState):
    response = model.invoke(state["messages"])
    return {"draft_response": response.content, "iteration": state.get("iteration", 0) + 1}

def reflect_node(state: ReflectiveState):
    reflection_prompt = f"""Review this response and identify improvements:

Response: {state["draft_response"]}

List specific issues and how to fix them. Say "APPROVED" if the response is good."""
    
    reflection = model.invoke([{"role": "user", "content": reflection_prompt}])
    return {"reflection": reflection.content}

def should_continue(state: ReflectiveState):
    if "APPROVED" in state.get("reflection", ""):
        return "finalize"
    if state.get("iteration", 0) >= 3:
        return "finalize"
    return "generate"

def finalize_node(state: ReflectiveState):
    return {"messages": [{"role": "assistant", "content": state["draft_response"]}]}

graph = StateGraph(ReflectiveState)
graph.add_node("generate", generate_node)
graph.add_node("reflect", reflect_node)
graph.add_node("finalize", finalize_node)

graph.set_entry_point("generate")
graph.add_edge("generate", "reflect")
graph.add_conditional_edges("reflect", should_continue)
graph.add_edge("finalize", END)
```

## Error Handling

### Graceful Tool Failures

```python
@tool
def risky_operation(input: str) -> str:
    """An operation that might fail."""
    try:
        result = perform_operation(input)
        return result
    except ConnectionError:
        return "Error: Unable to connect. Please try again."
    except ValueError as e:
        return f"Error: Invalid input - {str(e)}"
    except Exception as e:
        return f"Error: Operation failed - {str(e)}"
```

### Agent-Level Error Handling

```python
def safe_invoke(agent, messages, max_retries=3):
    """Invoke agent with retry logic."""
    
    for attempt in range(max_retries):
        try:
            return agent.invoke({"messages": messages})
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed: {e}")
            continue
```

## Testing Agents

### Unit Testing Tools

```python
import pytest

def test_search_tool():
    result = search("test query")
    assert isinstance(result, str)
    assert len(result) > 0

def test_calculate_tool():
    result = calculate("2 + 2")
    assert result == "4"
```

### Integration Testing

```python
def test_agent_uses_correct_tool():
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Calculate 15% of 200"}]
    })
    
    # Check that calculate tool was used
    messages = response["messages"]
    tool_calls = [m for m in messages if hasattr(m, "tool_calls") and m.tool_calls]
    
    assert len(tool_calls) > 0
    assert any("calculate" in str(tc) for tc in tool_calls)
```

## Debugging

### Verbose Output

```python
# LangChain verbose
import langchain
langchain.debug = True

# Or use callbacks
from langchain_core.callbacks import BaseCallbackHandler

class DebugCallback(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"Tool: {serialized['name']}, Input: {input_str}")
    
    def on_tool_end(self, output, **kwargs):
        print(f"Tool output: {output}")

response = agent.invoke(
    {"messages": messages},
    config={"callbacks": [DebugCallback()]}
)
```

### LangSmith Tracing

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-key"

# All agent invocations are now traced
response = agent.invoke({"messages": messages})
# View traces at smith.langchain.com
```

## Best Practices

1. **Clear tool descriptions.** The model uses descriptions to decide when to use tools.

2. **Specific system prompts.** Guide the agent on when and how to use tools.

3. **Handle errors gracefully.** Tools should return error messages, not raise exceptions.

4. **Set iteration limits.** Prevent infinite loops with max iterations.

5. **Test tools independently.** Verify tools work before integrating into agent.

6. **Use tracing.** LangSmith helps debug agent behavior.

## Related Recipes

- **Defining Tools**: Creating effective tools
- **Similarity Search**: RAG tool implementation
- **Streaming**: Real-time agent output
