# LangGraph Quick Reference

## Overview

LangGraph is a framework for building stateful, multi-step agent applications. It models workflows as graphs where nodes are functions and edges define control flow. Version 1.0 (October 2025) introduced significant improvements to the API.

## Installation

```bash
uv pip install langgraph langgraph-supervisor
```

## Core Concepts

### State

State is a TypedDict that flows through the graph:

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Accumulates messages
    context: str
    iteration: int
```

### Nodes

Nodes are functions that transform state:

```python
def process_node(state: AgentState) -> AgentState:
    """Node that processes state."""
    # Read from state
    messages = state["messages"]
    
    # Do work
    result = llm.invoke(messages)
    
    # Return state updates (merged with existing state)
    return {"messages": [result]}
```

### Edges

Edges connect nodes and control flow:

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("process", process_node)
graph.add_node("respond", respond_node)

# Simple edge (always follows)
graph.add_edge("process", "respond")

# Conditional edge (branches based on state)
def should_continue(state: AgentState) -> str:
    if state["iteration"] > 3:
        return "end"
    return "continue"

graph.add_conditional_edges(
    "process",
    should_continue,
    {"continue": "process", "end": END}
)
```

## Basic Graph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    input: str
    output: str

def step_one(state: State) -> State:
    return {"output": f"Processed: {state['input']}"}

def step_two(state: State) -> State:
    return {"output": state["output"] + " - Complete"}

# Build graph
graph = StateGraph(State)
graph.add_node("step_one", step_one)
graph.add_node("step_two", step_two)

graph.set_entry_point("step_one")
graph.add_edge("step_one", "step_two")
graph.add_edge("step_two", END)

# Compile and run
app = graph.compile()
result = app.invoke({"input": "Hello"})
print(result["output"])  # "Processed: Hello - Complete"
```

## Agent with Tools

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from typing import Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

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
```

## Conditional Routing

```python
from typing import Literal

def router(state: AgentState) -> Literal["search", "calculate", "respond"]:
    """Route based on state."""
    last_message = state["messages"][-1].content.lower()
    
    if "search" in last_message:
        return "search"
    elif "calculate" in last_message:
        return "calculate"
    return "respond"

graph.add_conditional_edges(
    "analyze",
    router,
    {
        "search": "search_node",
        "calculate": "calc_node",
        "respond": "respond_node"
    }
)
```

## Checkpointing (Memory)

```python
from langgraph.checkpoint.memory import MemorySaver

# Create checkpointer
checkpointer = MemorySaver()

# Compile with checkpointer
app = graph.compile(checkpointer=checkpointer)

# Run with thread_id for persistence
config = {"configurable": {"thread_id": "user-123"}}

# First invocation
result1 = app.invoke({"messages": [("user", "Hi")]}, config)

# Second invocation continues conversation
result2 = app.invoke({"messages": [("user", "What did I say?")]}, config)
```

### PostgreSQL Checkpointer (Production)

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost/db"
)
app = graph.compile(checkpointer=checkpointer)
```

## Subgraphs

Compose graphs within graphs:

```python
# Define inner graph
inner_graph = StateGraph(InnerState)
inner_graph.add_node("process", process_node)
inner_graph.set_entry_point("process")
inner_graph.add_edge("process", END)
inner_app = inner_graph.compile()

# Use as node in outer graph
def inner_node(state: OuterState) -> OuterState:
    inner_result = inner_app.invoke({"input": state["data"]})
    return {"result": inner_result["output"]}

outer_graph = StateGraph(OuterState)
outer_graph.add_node("inner", inner_node)
```

## Parallel Execution

```python
from langgraph.graph import StateGraph

class ParallelState(TypedDict):
    input: str
    result_a: str
    result_b: str

def task_a(state: ParallelState) -> ParallelState:
    return {"result_a": f"A processed: {state['input']}"}

def task_b(state: ParallelState) -> ParallelState:
    return {"result_b": f"B processed: {state['input']}"}

def combine(state: ParallelState) -> ParallelState:
    return {"output": f"{state['result_a']} | {state['result_b']}"}

graph = StateGraph(ParallelState)
graph.add_node("task_a", task_a)
graph.add_node("task_b", task_b)
graph.add_node("combine", combine)

graph.set_entry_point("task_a")
graph.set_entry_point("task_b")  # Multiple entry points = parallel
graph.add_edge("task_a", "combine")
graph.add_edge("task_b", "combine")
graph.add_edge("combine", END)
```

## Streaming

```python
# Stream events
for event in app.stream({"messages": [("user", "Hello")]}):
    print(event)

# Stream specific output
for chunk in app.stream({"messages": [("user", "Hello")]}, stream_mode="values"):
    print(chunk["messages"][-1])

# Async streaming
async for event in app.astream({"messages": [("user", "Hello")]}):
    print(event)
```

## Human-in-the-Loop

```python
from langgraph.graph import StateGraph, END

class HumanState(TypedDict):
    messages: list
    needs_approval: bool

def check_approval(state: HumanState) -> str:
    if state.get("needs_approval"):
        return "wait_for_human"
    return "proceed"

graph = StateGraph(HumanState)
graph.add_node("process", process_node)
graph.add_node("wait_for_human", lambda s: s)  # Pauses here
graph.add_node("proceed", proceed_node)

graph.set_entry_point("process")
graph.add_conditional_edges("process", check_approval)
graph.add_edge("wait_for_human", "proceed")
graph.add_edge("proceed", END)

app = graph.compile(checkpointer=checkpointer, interrupt_before=["wait_for_human"])

# First run - pauses at wait_for_human
result = app.invoke(initial_state, config)

# After human approval - resume
app.invoke(None, config)  # Continues from checkpoint
```

## Supervisor Pattern

```python
from langgraph_supervisor import create_supervisor

# Define worker agents
researcher = create_agent(model="gpt-4o-mini", tools=[search_tool])
writer = create_agent(model="gpt-4o-mini", tools=[])

# Create supervisor
supervisor = create_supervisor(
    agents={"researcher": researcher, "writer": writer},
    model="gpt-4o-mini",
    system_prompt="Coordinate research and writing tasks."
)

# Run
result = supervisor.invoke({
    "messages": [("user", "Research and write about AI agents")]
})
```

## Memory Store

```python
from langgraph.store.memory import InMemoryStore

# Create store
store = InMemoryStore()

# Use in graph
def node_with_memory(state, *, store):
    # Read from store
    memories = store.search(namespace=("user", state["user_id"]))
    
    # Write to store
    store.put(
        namespace=("user", state["user_id"]),
        key="last_topic",
        value={"topic": state["topic"]}
    )
    
    return state

app = graph.compile(store=store)
```

## Debugging

```python
# Visualize graph
print(app.get_graph().draw_ascii())

# Or get mermaid diagram
print(app.get_graph().draw_mermaid())

# Trace execution
for event in app.stream(state, stream_mode="debug"):
    print(f"Node: {event.get('node')}")
    print(f"State: {event.get('state')}")
```

## Common Patterns

### ReAct Agent

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=ChatOpenAI(model="gpt-4o-mini"),
    tools=[search, calculate],
    state_modifier="You are a helpful assistant."
)

result = agent.invoke({"messages": [("user", "What is 15% of 240?")]})
```

### Agentic RAG

```python
class RAGState(TypedDict):
    messages: Annotated[list, add_messages]
    context: list[str]
    needs_retrieval: bool

def analyze(state: RAGState) -> RAGState:
    # Determine if retrieval needed
    needs = check_needs_retrieval(state["messages"][-1])
    return {"needs_retrieval": needs}

def retrieve(state: RAGState) -> RAGState:
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    return {"context": [d.page_content for d in docs]}

def generate(state: RAGState) -> RAGState:
    context = "\n".join(state["context"])
    response = llm.invoke(format_prompt(state["messages"], context))
    return {"messages": [response]}

def route(state: RAGState) -> str:
    return "retrieve" if state["needs_retrieval"] else "generate"

graph = StateGraph(RAGState)
graph.add_node("analyze", analyze)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.set_entry_point("analyze")
graph.add_conditional_edges("analyze", route)
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)
```

## Best Practices

1. **Keep nodes focused.** Each node should do one thing well.

2. **Use TypedDict for state.** Explicit typing catches errors early.

3. **Add checkpointing for production.** Enables recovery and human-in-the-loop.

4. **Stream for better UX.** Users see progress, not just final result.

5. **Visualize your graph.** Helps understand and debug flow.

6. **Handle errors in nodes.** Return error state rather than crashing.

## Related Concepts

- **LangChain**: Foundation for models and tools
- **LangSmith**: Tracing and debugging LangGraph apps
- **Agents**: What LangGraph is primarily used to build
