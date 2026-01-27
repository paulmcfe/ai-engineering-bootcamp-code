# Multi-Agent Patterns

## Why Multiple Agents?

A single agent can handle many tasks, but complex systems often benefit from multiple specialized agents. Each agent focuses on a specific domain or capability, making the system more manageable, testable, and effective.

Think of it like an organization. You could have one person do everything, but specialized roles—researcher, writer, editor, fact-checker—produce better results. Each person develops deep expertise in their area.

## When NOT to Use Multi-Agent

Don't add agents just because you can. Multi-agent systems add complexity:

- More components to build and test
- Communication overhead between agents
- Harder to debug and trace issues
- Potential for coordination failures

**Use multi-agent when:**
- Task naturally decomposes into distinct subtasks
- Different subtasks require different expertise or tools
- Parallel execution would improve performance
- Context management becomes challenging with one agent

**Stick with single agent when:**
- Task is straightforward
- One agent can handle all required tools
- Latency is critical
- You're still learning the domain

## Context Optimization

One key reason for multiple agents: context management.

Single agents accumulate context over long interactions. The conversation history, tool results, retrieved documents—all compete for limited context window space. Eventually, the agent loses track of what matters. This is called "context rot."

Multiple agents can manage context more efficiently:
- Each agent maintains focused context for its domain
- Handoffs pass only relevant information
- Specialists don't need to know everything

```
Single Agent Context (everything mixed):
[user query] [search 1] [search 2] [calculation] [more search]
[user follow-up] [search 3] [analysis] [search 4] ...

Multi-Agent Context (separated):
Coordinator: [user query] [delegations] [summaries from specialists]
Researcher: [search queries] [results] [synthesis]
Analyzer: [data] [calculations] [conclusions]
```

## Common Architectures

### Supervisor Pattern

A central coordinator manages worker agents, delegating tasks and collecting results.

```
                    ┌─────────────────┐
                    │   Supervisor    │
                    │   (Coordinator) │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │  Researcher │   │   Writer    │   │   Editor    │
    └─────────────┘   └─────────────┘   └─────────────┘
```

The supervisor:
- Receives user requests
- Decides which worker(s) to engage
- Delegates tasks
- Collects and synthesizes results
- Returns final response

```python
from langgraph_supervisor import create_supervisor

# Define worker agents
researcher = create_agent(
    model="gpt-4o-mini",
    tools=[search_web, search_papers],
    system_prompt="You are a research specialist. Find and summarize information."
)

writer = create_agent(
    model="gpt-4o-mini",
    tools=[],
    system_prompt="You are a writer. Create clear, engaging content from research."
)

editor = create_agent(
    model="gpt-4o-mini",
    tools=[grammar_check],
    system_prompt="You are an editor. Improve clarity and fix errors."
)

# Create supervisor
supervisor = create_supervisor(
    agents=[researcher, writer, editor],
    model="gpt-4o-mini",
    system_prompt="""You coordinate a team:
    - Researcher: finds information
    - Writer: creates content
    - Editor: polishes output
    
    Delegate tasks appropriately and synthesize results."""
)
```

### Swarm Pattern

Agents coordinate peer-to-peer without central control. Each agent can hand off to others based on the conversation.

```
    ┌─────────────┐     ┌─────────────┐
    │   Agent A   │◄───►│   Agent B   │
    └──────┬──────┘     └──────┬──────┘
           │                   │
           └─────────┬─────────┘
                     ▼
              ┌─────────────┐
              │   Agent C   │
              └─────────────┘
```

Each agent decides:
- Can I handle this?
- If not, who should?

Good for: customer support flows, triage systems, domain routing.

### Hierarchical Pattern

Multiple levels of supervision. High-level coordinators manage mid-level managers who manage worker agents.

```
                    ┌─────────────────┐
                    │  Top Supervisor │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       ┌─────────────┐               ┌─────────────┐
       │  Research   │               │  Writing    │
       │   Manager   │               │   Manager   │
       └──────┬──────┘               └──────┬──────┘
              │                             │
       ┌──────┴──────┐               ┌──────┴──────┐
       ▼             ▼               ▼             ▼
   [Searcher]   [Analyst]       [Drafter]    [Editor]
```

Good for: complex workflows, large teams, when subtasks have their own complexity.

### Pipeline Pattern

Agents process sequentially. Output of one becomes input to next.

```
Input → [Agent A] → [Agent B] → [Agent C] → Output
```

Good for: content processing, analysis pipelines, transformation chains.

```python
def run_pipeline(input_text: str):
    # Stage 1: Research
    research = researcher.invoke({"input": input_text})
    
    # Stage 2: Draft
    draft = writer.invoke({"input": research["output"]})
    
    # Stage 3: Edit
    final = editor.invoke({"input": draft["output"]})
    
    return final["output"]
```

## Agent Handoffs

How agents pass work to each other:

### Explicit Handoff Tool

```python
@tool
def transfer_to_researcher(task: str) -> str:
    """
    Hand off a research task to the research specialist.
    Use when information gathering is needed.
    """
    result = researcher.invoke({"task": task})
    return result["output"]
```

### Structured Handoff

```python
from pydantic import BaseModel

class Handoff(BaseModel):
    target_agent: str
    task: str
    context: dict
    priority: str = "normal"

def process_handoff(handoff: Handoff):
    agent = agents[handoff.target_agent]
    return agent.invoke({
        "task": handoff.task,
        "context": handoff.context
    })
```

### Event-Based Handoff

```python
# Publish task completion
def on_task_complete(agent_name: str, result: dict):
    event_bus.publish({
        "event": "task_complete",
        "agent": agent_name,
        "result": result
    })

# Subscribe to relevant events
@event_bus.subscribe("task_complete")
def handle_completion(event: dict):
    if should_trigger_next_step(event):
        next_agent.invoke(event["result"])
```

## Communication Patterns

### Shared State

All agents read/write from common state:

```python
from typing import TypedDict

class SharedState(TypedDict):
    user_query: str
    research_findings: list[str]
    draft_content: str
    edit_suggestions: list[str]
    final_output: str

# Each agent updates relevant parts
def researcher_node(state: SharedState) -> SharedState:
    findings = do_research(state["user_query"])
    return {"research_findings": findings}

def writer_node(state: SharedState) -> SharedState:
    draft = write_from_research(state["research_findings"])
    return {"draft_content": draft}
```

### Message Passing

Agents communicate through messages:

```python
class Message(BaseModel):
    sender: str
    recipient: str
    content: str
    message_type: str  # "request", "response", "notification"

message_queue = []

def send_message(msg: Message):
    message_queue.append(msg)

def receive_messages(agent_name: str) -> list[Message]:
    return [m for m in message_queue if m.recipient == agent_name]
```

### Blackboard Pattern

Central knowledge store that agents contribute to and read from:

```python
blackboard = {
    "problem": None,
    "hypotheses": [],
    "evidence": [],
    "conclusions": []
}

def researcher_contribute(blackboard):
    evidence = gather_evidence(blackboard["problem"])
    blackboard["evidence"].extend(evidence)

def analyst_contribute(blackboard):
    conclusion = analyze(blackboard["evidence"])
    blackboard["conclusions"].append(conclusion)
```

## Implementing with LangGraph

LangGraph is ideal for multi-agent systems:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

class TeamState(TypedDict):
    messages: list
    current_agent: str
    task_status: str

def supervisor_node(state: TeamState) -> TeamState:
    """Decide which agent should work next."""
    decision = supervisor.invoke(state["messages"])
    return {"current_agent": decision["next_agent"]}

def researcher_node(state: TeamState) -> TeamState:
    """Research agent does its work."""
    result = researcher.invoke(state["messages"])
    return {"messages": state["messages"] + [result]}

def writer_node(state: TeamState) -> TeamState:
    """Writer agent does its work."""
    result = writer.invoke(state["messages"])
    return {"messages": state["messages"] + [result]}

def route_to_agent(state: TeamState) -> Literal["researcher", "writer", "end"]:
    return state["current_agent"]

# Build graph
graph = StateGraph(TeamState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", route_to_agent, {
    "researcher": "researcher",
    "writer": "writer",
    "end": END
})
graph.add_edge("researcher", "supervisor")
graph.add_edge("writer", "supervisor")

team = graph.compile()
```

## Debugging Multi-Agent Systems

Multi-agent systems are harder to debug. Strategies:

### Comprehensive Logging

```python
def log_agent_action(agent_name: str, action: str, details: dict):
    logger.info(
        f"[{agent_name}] {action}",
        extra={
            "agent": agent_name,
            "action": action,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
    )
```

### LangSmith Tracing

LangSmith shows the full execution graph:
- Which agents were called
- What each agent received and produced
- Where handoffs occurred
- Time spent in each agent

### Replay Testing

Record interactions and replay for debugging:

```python
def record_interaction(input_data, agent_calls, output):
    recording = {
        "input": input_data,
        "agent_calls": agent_calls,
        "output": output,
        "timestamp": datetime.now().isoformat()
    }
    save_recording(recording)

def replay_recording(recording_id: str):
    recording = load_recording(recording_id)
    # Step through agent calls for debugging
    for call in recording["agent_calls"]:
        print(f"Agent: {call['agent']}")
        print(f"Input: {call['input']}")
        print(f"Output: {call['output']}")
```

## Common Pitfalls

### Over-Engineering

Building complex multi-agent systems for simple problems. Start simple, add agents only when needed.

### Poor Role Definition

Ambiguous agent responsibilities lead to confusion. Define clear, non-overlapping roles.

### Inefficient Communication

Too much back-and-forth between agents wastes tokens and time. Design efficient handoffs.

### Missing Error Handling

What happens when one agent fails? Plan for failures at each point.

### Context Leakage

Sensitive information passing between agents inappropriately. Control what gets shared.

## Best Practices

1. **Start with one agent.** Add more only when you hit limitations.

2. **Define clear roles.** Each agent should have distinct responsibilities.

3. **Minimize handoffs.** Each handoff adds latency and potential failure points.

4. **Use appropriate architecture.** Supervisor for coordination, pipeline for sequential tasks, swarm for routing.

5. **Log extensively.** You'll need visibility into agent interactions.

6. **Test agent combinations.** Unit test individual agents, integration test the team.

7. **Monitor in production.** Track handoff success rates, agent utilization, end-to-end latency.

## Related Concepts

- **Agents**: Individual units that compose multi-agent systems
- **LangGraph**: Framework for building multi-agent architectures
- **Tool Use**: How agents interact (agents can be tools for other agents)
- **Supervisor Pattern**: Common multi-agent coordination approach
