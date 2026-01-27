# Supervisor Pattern

## Overview

The supervisor pattern uses a central coordinating agent to manage multiple specialized worker agents. The supervisor receives user requests, decides which worker(s) to engage, delegates tasks, collects results, and synthesizes final responses.

This is the most common multi-agent architecture because it's intuitive (mirrors human teams) and provides clear control flow.

## When to Use

**Good fit:**
- Tasks decompose into distinct subtasks
- Different subtasks require different expertise
- You need coordinated multi-step workflows
- Clear delegation logic exists

**Poor fit:**
- Simple tasks a single agent handles well
- Real-time requirements (supervision adds latency)
- Highly dynamic task routing (consider swarm pattern)

## Architecture

```
                         ┌─────────────────────┐
        User Query ─────►│    Supervisor       │◄───── Final Response
                         │   (Coordinator)     │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             ┌───────────┐   ┌───────────┐   ┌───────────┐
             │ Worker A  │   │ Worker B  │   │ Worker C  │
             │(Research) │   │ (Writer)  │   │ (Editor)  │
             └───────────┘   └───────────┘   └───────────┘
```

## Core Components

### Supervisor Agent

The coordinator that manages the workflow:

```python
SUPERVISOR_PROMPT = """You are a team supervisor coordinating specialized workers.

Your team:
- Researcher: Finds and gathers information
- Writer: Creates content from research
- Editor: Reviews and improves content

For each user request:
1. Analyze what's needed
2. Decide which worker(s) to engage and in what order
3. Delegate tasks with clear instructions
4. Review results and decide next steps
5. Synthesize final response

Available actions:
- delegate_to_researcher(task): Send research task
- delegate_to_writer(task, context): Send writing task with context
- delegate_to_editor(content): Send content for editing
- respond_to_user(response): Return final answer

Always think step by step about the best delegation strategy."""
```

### Worker Agents

Specialized agents focused on specific capabilities:

```python
# Researcher
researcher = create_agent(
    model="gpt-4o-mini",
    tools=[search_web, search_documents, search_papers],
    system_prompt="""You are a research specialist.
    
    Your job: Find accurate, relevant information.
    
    Guidelines:
    - Search multiple sources when appropriate
    - Verify facts across sources
    - Summarize findings clearly
    - Cite your sources
    - Flag when information is uncertain or conflicting"""
)

# Writer  
writer = create_agent(
    model="gpt-4o-mini",
    tools=[],  # Writers typically don't need tools
    system_prompt="""You are a content writer.
    
    Your job: Create clear, engaging content from provided research.
    
    Guidelines:
    - Write for the specified audience
    - Structure content logically
    - Use clear, concise language
    - Incorporate all relevant research
    - Match the requested tone and format"""
)

# Editor
editor = create_agent(
    model="gpt-4o-mini",
    tools=[grammar_check, readability_score],
    system_prompt="""You are an editor.
    
    Your job: Improve content quality.
    
    Guidelines:
    - Fix grammar and spelling
    - Improve clarity and flow
    - Ensure consistency
    - Check facts if possible
    - Suggest structural improvements"""
)
```

## Implementation with LangGraph

```python
from langgraph.graph import StateGraph, END
from langgraph_supervisor import create_supervisor
from typing import TypedDict, Literal

class TeamState(TypedDict):
    messages: list
    next_worker: str
    task_results: dict
    final_response: str

# Create supervisor using langgraph-supervisor
supervisor = create_supervisor(
    agents={
        "researcher": researcher,
        "writer": writer,
        "editor": editor
    },
    model="gpt-4o-mini",
    system_prompt=SUPERVISOR_PROMPT
)

# Or build manually for more control
def supervisor_node(state: TeamState) -> TeamState:
    """Supervisor decides next action."""
    
    decision = supervisor_llm.invoke(f"""
Based on the conversation and results so far:
{format_messages(state["messages"])}

Task results: {state["task_results"]}

What should happen next? Choose:
- delegate:researcher - for research tasks
- delegate:writer - for content creation
- delegate:editor - for content review
- respond - to give final answer
""")
    
    return {"next_worker": parse_decision(decision)}

def researcher_node(state: TeamState) -> TeamState:
    """Researcher does research."""
    task = get_current_task(state)
    result = researcher.invoke({"input": task})
    return {"task_results": {**state["task_results"], "research": result}}

def writer_node(state: TeamState) -> TeamState:
    """Writer creates content."""
    research = state["task_results"].get("research", "")
    task = get_current_task(state)
    result = writer.invoke({"input": task, "context": research})
    return {"task_results": {**state["task_results"], "draft": result}}

def editor_node(state: TeamState) -> TeamState:
    """Editor reviews content."""
    draft = state["task_results"].get("draft", "")
    result = editor.invoke({"input": f"Edit this: {draft}"})
    return {"task_results": {**state["task_results"], "edited": result}}

def respond_node(state: TeamState) -> TeamState:
    """Generate final response."""
    final = synthesize_results(state["task_results"])
    return {"final_response": final}

# Route based on supervisor decision
def route_decision(state: TeamState) -> Literal["researcher", "writer", "editor", "respond"]:
    decision = state["next_worker"]
    if "researcher" in decision:
        return "researcher"
    elif "writer" in decision:
        return "writer"
    elif "editor" in decision:
        return "editor"
    return "respond"

# Build graph
graph = StateGraph(TeamState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)
graph.add_node("editor", editor_node)
graph.add_node("respond", respond_node)

graph.set_entry_point("supervisor")
graph.add_conditional_edges("supervisor", route_decision)
graph.add_edge("researcher", "supervisor")
graph.add_edge("writer", "supervisor")
graph.add_edge("editor", "supervisor")
graph.add_edge("respond", END)

team = graph.compile()
```

## Delegation Patterns

### Sequential Delegation

Workers engaged one after another:

```
User: "Write a blog post about AI agents"

Supervisor → Researcher: "Find information about AI agents"
Researcher → Supervisor: [research results]
Supervisor → Writer: "Write blog post using this research"
Writer → Supervisor: [draft]
Supervisor → Editor: "Review and improve this draft"
Editor → Supervisor: [edited content]
Supervisor → User: [final post]
```

### Parallel Delegation

Multiple workers engaged simultaneously:

```python
async def parallel_delegation(tasks: dict):
    """Delegate to multiple workers in parallel."""
    
    results = await asyncio.gather(
        researcher.invoke(tasks.get("research")),
        writer.invoke(tasks.get("outline")),
        # Add more parallel tasks
    )
    
    return combine_results(results)
```

### Conditional Delegation

Delegation based on analysis:

```python
def conditional_delegation(query: str, analysis: dict):
    """Choose delegation strategy based on query analysis."""
    
    if analysis["needs_research"]:
        yield delegate_to("researcher", query)
    
    if analysis["type"] == "content_creation":
        yield delegate_to("writer", query)
        yield delegate_to("editor", get_draft())
    
    elif analysis["type"] == "analysis":
        yield delegate_to("analyst", query)
```

## Supervisor Prompting

### Clear Role Definition

```python
prompt = """You manage a team of specialists:

RESEARCHER (use for):
- Finding facts and information
- Verifying claims
- Gathering data from sources

WRITER (use for):
- Creating new content
- Drafting documents
- Formatting information

EDITOR (use for):
- Reviewing content
- Fixing errors
- Improving quality

For this request, decide WHO should work on WHAT and in WHAT ORDER."""
```

### Structured Output

```python
from pydantic import BaseModel

class Delegation(BaseModel):
    worker: str
    task: str
    priority: int
    depends_on: list[str] = []

class SupervisorDecision(BaseModel):
    analysis: str
    delegations: list[Delegation]
    notes: str
```

### Few-Shot Examples

```python
examples = """
Example 1:
User: "Research and summarize recent AI developments"
Decision: 
1. Researcher - find recent AI news and developments
2. Writer - create summary from research

Example 2:
User: "Fix the grammar in my essay"
Decision:
1. Editor - review and fix grammar (no research or writing needed)

Example 3:
User: "Create a detailed report on climate change"
Decision:
1. Researcher - gather data on climate change
2. Writer - create report structure and content
3. Editor - review and polish final report
"""
```

## Error Handling

### Worker Failure

```python
def handle_worker_failure(worker: str, error: Exception, state: TeamState):
    """Handle when a worker fails."""
    
    # Log the error
    logger.error(f"Worker {worker} failed: {error}")
    
    # Options:
    # 1. Retry with same worker
    if should_retry(error):
        return retry_worker(worker, state)
    
    # 2. Try different worker
    if has_backup(worker):
        return delegate_to_backup(worker, state)
    
    # 3. Continue without this result
    if is_optional(worker):
        return continue_workflow(state)
    
    # 4. Fail gracefully
    return fail_gracefully(worker, error)
```

### Stuck Detection

```python
def detect_stuck(state: TeamState) -> bool:
    """Detect if the team is stuck in a loop."""
    
    recent_workers = state.get("worker_history", [])[-5:]
    
    # Same worker called 3+ times in a row
    if len(set(recent_workers[-3:])) == 1:
        return True
    
    # Cycling between same workers
    if len(recent_workers) == 5 and len(set(recent_workers)) <= 2:
        return True
    
    return False

def handle_stuck(state: TeamState) -> TeamState:
    """Break out of stuck state."""
    return {"next_worker": "respond", "stuck_reason": "Loop detected"}
```

## Monitoring and Observability

### Delegation Logging

```python
def log_delegation(supervisor_id: str, worker: str, task: str):
    """Log each delegation for analysis."""
    logger.info(
        "Delegation",
        extra={
            "supervisor": supervisor_id,
            "worker": worker,
            "task_summary": task[:100],
            "timestamp": datetime.now().isoformat()
        }
    )
```

### Metrics

```python
metrics = {
    "delegations_per_request": [],
    "worker_utilization": {"researcher": 0, "writer": 0, "editor": 0},
    "average_completion_time": [],
    "failure_rate_by_worker": {}
}
```

## Best Practices

1. **Keep worker roles focused.** Each worker should have a clear, distinct purpose.

2. **Limit delegation depth.** Deep chains of delegations add latency and complexity.

3. **Provide context with delegations.** Workers need enough context to do their jobs.

4. **Set iteration limits.** Prevent infinite delegation loops.

5. **Log all decisions.** Understand why the supervisor made each choice.

6. **Test individual workers.** Unit test workers before integration.

7. **Monitor supervisor decisions.** Track patterns in delegation to identify issues.

## Common Variations

### Hierarchical Supervisor

Supervisors managing other supervisors:

```
Top Supervisor
    ├── Research Manager
    │       ├── Web Researcher
    │       └── Academic Researcher
    └── Content Manager
            ├── Writer
            └── Editor
```

### Supervisor with Memory

Supervisor remembers past interactions to improve delegation:

```python
supervisor_with_memory = create_agent(
    model="gpt-4o-mini",
    tools=[manage_memory, search_memory, *delegation_tools],
    system_prompt="""You are a supervisor with memory.
    
    Use memory to:
    - Remember what worked well before
    - Recall user preferences
    - Learn from past mistakes"""
)
```

## Related Patterns

- **Swarm Pattern**: Peer-to-peer coordination without central supervisor
- **Pipeline Pattern**: Fixed sequence instead of dynamic delegation
- **Hierarchical Pattern**: Multiple levels of supervision
