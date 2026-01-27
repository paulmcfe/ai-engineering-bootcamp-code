# Plan-and-Execute Pattern

## Overview

The plan-and-execute pattern separates planning from execution. The agent first creates a complete plan for accomplishing a task, then executes each step in sequence. This contrasts with ReAct, which decides the next step based on the current state.

## When to Use

**Good fit:**
- Tasks with clear, predictable steps
- Complex tasks benefiting from upfront planning
- Situations where you want to show users the plan before executing
- Tasks where steps don't depend heavily on intermediate results

**Poor fit:**
- Highly dynamic tasks where the path depends on discoveries
- Simple tasks that don't need planning overhead
- Real-time interactions requiring immediate responses

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      PLANNER                                 │
│  Create step-by-step plan to accomplish the task            │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                       PLAN                                   │
│  Step 1: [action]                                           │
│  Step 2: [action]                                           │
│  Step 3: [action]                                           │
│  ...                                                        │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     EXECUTOR                                 │
│  Execute each step, collect results                         │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  SYNTHESIZER                                 │
│  Combine step results into final response                   │
└─────────────────────────────────────────────────────────────┘
```

## Basic Implementation

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

class Step(BaseModel):
    step_number: int
    action: str
    description: str
    expected_output: str

class Plan(BaseModel):
    goal: str
    steps: list[Step]
    estimated_time: str

def create_plan(query: str) -> Plan:
    """Create a plan for accomplishing the task."""
    
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="""You are a planning assistant.
Create detailed, actionable plans for tasks.

For each step, specify:
- What action to take
- Why this step is needed
- What output to expect

Return as JSON matching the Plan schema.""",
        input=f"Create a plan to: {query}",
        response_format={"type": "json_object"}
    )
    
    return Plan.model_validate_json(response.output_text)

def execute_step(step: Step, context: dict) -> str:
    """Execute a single step of the plan."""
    
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="Execute the given step using available context.",
        input=f"""Step: {step.action}
Description: {step.description}
Previous results: {context}

Execute this step and provide the result."""
    )
    
    return response.output_text

def plan_and_execute(query: str) -> str:
    """Full plan-and-execute workflow."""
    
    # Planning phase
    plan = create_plan(query)
    print(f"Plan created with {len(plan.steps)} steps")
    
    # Execution phase
    context = {"goal": plan.goal, "results": {}}
    
    for step in plan.steps:
        print(f"Executing step {step.step_number}: {step.action}")
        result = execute_step(step, context)
        context["results"][f"step_{step.step_number}"] = result
    
    # Synthesis phase
    final = synthesize_results(query, context["results"])
    
    return final

def synthesize_results(query: str, results: dict) -> str:
    """Combine step results into final response."""
    
    response = client.responses.create(
        model="gpt-4o-mini",
        input=f"""Original task: {query}

Step results:
{format_results(results)}

Synthesize these results into a complete, coherent response to the original task."""
    )
    
    return response.output_text
```

## With LangGraph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class PlanExecuteState(TypedDict):
    query: str
    plan: list[dict]
    current_step: int
    step_results: dict
    final_response: str

def planner_node(state: PlanExecuteState) -> PlanExecuteState:
    """Create the execution plan."""
    plan = create_plan(state["query"])
    return {
        "plan": [s.model_dump() for s in plan.steps],
        "current_step": 0,
        "step_results": {}
    }

def executor_node(state: PlanExecuteState) -> PlanExecuteState:
    """Execute the current step."""
    step_idx = state["current_step"]
    step = state["plan"][step_idx]
    
    result = execute_step(Step(**step), state["step_results"])
    
    new_results = {**state["step_results"], f"step_{step_idx}": result}
    
    return {
        "step_results": new_results,
        "current_step": step_idx + 1
    }

def synthesizer_node(state: PlanExecuteState) -> PlanExecuteState:
    """Synthesize final response."""
    response = synthesize_results(state["query"], state["step_results"])
    return {"final_response": response}

def should_continue(state: PlanExecuteState) -> str:
    """Check if more steps remain."""
    if state["current_step"] >= len(state["plan"]):
        return "synthesize"
    return "execute"

# Build graph
graph = StateGraph(PlanExecuteState)

graph.add_node("plan", planner_node)
graph.add_node("execute", executor_node)
graph.add_node("synthesize", synthesizer_node)

graph.set_entry_point("plan")
graph.add_edge("plan", "execute")
graph.add_conditional_edges("execute", should_continue, {
    "execute": "execute",
    "synthesize": "synthesize"
})
graph.add_edge("synthesize", END)

plan_execute_agent = graph.compile()
```

## Planning Strategies

### Hierarchical Planning

Break complex plans into sub-plans:

```python
def hierarchical_plan(query: str) -> dict:
    """Create hierarchical plan with sub-plans."""
    
    # High-level plan
    high_level = create_plan(f"Create high-level plan for: {query}")
    
    # Create sub-plans for complex steps
    detailed_plan = {"goal": query, "phases": []}
    
    for step in high_level.steps:
        if is_complex(step):
            sub_plan = create_plan(f"Detail how to: {step.action}")
            detailed_plan["phases"].append({
                "main_step": step,
                "sub_steps": sub_plan.steps
            })
        else:
            detailed_plan["phases"].append({
                "main_step": step,
                "sub_steps": []
            })
    
    return detailed_plan
```

### Conditional Planning

Plan with branching based on conditions:

```python
class ConditionalStep(BaseModel):
    step_number: int
    action: str
    condition: str | None = None
    if_true: int | None = None  # Next step if condition true
    if_false: int | None = None  # Next step if condition false

def execute_conditional_plan(plan: list[ConditionalStep]) -> dict:
    """Execute plan with conditional branching."""
    
    current = 0
    results = {}
    
    while current < len(plan):
        step = plan[current]
        result = execute_step(step, results)
        results[f"step_{step.step_number}"] = result
        
        if step.condition:
            condition_met = evaluate_condition(step.condition, result)
            current = step.if_true if condition_met else step.if_false
        else:
            current += 1
    
    return results
```

### Replanning

Adjust plan based on execution results:

```python
def plan_with_replanning(query: str, max_replans: int = 2) -> str:
    """Execute with ability to replan on failure."""
    
    plan = create_plan(query)
    replan_count = 0
    
    for i, step in enumerate(plan.steps):
        result = execute_step(step, {})
        
        if step_failed(result) and replan_count < max_replans:
            # Create new plan from current point
            remaining_goal = f"""
Original goal: {query}
Completed steps: {i}
Failed step: {step.action}
Failure reason: {result}

Create a new plan to complete the remaining work."""
            
            plan = create_plan(remaining_goal)
            replan_count += 1
            continue
    
    return synthesize_results(query, all_results)
```

## Plan Validation

Validate plans before execution:

```python
def validate_plan(plan: Plan, available_tools: list[str]) -> dict:
    """Validate plan feasibility."""
    
    issues = []
    
    for step in plan.steps:
        # Check if required tools are available
        required_tools = extract_required_tools(step)
        missing = set(required_tools) - set(available_tools)
        if missing:
            issues.append(f"Step {step.step_number}: Missing tools {missing}")
        
        # Check for circular dependencies
        if has_circular_dependency(step, plan.steps):
            issues.append(f"Step {step.step_number}: Circular dependency")
        
        # Check step clarity
        if is_too_vague(step):
            issues.append(f"Step {step.step_number}: Too vague to execute")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues
    }
```

## Comparison with ReAct

| Aspect | Plan-and-Execute | ReAct |
|--------|------------------|-------|
| Planning | Upfront, complete | Step-by-step |
| Adaptability | Lower (fixed plan) | Higher (dynamic) |
| Predictability | Higher | Lower |
| Transparency | Can show plan before executing | Plan emerges during execution |
| Latency | Higher initial (planning), then fast | Consistent per step |
| Best for | Structured tasks | Exploratory tasks |

## Hybrid Approach

Combine planning with reactive execution:

```python
def hybrid_plan_execute(query: str) -> str:
    """Plan upfront but allow reactive adjustments."""
    
    # Create initial plan
    plan = create_plan(query)
    
    results = {}
    for i, step in enumerate(plan.steps):
        # Execute step
        result = execute_step(step, results)
        results[f"step_{i}"] = result
        
        # Reactive check: should we adjust?
        assessment = assess_progress(query, plan, results, i)
        
        if assessment["needs_adjustment"]:
            # Switch to ReAct mode for remaining steps
            remaining = react_complete(
                query, 
                context=results,
                completed_steps=i
            )
            results.update(remaining)
            break
    
    return synthesize_results(query, results)
```

## Best Practices

1. **Validate plans before executing.** Catch issues early.

2. **Keep steps atomic.** Each step should be independently executable.

3. **Include checkpoints.** Allow for progress verification between steps.

4. **Plan for failure.** Include error handling in plans.

5. **Show plans to users.** Transparency builds trust.

6. **Allow replanning.** Real execution rarely matches perfect plans.

## Common Pitfalls

### Over-Planning

Creating unnecessarily detailed plans for simple tasks. Solution: assess task complexity first.

### Rigid Plans

Plans that can't adapt to unexpected results. Solution: build in decision points and replanning.

### Missing Dependencies

Steps that require outputs from other steps not yet executed. Solution: validate dependencies during planning.

### Vague Steps

Steps too ambiguous to execute reliably. Solution: require specific, actionable step descriptions.

## Related Patterns

- **ReAct**: Reactive alternative to plan-and-execute
- **Hierarchical Planning**: Breaking plans into sub-plans
- **Goal-Oriented Action Planning (GOAP)**: Game AI planning technique
- **HTN Planning**: Hierarchical task network planning
