# Troubleshooting: Agent Debugging

## Overview

Common issues with LLM agents and how to diagnose and fix them.

---

## Agent Not Using Tools

### Symptoms
- Agent answers directly without calling tools
- Tools defined but never invoked
- "I don't have access to that information" responses

### Causes & Solutions

**1. Poor Tool Descriptions**

```python
# BAD: Vague description
@tool
def search(q: str) -> str:
    """Search."""  # Model doesn't know when to use this
    pass

# GOOD: Detailed description
@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for information about company policies,
    procedures, and documentation.
    
    Use this tool when the user asks about:
    - Company policies (HR, IT, security)
    - Internal procedures
    - Product documentation
    
    Args:
        query: Search terms related to internal documentation
    """
    pass
```

**2. System Prompt Doesn't Mention Tools**

```python
# BAD: No guidance on tool use
system_prompt = "You are a helpful assistant."

# GOOD: Explicit tool instructions
system_prompt = """You are a helpful assistant with access to tools.

IMPORTANT: Always use the search_knowledge_base tool before answering questions
about company information. Don't rely on your general knowledge for company-specific questions.

Available tools:
- search_knowledge_base: For company policies and documentation
- calculate: For mathematical calculations"""
```

**3. Tool Not Passed to Agent**

```python
# Check tools are included
agent = create_agent(
    model="gpt-4o-mini",
    tools=[search, calculate],  # Make sure tools are listed
    system_prompt=system_prompt
)
```

**4. Model Doesn't Support Function Calling**

```python
# Ensure you're using a model that supports tools
# gpt-4o-mini, gpt-4o support tools
# Some older/smaller models may not

agent = create_agent(
    model="gpt-4o-mini",  # ✓ Supports tools
    tools=tools
)
```

---

## Agent Using Wrong Tool

### Symptoms
- Agent calls calculator when it should search
- Wrong tool selected for the task
- Repeated incorrect tool selections

### Causes & Solutions

**1. Overlapping Tool Descriptions**

```python
# BAD: Overlapping descriptions
@tool
def search_web(query: str) -> str:
    """Search for information."""
    pass

@tool
def search_docs(query: str) -> str:
    """Search for information."""  # Same description!
    pass

# GOOD: Distinct descriptions
@tool
def search_web(query: str) -> str:
    """Search the public web for current events, general knowledge, 
    and information not in our internal systems.
    
    Use for: news, public information, general questions"""
    pass

@tool
def search_docs(query: str) -> str:
    """Search internal company documentation for policies, procedures,
    and proprietary information.
    
    Use for: company policies, internal procedures, product docs"""
    pass
```

**2. Missing Negative Guidance**

```python
@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations.
    
    Use for: arithmetic, algebra, unit conversions
    
    DO NOT use for:
    - Looking up numerical facts (use search instead)
    - Estimating or guessing numbers
    - Non-mathematical questions"""
    pass
```

**3. Add Routing Logic**

```python
# In system prompt
system_prompt = """Before using a tool, think about which one is appropriate:

1. Is this a factual question about the company? → search_docs
2. Is this a general knowledge question? → search_web  
3. Is this a math calculation? → calculate
4. Is this a question I can answer directly? → no tool needed

Always explain your tool choice in your reasoning."""
```

---

## Infinite Loops

### Symptoms
- Agent keeps calling tools repeatedly
- Never reaches final answer
- Hits maximum iteration limit

### Causes & Solutions

**1. No Iteration Limit**

```python
# LangGraph: Add conditional to stop
def should_continue(state):
    # Check iteration count
    if state.get("iteration_count", 0) >= 10:
        return END
    
    # Check for final answer
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return END
    
    return "tools"
```

**2. Tool Returns Useless Results**

```python
# BAD: Vague error that prompts retry
@tool
def search(query: str) -> str:
    if not results:
        return "No results"  # Agent might keep trying

# GOOD: Clear, actionable response
@tool
def search(query: str) -> str:
    if not results:
        return "No results found for this query. Try different search terms or ask a different question."
```

**3. Agent Doesn't Know When to Stop**

```python
system_prompt = """...

IMPORTANT: After getting useful information from a tool, provide your final answer.
Don't keep searching unless you genuinely need more information.

If a search returns no results, inform the user rather than searching repeatedly."""
```

**4. Circular Dependencies**

```python
# Tool A calls Tool B, Tool B calls Tool A
# Fix: Design tools to be independent

@tool
def tool_a(input: str) -> str:
    # Don't call tool_b from here
    return process_a(input)

@tool  
def tool_b(input: str) -> str:
    # Don't call tool_a from here
    return process_b(input)
```

---

## Tool Execution Errors

### Symptoms
- Tools throw exceptions
- Agent gets stuck after tool failure
- Cryptic error messages

### Causes & Solutions

**1. Unhandled Exceptions in Tools**

```python
# BAD: Exception crashes agent
@tool
def search(query: str) -> str:
    response = requests.get(url)  # Might throw!
    return response.json()

# GOOD: Handle all errors
@tool
def search(query: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.Timeout:
        return "Error: Search timed out. Please try again."
    except requests.HTTPError as e:
        return f"Error: Search failed with status {e.response.status_code}"
    except Exception as e:
        return f"Error: Search failed - {str(e)}"
```

**2. Invalid Tool Arguments**

```python
@tool
def calculate(expression: str) -> str:
    """Calculate mathematical expression."""
    
    # Validate input
    if not expression or not expression.strip():
        return "Error: Please provide a mathematical expression"
    
    # Sanitize for safety
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return "Error: Expression contains invalid characters"
    
    try:
        result = eval(expression)
        return f"Result: {result}"
    except SyntaxError:
        return "Error: Invalid mathematical expression"
    except ZeroDivisionError:
        return "Error: Division by zero"
```

**3. Type Mismatches**

```python
# Define expected types clearly
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="Search query string")
    limit: int = Field(default=5, ge=1, le=20, description="Max results")

@tool(args_schema=SearchInput)
def search(query: str, limit: int) -> str:
    # Types are validated before function is called
    pass
```

---

## Agent Reasoning Issues

### Symptoms
- Agent makes illogical decisions
- Poor reasoning in traces
- Doesn't follow instructions

### Causes & Solutions

**1. Enable Reasoning Visibility**

```python
# Check what the agent is thinking
for event in agent.stream({"messages": messages}):
    print(event)  # See reasoning traces
```

**2. Add Explicit Reasoning Steps**

```python
system_prompt = """Before taking any action:

1. UNDERSTAND: What is the user asking for?
2. PLAN: What steps do I need to take?
3. CHOOSE: Which tool (if any) should I use?
4. EXECUTE: Use the tool or provide answer
5. VERIFY: Does my response answer the question?

Think through each step before acting."""
```

**3. Use Few-Shot Examples**

```python
system_prompt = """...

Example interaction:
User: What's our vacation policy?
Thinking: This is a company policy question. I should search internal docs.
Action: search_docs("vacation policy")
Result: [vacation policy details]
Answer: Based on our documentation, the vacation policy is...

Now handle the user's request similarly."""
```

---

## Memory Issues

### Symptoms
- Agent forgets previous context
- Conversation history not maintained
- Repeats questions already answered

### Causes & Solutions

**1. Not Using Checkpointing**

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
agent = graph.compile(checkpointer=checkpointer)

# MUST provide thread_id
config = {"configurable": {"thread_id": "user-123"}}
response = agent.invoke({"messages": messages}, config)
```

**2. Thread ID Changes**

```python
# Each thread_id is a separate conversation
# Ensure consistent thread_id per user/session

# BAD: Random thread_id
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

# GOOD: Consistent thread_id
config = {"configurable": {"thread_id": f"user-{user_id}"}}
```

**3. Context Window Overflow**

```python
# Long conversations exceed context limit
# Implement message summarization

def summarize_if_needed(messages: list, max_messages: int = 20) -> list:
    if len(messages) <= max_messages:
        return messages
    
    # Keep system message and recent messages
    system = messages[0] if messages[0]["role"] == "system" else None
    recent = messages[-max_messages:]
    
    # Summarize old messages
    old_messages = messages[1:-max_messages] if system else messages[:-max_messages]
    summary = summarize(old_messages)
    
    result = []
    if system:
        result.append(system)
    result.append({"role": "system", "content": f"Previous conversation summary: {summary}"})
    result.extend(recent)
    
    return result
```

---

## Debugging Tools

### Enable Verbose Logging

```python
import langchain
langchain.debug = True

# Or use callbacks
from langchain_core.callbacks import BaseCallbackHandler

class DebugCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM Input: {prompts}")
    
    def on_llm_end(self, response, **kwargs):
        print(f"LLM Output: {response}")
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"Tool: {serialized['name']}, Input: {input_str}")
    
    def on_tool_end(self, output, **kwargs):
        print(f"Tool Output: {output}")

response = agent.invoke(messages, config={"callbacks": [DebugCallback()]})
```

### LangSmith Tracing

```python
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-key"

# All agent runs are now traced
# View at smith.langchain.com
```

### Manual Trace Inspection

```python
def inspect_agent_run(response: dict):
    """Print agent reasoning trace."""
    
    print("=== Agent Trace ===\n")
    
    for msg in response["messages"]:
        msg_type = getattr(msg, 'type', 'unknown')
        
        if msg_type == 'human':
            print(f"USER: {msg.content}\n")
        
        elif msg_type == 'ai':
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"TOOL CALL: {tc['name']}")
                    print(f"  Args: {tc['args']}\n")
            elif msg.content:
                print(f"ASSISTANT: {msg.content}\n")
        
        elif msg_type == 'tool':
            print(f"TOOL RESULT ({msg.name}):")
            print(f"  {msg.content[:200]}...\n")
```

---

## Quick Diagnostic Checklist

1. **Tools not called**
   - [ ] Tool descriptions are detailed and specific
   - [ ] System prompt mentions when to use tools
   - [ ] Tools are passed to agent correctly
   - [ ] Model supports function calling

2. **Wrong tool used**
   - [ ] Tool descriptions don't overlap
   - [ ] Include negative guidance (when NOT to use)
   - [ ] System prompt has routing logic

3. **Infinite loops**
   - [ ] Iteration limit is set
   - [ ] Tools return actionable error messages
   - [ ] Agent knows when to stop

4. **Tool errors**
   - [ ] All exceptions handled in tools
   - [ ] Input validation in tools
   - [ ] Type hints and schemas defined

5. **Memory issues**
   - [ ] Checkpointer configured
   - [ ] Consistent thread_id used
   - [ ] Context window not exceeded

---

## Related Guides

- **RAG Problems**: When agents don't retrieve correctly
- **Performance Optimization**: Speeding up agent execution
