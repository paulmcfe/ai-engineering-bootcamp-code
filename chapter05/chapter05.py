#!/usr/bin/env python
# coding: utf-8

# # Multi-Agent Applications with LangGraph
# 
# ## The Supervisor Pattern
# 
# This notebook implements a multi-agent system using the **supervisor pattern** - a central coordinator that delegates tasks to specialized worker agents. Our team consists of:
# 
# - **Supervisor**: Analyzes requests and coordinates the team
# - **Researcher**: Gathers information using web search
# - **Writer**: Creates content from research findings
# - **Editor**: Reviews and polishes the final content
# 
# ```
#                     ┌─────────────┐
#                     │  Supervisor │
#                     │    Agent    │
#                     └──────┬──────┘
#                            │
#            ┌───────────────┼───────────────┐
#            │               │               │
#            ▼               ▼               ▼
#     ┌────────────┐  ┌────────────┐  ┌────────────┐
#     │ Researcher │  │   Writer   │  │   Editor   │
#     │(web search)│  │   Agent    │  │   Agent    │
#     └────────────┘  └────────────┘  └────────────┘
# ```

# ## Setup and Imports

# In[ ]:


import os
import json
import re
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END, add_messages
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables
load_dotenv()

# Verify API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Initialize web search tool
search_tool = DuckDuckGoSearchRun()

print("Setup complete!")


# ## Define Shared State
# 
# The `TeamState` manages information flow between agents. Each agent reads from and writes to specific fields, enabling coordination without direct communication.

# In[ ]:


class TeamState(TypedDict):
    """Shared state for multi-agent coordination."""
    messages: Annotated[list, add_messages]   # Conversation history
    user_request: str                         # Original user request
    research_findings: str                    # Output from researcher
    draft_content: str                        # Output from writer
    edited_content: str                       # Output from editor
    next_action: str                          # Supervisor's routing decision
    final_response: str                       # Final output to user

print("TeamState defined!")


# ## Agent Prompts
# 
# Each agent has a specialized system prompt that defines its role, capabilities, and guidelines. Clear role definitions prevent overlap and confusion.

# In[ ]:


SUPERVISOR_PROMPT = """You coordinate a team of specialists to handle content creation requests.

Your team:
- RESEARCHER: Use for finding information, verifying facts, gathering data from the web
- WRITER: Use for creating content, drafting documents, structuring information
- EDITOR: Use for reviewing content, fixing errors, improving quality

For each request, analyze what needs to be done and decide the next action.

Current workflow state:
- User request: {user_request}
- Research findings: {research_findings}
- Draft content: {draft_content}
- Edited content: {edited_content}

Based on the current state, decide the next action:
- If no research has been done yet, delegate to RESEARCHER
- If research is complete but no draft exists, delegate to WRITER
- If a draft exists but hasn't been edited, delegate to EDITOR
- If edited content exists, mark as COMPLETE

Respond with JSON: {{"next_action": "researcher" | "writer" | "editor" | "complete", "reasoning": "brief explanation"}}
"""

RESEARCHER_PROMPT = """You are a research specialist. Your job is to gather accurate, relevant information using web search.

User request: {user_request}

Instructions:
1. Identify key topics that need research
2. Search for factual, current information
3. Synthesize findings into a clear summary
4. Include relevant facts, statistics, and insights
5. Note sources when possible

Provide comprehensive research findings that will help a writer create quality content."""

WRITER_PROMPT = """You are a content writer. Your job is to create well-structured, engaging content based on research.

User request: {user_request}

Research findings:
{research_findings}

Instructions:
1. Create content that directly addresses the user's request
2. Use the research findings to support your points
3. Structure the content logically with clear sections
4. Write in a clear, professional tone
5. Make the content informative and engaging

Create a complete draft based on the research provided."""

EDITOR_PROMPT = """You are an editor. Your job is to review and improve written content.

User request: {user_request}

Draft to edit:
{draft_content}

Instructions:
1. Review for clarity and readability
2. Fix any grammatical or spelling errors
3. Improve sentence structure and flow
4. Ensure the content fully addresses the user's request
5. Polish the writing to be professional and engaging

Provide the final, polished version of the content."""

print("Agent prompts defined!")


# ## Helper Functions
# 
# Utility functions for JSON parsing and web search.

# In[ ]:


def parse_json_response(text: str, default: dict = None) -> dict:
    """Extract and parse JSON from LLM response, handling markdown code blocks."""
    if not text or not text.strip():
        return default or {}

    # Try to find JSON in code blocks first
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if code_block_match:
        text = code_block_match.group(1).strip()

    # Try to find JSON object or array
    json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    if json_match:
        text = json_match.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default or {}


def web_search(query: str) -> str:
    """Perform web search and return results."""
    try:
        results = search_tool.run(query)
        return results
    except Exception as e:
        return f"Search failed: {str(e)}"

print("Helper functions defined!")


# ## Agent Node Functions
# 
# Each agent is implemented as a node function that reads from state, performs its task, and returns state updates.

# In[ ]:


def supervisor_node(state: TeamState) -> dict:
    """Supervisor analyzes state and decides the next action."""

    # Format the prompt with current state
    prompt = SUPERVISOR_PROMPT.format(
        user_request=state.get("user_request", ""),
        research_findings=state.get("research_findings", "None yet"),
        draft_content=state.get("draft_content", "None yet"),
        edited_content=state.get("edited_content", "None yet")
    )

    response = llm.invoke([SystemMessage(content=prompt)])

    # Parse the JSON response
    default = {"next_action": "researcher", "reasoning": "Starting with research"}
    result = parse_json_response(response.content, default)

    next_action = result.get("next_action", "researcher")
    reasoning = result.get("reasoning", "")

    print(f"[Supervisor] Decision: {next_action}")
    print(f"[Supervisor] Reasoning: {reasoning}\n")

    return {"next_action": next_action}


def researcher_node(state: TeamState) -> dict:
    """Researcher gathers information using web search."""

    user_request = state.get("user_request", "")

    print("[Researcher] Starting research...")

    # Generate search queries based on the request
    query_prompt = f"""Based on this content request, generate 2-3 search queries to gather relevant information.

Request: {user_request}

Respond with JSON: {{"queries": ["query1", "query2"]}}"""

    query_response = llm.invoke([SystemMessage(content=query_prompt)])
    queries_result = parse_json_response(query_response.content, {"queries": [user_request]})
    queries = queries_result.get("queries", [user_request])

    # Perform searches
    all_results = []
    for query in queries[:3]:  # Limit to 3 queries
        print(f"[Researcher] Searching: {query}")
        results = web_search(query)
        all_results.append(f"Search: {query}\nResults: {results}\n")

    combined_results = "\n".join(all_results)

    # Synthesize findings
    synthesis_prompt = RESEARCHER_PROMPT.format(user_request=user_request)
    synthesis_prompt += f"\n\nSearch Results:\n{combined_results}\n\nSynthesize these findings into a comprehensive summary."

    synthesis = llm.invoke([SystemMessage(content=synthesis_prompt)])

    print("[Researcher] Research complete!\n")

    return {"research_findings": synthesis.content}


def writer_node(state: TeamState) -> dict:
    """Writer creates content based on research findings."""

    print("[Writer] Creating draft...")

    prompt = WRITER_PROMPT.format(
        user_request=state.get("user_request", ""),
        research_findings=state.get("research_findings", "No research available")
    )

    response = llm.invoke([SystemMessage(content=prompt)])

    print("[Writer] Draft complete!\n")

    return {"draft_content": response.content}


def editor_node(state: TeamState) -> dict:
    """Editor reviews and polishes the content."""

    print("[Editor] Editing content...")

    prompt = EDITOR_PROMPT.format(
        user_request=state.get("user_request", ""),
        draft_content=state.get("draft_content", "No draft available")
    )

    response = llm.invoke([SystemMessage(content=prompt)])

    print("[Editor] Editing complete!\n")

    return {"edited_content": response.content}


def complete_node(state: TeamState) -> dict:
    """Finalize the response."""

    print("[Complete] Finalizing response...")

    final_content = state.get("edited_content", state.get("draft_content", ""))

    return {"final_response": final_content}

print("Agent node functions defined!")


# ## Routing Logic
# 
# The router function directs the workflow based on the supervisor's decisions.

# In[ ]:


def supervisor_router(state: TeamState) -> Literal["researcher", "writer", "editor", "complete"]:
    """Route to the appropriate agent based on supervisor's decision."""
    next_action = state.get("next_action", "researcher")

    if next_action in ["researcher", "writer", "editor", "complete"]:
        return next_action

    # Default to researcher if unknown action
    return "researcher"

print("Routing logic defined!")


# ## Build the StateGraph
# 
# Now we assemble all the pieces into a LangGraph workflow.

# In[ ]:


# Build the multi-agent graph
graph = StateGraph(TeamState)

# Add all agent nodes
graph.add_node("supervisor", supervisor_node)
graph.add_node("researcher", researcher_node)
graph.add_node("writer", writer_node)
graph.add_node("editor", editor_node)
graph.add_node("complete", complete_node)

# Set the entry point
graph.set_entry_point("supervisor")

# Add conditional edges from supervisor to workers
graph.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        "researcher": "researcher",
        "writer": "writer",
        "editor": "editor",
        "complete": "complete"
    }
)

# Workers return to supervisor for next decision
graph.add_edge("researcher", "supervisor")
graph.add_edge("writer", "supervisor")
graph.add_edge("editor", "supervisor")

# Complete leads to END
graph.add_edge("complete", END)

# Compile the graph
multi_agent_app = graph.compile()

print("Multi-agent graph built and compiled!")


# In[ ]:


# Visualize the graph structure
print("Graph Structure (ASCII):")
print(multi_agent_app.get_graph().draw_ascii())


# ## Run the Multi-Agent System
# 
# Let's test the system with a content creation request. The supervisor will coordinate the team through research, writing, and editing phases.

# 

# In[ ]:


# Test the multi-agent system
test_request = "Write a brief article about the benefits of multi-agent AI systems"

print("=" * 60)
print("MULTI-AGENT CONTENT CREATION")
print("=" * 60)
print(f"\nUser Request: {test_request}\n")
print("=" * 60)

# Run the multi-agent workflow
result = multi_agent_app.invoke({
    "user_request": test_request,
    "messages": []
})

print("=" * 60)
print("FINAL OUTPUT")
print("=" * 60)
print(result.get("final_response", result.get("edited_content", "No output")))


# ## Examining the Workflow State
# 
# Let's look at what each agent produced during the workflow.

# In[ ]:


# Examine what each agent produced
print("=" * 60)
print("RESEARCH FINDINGS (from Researcher)")
print("=" * 60)
print(result.get("research_findings", "None")[:1000] + "..." if len(result.get("research_findings", "")) > 1000 else result.get("research_findings", "None"))

print("\n" + "=" * 60)
print("DRAFT CONTENT (from Writer)")
print("=" * 60)
print(result.get("draft_content", "None")[:1000] + "..." if len(result.get("draft_content", "")) > 1000 else result.get("draft_content", "None"))

print("\n" + "=" * 60)
print("EDITED CONTENT (from Editor)")
print("=" * 60)
print(result.get("edited_content", "None")[:1000] + "..." if len(result.get("edited_content", "")) > 1000 else result.get("edited_content", "None"))


# ## Try Your Own Request
# 
# Try running the multi-agent system with your own content request!

# In[ ]:


# Try your own request
your_request = "Write a short summary about the latest trends in renewable energy"

print("=" * 60)
print("CUSTOM REQUEST")
print("=" * 60)
print(f"\nUser Request: {your_request}\n")
print("=" * 60)

custom_result = multi_agent_app.invoke({
    "user_request": your_request,
    "messages": []
})

print("=" * 60)
print("FINAL OUTPUT")
print("=" * 60)
print(custom_result.get("final_response", custom_result.get("edited_content", "No output")))

