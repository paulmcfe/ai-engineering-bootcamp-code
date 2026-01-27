# Tool Use

## What Are Tools?

Tools are functions that agents can call to interact with the world beyond generating text. They bridge the gap between language model reasoning and real-world actions. An LLM can't actually search a database or send an email—but it can generate a request to call a function that does.

Tools give agents capabilities: searching, calculating, reading files, calling APIs, executing code, and more. Without tools, agents are limited to what's in their training data and their reasoning abilities. With tools, they can access current information and affect external systems.

## How Tool Calling Works

Modern LLMs have built-in support for function calling. You provide tool schemas in your request, and the model generates structured requests to call those functions.

The flow:

1. **Define tools** with names, descriptions, and parameter schemas
2. **Include tools** in your API request
3. **Model decides** whether to call a tool (and which one)
4. **Model generates** structured tool call with parameters
5. **Your code executes** the actual function
6. **Results return** to the model for continued reasoning

```python
from openai import OpenAI

client = OpenAI()

# Define tools as schemas
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }
]

# Call with tools
response = client.responses.create(
    model="gpt-4o-mini",
    input="What's the weather in Tokyo?",
    tools=tools
)

# Model may generate a tool call
if response.tool_calls:
    tool_call = response.tool_calls[0]
    print(f"Tool: {tool_call.function.name}")
    print(f"Args: {tool_call.function.arguments}")
```

## Defining Tools in LangChain

LangChain simplifies tool creation with the `@tool` decorator:

```python
from langchain.tools import tool

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for information about a topic.
    
    Use this when you need factual information from the document collection.
    Returns relevant passages with source citations.
    
    Args:
        query: The search term or question to look up
    """
    results = vector_store.similarity_search(query, k=3)
    
    if not results:
        return "No relevant information found."
    
    formatted = []
    for doc in results:
        source = doc.metadata.get("source", "Unknown")
        formatted.append(f"[{source}]: {doc.page_content}")
    
    return "\n\n".join(formatted)
```

Key elements:

- **Function name** becomes the tool name
- **Docstring** becomes the description (crucial for tool selection)
- **Type hints** define parameter types
- **Return value** goes back to the agent as observation

### Complex Parameters

```python
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Maximum results to return")
    filter_type: str = Field(default=None, description="Filter by type: 'recent', 'popular'")

@tool(args_schema=SearchParams)
def advanced_search(query: str, max_results: int = 5, filter_type: str = None) -> str:
    """Search with advanced filtering options."""
    # Implementation
    pass
```

## Tool Description Best Practices

The description is what the LLM uses to decide when to call a tool. Poor descriptions lead to wrong tool choices.

### Bad Description

```python
@tool
def search(q: str) -> str:
    """Search for stuff."""
    pass
```

Problems: vague name, unclear when to use, no parameter guidance.

### Good Description

```python
@tool
def search_product_catalog(query: str) -> str:
    """
    Search the product catalog for items matching a description.
    
    Use this tool when users ask about:
    - Product availability
    - Product specifications
    - Pricing information
    - Product comparisons
    
    Do NOT use for:
    - Order status (use check_order_status instead)
    - Customer account info (use get_customer_info instead)
    
    Args:
        query: Natural language description of products to find.
               Examples: "red running shoes", "laptop under $1000"
    
    Returns:
        List of matching products with name, price, and availability.
    """
    pass
```

## Common Tool Patterns

### Search Tool

```python
@tool
def search_documents(query: str) -> str:
    """
    Search indexed documents for information.
    
    Args:
        query: Topic or question to search for
    """
    results = vector_store.similarity_search(query, k=5)
    return format_search_results(results)
```

### API Tool

```python
@tool
def get_stock_price(symbol: str) -> str:
    """
    Get current stock price for a ticker symbol.
    
    Args:
        symbol: Stock ticker (e.g., AAPL, GOOGL)
    """
    response = requests.get(f"https://api.stocks.com/price/{symbol}")
    if response.status_code == 200:
        data = response.json()
        return f"{symbol}: ${data['price']} ({data['change']})"
    return f"Could not fetch price for {symbol}"
```

### Calculator Tool

```python
@tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: Math expression (e.g., "15 * 0.2", "sqrt(144)")
    """
    try:
        # Use safe evaluation
        import numexpr
        result = numexpr.evaluate(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"
```

### Database Tool

```python
@tool
def query_database(sql: str) -> str:
    """
    Execute a read-only SQL query against the database.
    
    Only SELECT queries are allowed. Use for retrieving customer data,
    order history, inventory counts, etc.
    
    Args:
        sql: SQL SELECT query
    """
    if not sql.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed"
    
    try:
        results = db.execute(sql).fetchall()
        return format_as_table(results)
    except Exception as e:
        return f"Query error: {e}"
```

## Error Handling

Tools should return informative errors that help the agent adapt:

```python
@tool
def search_api(query: str) -> str:
    """Search external API for information."""
    try:
        response = requests.get(f"https://api.example.com/search?q={query}", timeout=10)
        
        if response.status_code == 429:
            return "Rate limit exceeded. Try again in a few seconds with a simpler query."
        
        if response.status_code == 404:
            return f"No results found for '{query}'. Try different search terms."
        
        if response.status_code != 200:
            return f"API error (status {response.status_code}). Service may be unavailable."
        
        data = response.json()
        if not data.get("results"):
            return f"Search for '{query}' returned no results. Try broader terms."
        
        return format_results(data["results"])
        
    except requests.Timeout:
        return "Search timed out. Try a more specific query."
    except requests.ConnectionError:
        return "Cannot connect to search service. Check network connection."
    except Exception as e:
        return f"Unexpected error: {type(e).__name__}"
```

Good error messages help the agent:
- Understand what went wrong
- Decide whether to retry
- Adjust its approach

## Tool Selection

When agents have multiple tools, they must choose correctly. Factors affecting selection:

1. **Tool descriptions**: Primary source of guidance
2. **Parameter schemas**: What inputs each tool accepts
3. **User query**: What the user is asking for
4. **Conversation context**: What's happened before
5. **System prompt**: High-level guidance about tool use

### Improving Selection

**Reduce ambiguity:**
```python
# Instead of two vague search tools:
search_documents(query)
search_database(query)

# Make them specific:
search_technical_docs(query)  # For documentation and how-tos
lookup_customer_record(customer_id)  # For specific customer data
```

**Add negative guidance:**
```python
@tool
def search_recent_news(query: str) -> str:
    """
    Search news from the past 7 days.
    
    Use for current events and recent developments.
    Do NOT use for historical information or general knowledge.
    """
```

**Limit tool count:**
Having 20+ tools confuses models. Group related functionality or create specialized agents.

## Parallel Tool Calls

Modern LLMs can call multiple tools in parallel when appropriate:

```python
# User: "What's the weather in Tokyo and the current time there?"

# Model might generate two parallel calls:
tool_calls = [
    {"name": "get_weather", "args": {"city": "Tokyo"}},
    {"name": "get_time", "args": {"timezone": "Asia/Tokyo"}}
]
```

Handle parallel calls by executing all tools and returning all results:

```python
def execute_tool_calls(tool_calls):
    results = []
    for call in tool_calls:
        tool_name = call.function.name
        args = json.loads(call.function.arguments)
        
        result = available_tools[tool_name](**args)
        results.append({
            "tool_call_id": call.id,
            "result": result
        })
    
    return results
```

## Tool Validation

Validate inputs before execution to catch errors early:

```python
from pydantic import BaseModel, Field, validator

class EmailParams(BaseModel):
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")
    
    @validator("to")
    def validate_email(cls, v):
        import re
        if not re.match(r"[^@]+@[^@]+\.[^@]+", v):
            raise ValueError(f"Invalid email address: {v}")
        return v
    
    @validator("subject")
    def validate_subject(cls, v):
        if len(v) > 200:
            raise ValueError("Subject too long (max 200 characters)")
        return v

@tool(args_schema=EmailParams)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the specified recipient."""
    # Validation happens automatically via Pydantic
    email_service.send(to, subject, body)
    return f"Email sent successfully to {to}"
```

## Security Considerations

Tools can have real-world effects. Consider:

**Input sanitization:**
```python
@tool
def run_query(sql: str) -> str:
    """Execute SQL query."""
    # Prevent SQL injection
    forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", ";"]
    if any(word in sql.upper() for word in forbidden):
        return "Error: Only SELECT queries allowed"
    
    # Use parameterized queries
    return db.execute_safe(sql)
```

**Rate limiting:**
```python
from functools import lru_cache
from time import time

call_times = []

@tool
def expensive_api_call(query: str) -> str:
    """Call external API."""
    global call_times
    
    # Rate limit: max 10 calls per minute
    now = time()
    call_times = [t for t in call_times if now - t < 60]
    
    if len(call_times) >= 10:
        return "Rate limit reached. Try again later."
    
    call_times.append(now)
    return make_api_call(query)
```

**Permission checks:**
```python
@tool
def access_user_data(user_id: str) -> str:
    """Access user data."""
    # Check permissions
    if not current_user.can_access(user_id):
        return "Permission denied"
    
    return get_user_data(user_id)
```

## Testing Tools

Test tools independently before using with agents:

```python
def test_search_tool():
    # Test normal operation
    result = search_documents("embeddings")
    assert "embedding" in result.lower()
    assert len(result) < 5000  # Reasonable length
    
    # Test empty results
    result = search_documents("xyznonexistentquery123")
    assert "no results" in result.lower() or "not found" in result.lower()
    
    # Test special characters
    result = search_documents("what's the meaning of 'RAG'?")
    # Should not crash
    
    # Test very long query
    result = search_documents("a" * 10000)
    # Should handle gracefully

def test_calculator_tool():
    assert "36" in calculate("6 * 6")
    assert "Error" in calculate("1/0")
    assert "Error" in calculate("import os; os.system('rm -rf /')")
```

## Related Concepts

- **Agents**: Systems that use tools to accomplish tasks
- **ReAct Pattern**: Reasoning about which tools to use
- **Function Calling**: LLM capability enabling tool use
- **Multi-Agent Systems**: Agents that can call other agents as tools
