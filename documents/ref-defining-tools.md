# Recipe: Defining Tools

## Goal

Create tools that agents can use to interact with the world.

## Quick Start

```python
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """Search for information about a topic."""
    # Your implementation
    return f"Results for: {query}"

# Use in agent
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search],
    system_prompt="You are a helpful assistant."
)
```

## The @tool Decorator

### Basic Tool

```python
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.
    
    Args:
        city: Name of the city (e.g., 'Tokyo', 'New York')
    """
    # Implementation
    return f"Weather in {city}: 72°F, sunny"
```

### Multiple Parameters

```python
@tool
def search_products(
    query: str,
    category: str = "all",
    max_price: float = None,
    in_stock: bool = True
) -> str:
    """Search for products in the catalog.
    
    Args:
        query: Search terms
        category: Product category (electronics, clothing, etc.)
        max_price: Maximum price filter
        in_stock: Only show in-stock items
    """
    results = catalog.search(
        query=query,
        category=category,
        max_price=max_price,
        in_stock=in_stock
    )
    return format_results(results)
```

### With Type Hints

```python
from typing import Optional

@tool
def create_task(
    title: str,
    description: str,
    priority: int = 1,
    due_date: Optional[str] = None
) -> str:
    """Create a new task in the task manager.
    
    Args:
        title: Task title
        description: Detailed description
        priority: Priority level 1-5 (1=highest)
        due_date: Due date in YYYY-MM-DD format
    """
    task = Task(
        title=title,
        description=description,
        priority=priority,
        due_date=due_date
    )
    task.save()
    return f"Created task: {task.id}"
```

## Pydantic Schema Tools

For complex inputs, use Pydantic models:

```python
from langchain.tools import tool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    filters: dict = Field(default={}, description="Optional filters")
    limit: int = Field(default=10, ge=1, le=100, description="Max results")

@tool(args_schema=SearchInput)
def advanced_search(query: str, filters: dict, limit: int) -> str:
    """Perform an advanced search with filters."""
    results = db.search(query, filters=filters, limit=limit)
    return format_results(results)
```

### Complex Nested Schema

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class DateRange(BaseModel):
    start: str = Field(description="Start date YYYY-MM-DD")
    end: str = Field(description="End date YYYY-MM-DD")

class SearchFilters(BaseModel):
    categories: List[str] = Field(default=[], description="Category filters")
    date_range: Optional[DateRange] = Field(default=None, description="Date range")
    author: Optional[str] = Field(default=None, description="Author name")

class DocumentSearchInput(BaseModel):
    query: str = Field(description="Search query")
    filters: SearchFilters = Field(default=SearchFilters(), description="Search filters")
    include_archived: bool = Field(default=False, description="Include archived docs")

@tool(args_schema=DocumentSearchInput)
def search_documents(query: str, filters: SearchFilters, include_archived: bool) -> str:
    """Search documents with advanced filtering."""
    # Implementation
    pass
```

## Tool Classes

For more control, create tool classes:

```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate")

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Evaluate mathematical expressions. Input should be a valid math expression."
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, expression: str) -> str:
        """Execute the tool."""
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def _arun(self, expression: str) -> str:
        """Async execution."""
        return self._run(expression)

# Use it
calculator = CalculatorTool()
result = calculator.invoke({"expression": "2 + 2"})
```

### Tool with State

```python
class DatabaseSearchTool(BaseTool):
    name: str = "database_search"
    description: str = "Search the database"
    
    # Instance variables
    db_connection: Any = None
    cache: dict = {}
    
    def __init__(self, connection_string: str):
        super().__init__()
        self.db_connection = connect(connection_string)
        self.cache = {}
    
    def _run(self, query: str) -> str:
        if query in self.cache:
            return self.cache[query]
        
        results = self.db_connection.search(query)
        self.cache[query] = results
        return results
```

## Writing Good Tool Descriptions

The description is critical—it's how the model decides when to use the tool.

### Bad Description

```python
@tool
def search(q: str) -> str:
    """Search."""  # Too vague!
    pass
```

### Good Description

```python
@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for information about company policies,
    procedures, and documentation.
    
    Use this tool when users ask about:
    - Company policies (HR, IT, security)
    - Internal procedures and processes
    - Product documentation
    - Employee guidelines
    
    Do NOT use for:
    - General knowledge questions
    - Current events
    - External information
    
    Args:
        query: Specific search terms related to internal documentation
    
    Returns:
        Relevant excerpts from matching documents with source citations
    """
    pass
```

### Description Best Practices

```python
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient.
    
    IMPORTANT: Only use this tool when the user explicitly asks to send an email.
    Never use this tool without user confirmation.
    
    Args:
        to: Email address of recipient (must be valid email format)
        subject: Email subject line (keep under 100 characters)
        body: Email body content (supports plain text only)
    
    Returns:
        Confirmation message with email ID
    
    Example usage:
        send_email(
            to="user@example.com",
            subject="Meeting Tomorrow",
            body="Hi, just confirming our meeting at 2pm."
        )
    """
    pass
```

## Error Handling

### Return Errors, Don't Raise

```python
@tool
def fetch_data(url: str) -> str:
    """Fetch data from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.Timeout:
        return "Error: Request timed out. Try again or use a different URL."
    except requests.HTTPError as e:
        return f"Error: HTTP {e.response.status_code}. The URL may be invalid."
    except requests.RequestException as e:
        return f"Error: Could not fetch URL. {str(e)}"
```

### Validation

```python
@tool
def create_user(email: str, name: str, role: str = "user") -> str:
    """Create a new user account."""
    
    # Validate email
    import re
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return "Error: Invalid email format. Please provide a valid email address."
    
    # Validate role
    valid_roles = ["user", "admin", "moderator"]
    if role not in valid_roles:
        return f"Error: Invalid role '{role}'. Must be one of: {', '.join(valid_roles)}"
    
    # Create user
    user = User.create(email=email, name=name, role=role)
    return f"Created user {user.id} with email {email}"
```

## Common Tool Patterns

### Search Tool

```python
@tool
def search(query: str) -> str:
    """Search the knowledge base for relevant information.
    
    Args:
        query: Search query describing what to find
    """
    results = vector_store.similarity_search(query, k=3)
    
    if not results:
        return "No relevant information found."
    
    formatted = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        formatted.append(f"[{i}] Source: {source}\n{doc.page_content}")
    
    return "\n\n".join(formatted)
```

### API Tool

```python
@tool
def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a ticker symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    """
    try:
        response = requests.get(
            f"https://api.stocks.com/price/{symbol}",
            headers={"Authorization": f"Bearer {API_KEY}"}
        )
        response.raise_for_status()
        data = response.json()
        return f"{symbol}: ${data['price']:.2f} ({data['change']:+.2f}%)"
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            return f"Error: Symbol '{symbol}' not found. Check the ticker symbol."
        return f"Error: Could not fetch stock price. Try again later."
```

### Database Tool

```python
@tool
def query_database(sql: str) -> str:
    """Execute a read-only SQL query on the database.
    
    IMPORTANT: Only SELECT queries are allowed. No modifications.
    
    Args:
        sql: SQL SELECT query
    """
    # Validate query
    sql_upper = sql.strip().upper()
    if not sql_upper.startswith("SELECT"):
        return "Error: Only SELECT queries are allowed."
    
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE"]
    if any(word in sql_upper for word in forbidden):
        return "Error: Modification queries are not allowed."
    
    try:
        results = db.execute(sql)
        return format_table(results)
    except Exception as e:
        return f"Error: Query failed - {str(e)}"
```

### File Tool

```python
@tool
def read_file(filepath: str) -> str:
    """Read contents of a file.
    
    Args:
        filepath: Path to the file (relative to workspace)
    """
    import os
    
    # Security: restrict to workspace
    workspace = "/app/workspace"
    full_path = os.path.normpath(os.path.join(workspace, filepath))
    
    if not full_path.startswith(workspace):
        return "Error: Access denied. Can only read files in workspace."
    
    if not os.path.exists(full_path):
        return f"Error: File not found: {filepath}"
    
    try:
        with open(full_path, "r") as f:
            content = f.read()
        return content[:10000]  # Limit size
    except Exception as e:
        return f"Error: Could not read file - {str(e)}"
```

## Async Tools

```python
import aiohttp

@tool
async def async_fetch(url: str) -> str:
    """Fetch data from URL asynchronously."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()
```

## Testing Tools

```python
import pytest

def test_search_tool_returns_results():
    result = search("machine learning")
    assert isinstance(result, str)
    assert "Error" not in result

def test_search_tool_handles_empty_query():
    result = search("")
    # Should return something reasonable
    assert isinstance(result, str)

def test_calculator_valid_expression():
    result = calculator.invoke({"expression": "2 + 2"})
    assert "4" in result

def test_calculator_invalid_expression():
    result = calculator.invoke({"expression": "invalid"})
    assert "Error" in result
```

## Best Practices

1. **Descriptive names.** Tool name should indicate its purpose.

2. **Detailed descriptions.** Include when to use, when not to use, and examples.

3. **Type hints.** Always use type hints for parameters.

4. **Return strings.** Tools should return strings, not complex objects.

5. **Handle errors.** Return error messages, don't raise exceptions.

6. **Validate inputs.** Check inputs before processing.

7. **Limit output size.** Truncate large outputs to avoid context overflow.

8. **Security first.** Validate and sanitize all inputs, especially for file/database tools.

## Related Recipes

- **Creating Agents**: Using tools in agents
- **Similarity Search**: Implementing search tools
- **Streaming**: Tools with streaming output
