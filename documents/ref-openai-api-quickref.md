# OpenAI API Quick Reference

## Overview

OpenAI provides APIs for language models (GPT-5, GPT-4o, GPT-4o-mini), embeddings, and other AI capabilities. The Python SDK was significantly updated in 2025 with the Responses API as the primary interface.

## Installation

```bash
uv pip install openai
```

## Client Setup

```python
from openai import OpenAI

# Uses OPENAI_API_KEY env var by default
client = OpenAI()

# Or explicit API key
client = OpenAI(api_key="sk-...")

# With organization
client = OpenAI(
    api_key="sk-...",
    organization="org-..."
)
```

## Responses API (Primary Interface)

The Responses API is the recommended interface for most use cases.

### Basic Response

```python
response = client.responses.create(
    model="gpt-4o-mini",
    input="What is the capital of France?"
)

print(response.output_text)
```

### With Instructions (System Prompt)

```python
response = client.responses.create(
    model="gpt-4o-mini",
    instructions="You are a helpful tutor. Explain concepts clearly and concisely.",
    input="What is machine learning?"
)

print(response.output_text)
```

### With Context (for RAG)

```python
response = client.responses.create(
    model="gpt-4o-mini",
    instructions="Answer based only on the provided context. If the answer isn't in the context, say so.",
    input="What is the main topic?",
    context="The document discusses machine learning algorithms and their applications in natural language processing..."
)

print(response.output_text)
```

### With Parameters

```python
response = client.responses.create(
    model="gpt-4o-mini",
    instructions="You are a creative writing assistant.",
    input="Write a haiku about coding",
    temperature=0.7,      # 0-2, higher = more creative
    max_tokens=100        # Maximum response length
)
```

### Streaming

```python
stream = client.responses.create(
    model="gpt-4o-mini",
    input="Tell me a story about a robot",
    stream=True
)

for chunk in stream:
    if chunk.output_text:
        print(chunk.output_text, end="", flush=True)
```

## Function Calling / Tools

### Define Tools

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = client.responses.create(
    model="gpt-4o-mini",
    input="What's the weather in Tokyo?",
    tools=tools
)
```

### Handle Tool Calls

```python
if response.tool_calls:
    for tool_call in response.tool_calls:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        # Execute function
        if name == "get_weather":
            result = get_weather(**args)
        
        # Send result back for final response
        final = client.responses.create(
            model="gpt-4o-mini",
            input=f"Tool result: {json.dumps(result)}",
            instructions="Use the tool result to answer the user's question."
        )
        print(final.output_text)
```

## Structured Outputs

### JSON Mode

```python
response = client.responses.create(
    model="gpt-4o-mini",
    input="List 3 programming languages with their main use cases",
    instructions="Respond with valid JSON only. No markdown, no explanation.",
    response_format={"type": "json_object"}
)

data = json.loads(response.output_text)
```

### With Pydantic Schema

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float
    keywords: list[str]

response = client.responses.create(
    model="gpt-4o-mini",
    input="Analyze: I love this product!",
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "analysis",
            "schema": Analysis.model_json_schema()
        }
    }
)

result = Analysis.model_validate_json(response.output_text)
```

## Embeddings

### Single Text

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="What are embeddings?"
)

vector = response.data[0].embedding  # List of 1536 floats
```

### Batch Embeddings

```python
texts = ["First document", "Second document", "Third document"]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

vectors = [item.embedding for item in response.data]
```

### With Dimensions (text-embedding-3 only)

```python
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Shorter embedding",
    dimensions=512  # Reduce from 1536 to 512
)
```

## Vision

### Image from URL

```python
response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    ]
)

print(response.output_text)
```

### Image from Base64

```python
import base64

with open("image.png", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
    ]
)

print(response.output_text)
```

## Async Usage

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def get_response(prompt: str) -> str:
    response = await client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )
    return response.output_text

# Async streaming
async def stream_response(prompt: str):
    stream = await client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        stream=True
    )
    
    async for chunk in stream:
        if chunk.output_text:
            yield chunk.output_text
```

## Error Handling

```python
from openai import (
    APIError,
    APIConnectionError,
    RateLimitError,
    AuthenticationError
)

try:
    response = client.responses.create(
        model="gpt-4o-mini",
        input="Hello"
    )
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Rate limit exceeded, retry later")
except APIConnectionError:
    print("Network error")
except APIError as e:
    print(f"API error: {e}")
```

### Retry with Backoff

```python
import time
from openai import RateLimitError

def call_with_retry(input_text: str, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return client.responses.create(
                model="gpt-4o-mini",
                input=input_text
            )
        except RateLimitError:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # Exponential backoff
                time.sleep(wait)
            else:
                raise
```

## Models

### Current Models (January 2026)

| Model | Context | Use Case |
|-------|---------|----------|
| gpt-4o | 128K | Most capable |
| gpt-4o-mini | 128K | Fast, cost-effective |
| gpt-4o-mini | 128K | Balanced |
| text-embedding-3-small | 8K | Embeddings (1536d) |
| text-embedding-3-large | 8K | Embeddings (3072d) |

### List Available Models

```python
models = client.models.list()
for model in models.data:
    print(model.id)
```

## Token Counting

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Count message tokens
def count_message_tokens(messages: list, model: str = "gpt-4o-mini") -> int:
    encoding = tiktoken.encoding_for_model(model)
    total = 0
    for message in messages:
        total += 4  # Message overhead
        for key, value in message.items():
            total += len(encoding.encode(str(value)))
    total += 2  # Reply priming
    return total
```

## Best Practices

### Instructions (System Prompts)

```python
# Be specific and structured
instructions = """You are a helpful coding assistant.

Rules:
- Provide working code examples
- Explain your reasoning
- Use Python unless asked otherwise
- Keep responses concise"""

response = client.responses.create(
    model="gpt-4o-mini",
    instructions=instructions,
    input="How do I read a CSV file?"
)
```

### Cost Management

```python
# Use appropriate model for task
# gpt-4o-mini for simple tasks, gpt-4o for complex reasoning

# Set max_tokens to limit response length
response = client.responses.create(
    model="gpt-4o-mini",
    input="Explain quantum computing",
    max_tokens=500  # Limit output
)

# Track usage
print(f"Tokens used: {response.usage.total_tokens}")
```

### Temperature Guidelines

- `0.0`: Deterministic, factual tasks
- `0.3-0.5`: Balanced, most use cases
- `0.7-1.0`: Creative writing
- `1.0+`: Very creative, may be incoherent

## Related Concepts

- **LangChain**: Framework that wraps OpenAI
- **Embeddings**: Vector representations
- **Prompt Engineering**: Getting better outputs
- **Function Calling**: Tool use patterns
