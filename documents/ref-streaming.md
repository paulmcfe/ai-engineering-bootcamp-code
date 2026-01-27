# Recipe: Streaming Responses

## Goal

Stream LLM responses in real-time for better user experience.

## Quick Start

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

for chunk in llm.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

## OpenAI API Streaming

### Basic Streaming

```python
from openai import OpenAI

client = OpenAI()

stream = client.responses.create(
    model="gpt-4o-mini",
    input="Write a poem about coding",
    stream=True
)

for chunk in stream:
    if chunk.output_text:
        print(chunk.output_text, end="", flush=True)
```

### Collecting Full Response

```python
def stream_and_collect(input_text: str, instructions: str = None) -> str:
    """Stream response and return full text."""
    
    stream = client.responses.create(
        model="gpt-4o-mini",
        input=input_text,
        instructions=instructions,
        stream=True
    )
    
    full_response = []
    for chunk in stream:
        if chunk.output_text:
            print(chunk.output_text, end="", flush=True)
            full_response.append(chunk.output_text)
    
    print()  # Newline at end
    return "".join(full_response)
```

### Async Streaming

```python
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def stream_response(prompt: str):
    stream = await async_client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        stream=True
    )
    
    async for chunk in stream:
        if chunk.output_text:
            yield chunk.output_text
```

## LangChain Streaming

### Basic Streaming

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

for chunk in llm.stream("Explain machine learning"):
    print(chunk.content, end="", flush=True)
```

### Streaming Chains

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Explain {topic} simply")
chain = prompt | llm | StrOutputParser()

for chunk in chain.stream({"topic": "neural networks"}):
    print(chunk, end="", flush=True)
```

### Streaming RAG

```python
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_prompt = ChatPromptTemplate.from_template("""
Answer based on this context:
{context}

Question: {question}
""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

for chunk in rag_chain.stream("What is RAG?"):
    print(chunk, end="", flush=True)
```

## LangGraph Streaming

### Stream Events

```python
from langgraph.graph import StateGraph

# Assuming agent is compiled LangGraph

for event in agent.stream({"messages": [("user", "Hello")]}):
    print(event)
```

### Stream Values

```python
for chunk in agent.stream(
    {"messages": [("user", "Search for AI")]},
    stream_mode="values"
):
    # Each chunk is the full state at that point
    messages = chunk.get("messages", [])
    if messages:
        print(messages[-1])
```

### Stream Updates

```python
for chunk in agent.stream(
    {"messages": [("user", "Hello")]},
    stream_mode="updates"
):
    # Each chunk is just the update from one node
    for node_name, update in chunk.items():
        print(f"Node {node_name}: {update}")
```

### Async Streaming

```python
async for event in agent.astream(
    {"messages": [("user", "Tell me about AI")]}
):
    print(event)
```

## FastAPI Streaming

### Server-Sent Events (SSE)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI

app = FastAPI()
client = OpenAI()

def generate_stream(prompt: str):
    """Generator for streaming response."""
    stream = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        stream=True
    )
    
    for chunk in stream:
        if chunk.output_text:
            yield f"data: {chunk.output_text}\n\n"
    
    yield "data: [DONE]\n\n"

@app.get("/stream")
async def stream_endpoint(prompt: str):
    return StreamingResponse(
        generate_stream(prompt),
        media_type="text/event-stream"
    )
```

### With LangChain

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI

app = FastAPI()
llm = ChatOpenAI(model="gpt-4o-mini")

async def langchain_stream(prompt: str):
    async for chunk in llm.astream(prompt):
        if chunk.content:
            yield f"data: {chunk.content}\n\n"
    yield "data: [DONE]\n\n"

@app.get("/chat")
async def chat_stream(message: str):
    return StreamingResponse(
        langchain_stream(message),
        media_type="text/event-stream"
    )
```

### With Agent Reasoning

```python
async def stream_agent_with_reasoning(message: str):
    """Stream agent response with intermediate steps."""
    
    async for event in agent.astream(
        {"messages": [("user", message)]},
        stream_mode="updates"
    ):
        for node_name, update in event.items():
            if node_name == "agent":
                # Tool calls or intermediate reasoning
                messages = update.get("messages", [])
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            yield f"data: {{\"type\": \"tool_call\", \"tool\": \"{tc['name']}\"}}\n\n"
                    elif msg.content:
                        yield f"data: {{\"type\": \"content\", \"text\": \"{msg.content}\"}}\n\n"
            
            elif node_name == "tools":
                # Tool results
                messages = update.get("messages", [])
                for msg in messages:
                    yield f"data: {{\"type\": \"tool_result\", \"result\": \"{msg.content[:100]}\"}}\n\n"
    
    yield "data: [DONE]\n\n"
```

## Frontend Consumption

### JavaScript/Fetch API

```javascript
async function streamChat(message) {
    const response = await fetch(`/stream?prompt=${encodeURIComponent(message)}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = line.slice(6);
                if (data === '[DONE]') return;
                
                // Append to UI
                document.getElementById('response').textContent += data;
            }
        }
    }
}
```

### EventSource API

```javascript
function streamWithEventSource(message) {
    const eventSource = new EventSource(`/stream?prompt=${encodeURIComponent(message)}`);
    
    eventSource.onmessage = (event) => {
        if (event.data === '[DONE]') {
            eventSource.close();
            return;
        }
        document.getElementById('response').textContent += event.data;
    };
    
    eventSource.onerror = (error) => {
        console.error('Stream error:', error);
        eventSource.close();
    };
}
```

### React Hook

```jsx
function useStreamingChat() {
    const [response, setResponse] = useState('');
    const [isStreaming, setIsStreaming] = useState(false);
    
    const sendMessage = async (message) => {
        setResponse('');
        setIsStreaming(true);
        
        try {
            const res = await fetch(`/stream?prompt=${encodeURIComponent(message)}`);
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ') && line.slice(6) !== '[DONE]') {
                        setResponse(prev => prev + line.slice(6));
                    }
                }
            }
        } finally {
            setIsStreaming(false);
        }
    };
    
    return { response, isStreaming, sendMessage };
}
```

## Handling Tool Calls During Streaming

```python
def stream_with_tools(prompt: str):
    """Stream response, handling tool calls."""
    
    context = prompt
    
    while True:
        stream = client.responses.create(
            model="gpt-4o-mini",
            input=context,
            tools=tools,
            stream=True
        )
        
        collected_content = []
        tool_calls = []
        
        for chunk in stream:
            # Stream content
            if chunk.output_text:
                yield {"type": "content", "text": chunk.output_text}
                collected_content.append(chunk.output_text)
            
            # Collect tool calls (when streaming completes)
            if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
                tool_calls = chunk.tool_calls
        
        # If no tool calls, we're done
        if not tool_calls:
            break
        
        # Execute tools and continue
        tool_results = []
        for tc in tool_calls:
            yield {"type": "tool_call", "tool": tc.function.name}
            result = execute_tool(tc.function.name, tc.function.arguments)
            yield {"type": "tool_result", "result": result[:100]}
            tool_results.append(f"{tc.function.name}: {result}")
        
        # Continue with tool results
        context = f"Tool results:\n" + "\n".join(tool_results) + "\n\nContinue responding to the user."
```

## Error Handling

```python
async def robust_stream(prompt: str):
    """Stream with error handling."""
    
    try:
        async for chunk in llm.astream(prompt):
            yield f"data: {chunk.content}\n\n"
    except Exception as e:
        yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
    finally:
        yield "data: [DONE]\n\n"
```

## Performance Tips

### Buffering

```python
def buffered_stream(prompt: str, buffer_size: int = 5):
    """Buffer small chunks for smoother streaming."""
    
    buffer = []
    
    for chunk in llm.stream(prompt):
        buffer.append(chunk.content)
        
        if len(buffer) >= buffer_size:
            yield "".join(buffer)
            buffer = []
    
    if buffer:
        yield "".join(buffer)
```

### Timeout Handling

```python
import asyncio

async def stream_with_timeout(prompt: str, timeout: float = 30.0):
    """Stream with timeout."""
    
    try:
        async with asyncio.timeout(timeout):
            async for chunk in llm.astream(prompt):
                yield chunk.content
    except asyncio.TimeoutError:
        yield "[Stream timeout]"
```

## Best Practices

1. **Always flush.** Use `flush=True` when printing to ensure immediate output.

2. **Handle [DONE].** Send a clear end signal to clients.

3. **Error handling.** Stream errors gracefully, don't crash.

4. **Buffer wisely.** Small buffers for responsiveness, larger for efficiency.

5. **Cancel support.** Allow clients to cancel streams.

6. **Test both modes.** Ensure your app works with and without streaming.

## Related Recipes

- **Creating Agents**: Agents that stream
- **FastAPI Integration**: API streaming patterns
