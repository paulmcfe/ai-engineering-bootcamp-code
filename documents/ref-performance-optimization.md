# Troubleshooting: Performance Optimization

## Overview

Common performance issues in LLM applications and how to address them.

---

## Slow Response Times

### Symptoms
- Long wait times for responses
- Timeouts
- Poor user experience

### Causes & Solutions

**1. No Streaming**

```python
# BAD: User waits for full response
response = llm.invoke("Tell me a story")
print(response)

# GOOD: Stream tokens as they arrive
for chunk in llm.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

**2. Sequential API Calls**

```python
# BAD: Sequential
results = []
for query in queries:
    result = llm.invoke(query)  # Waits for each one
    results.append(result)

# GOOD: Parallel with asyncio
import asyncio

async def batch_invoke(queries):
    tasks = [llm.ainvoke(q) for q in queries]
    return await asyncio.gather(*tasks)

results = asyncio.run(batch_invoke(queries))
```

**3. Large Context Windows**

```python
# Reduce unnecessary context
def optimize_context(docs: list, max_tokens: int = 3000) -> str:
    """Keep only essential context."""
    
    context = ""
    for doc in docs:
        # Estimate tokens (rough: 1 token ≈ 4 chars)
        if len(context) + len(doc.page_content) > max_tokens * 4:
            break
        context += doc.page_content + "\n\n"
    
    return context
```

**4. Too Many Retrieved Documents**

```python
# BAD: Retrieve many, use all
results = vector_store.similarity_search(query, k=20)

# GOOD: Retrieve more, filter to best
results = vector_store.similarity_search_with_score(query, k=20)
filtered = [doc for doc, score in results if score >= 0.7][:5]
```

---

## High API Costs

### Symptoms
- Unexpected bills
- Budget exceeded
- Cost per query too high

### Causes & Solutions

**1. Using Expensive Models for Simple Tasks**

```python
# Route based on complexity
def choose_model(query: str) -> str:
    # Simple queries → cheap model
    if is_simple_query(query):
        return "gpt-4o-mini"  # Cheaper
    # Complex queries → capable model
    return "gpt-4o"  # More expensive

def is_simple_query(query: str) -> bool:
    # Short queries, simple questions
    return len(query.split()) < 20 and "?" in query
```

**2. No Response Caching**

```python
from functools import lru_cache
import hashlib

# In-memory cache
@lru_cache(maxsize=1000)
def cached_invoke(prompt_hash: str) -> str:
    # This only works with immutable inputs
    pass

def invoke_with_cache(prompt: str) -> str:
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
    return cached_invoke(prompt_hash)

# For more robust caching, use Redis or similar
import redis
r = redis.Redis()

def cached_llm_call(prompt: str, ttl: int = 3600) -> str:
    cache_key = f"llm:{hashlib.md5(prompt.encode()).hexdigest()}"
    
    cached = r.get(cache_key)
    if cached:
        return cached.decode()
    
    result = llm.invoke(prompt)
    r.setex(cache_key, ttl, result.content)
    return result.content
```

**3. No Embedding Caching**

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

# Cache embeddings locally
store = LocalFileStore("./embedding_cache")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=embeddings,
    document_embedding_cache=store,
    namespace="text-embedding-3-small"
)

# Now repeated embeddings are free
```

**4. Verbose System Prompts**

```python
# BAD: Huge system prompt sent every message
system_prompt = "..." * 2000  # 2000 tokens every call!

# GOOD: Concise system prompt
system_prompt = """You are a helpful assistant.
- Answer questions clearly
- Use tools when needed
- Cite sources"""  # ~20 tokens
```

**5. No Token Limits**

```python
# Set max_tokens to limit response length
response = llm.invoke(
    prompt,
    max_tokens=500  # Don't let model ramble
)
```

### Cost Monitoring

```python
def track_usage(response) -> dict:
    """Track token usage for cost monitoring."""
    usage = response.usage
    
    # Approximate costs (check current pricing)
    input_cost = usage.prompt_tokens * 0.00001  # $0.01/1K
    output_cost = usage.completion_tokens * 0.00003  # $0.03/1K
    
    return {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "estimated_cost": input_cost + output_cost
    }
```

---

## Memory Issues

### Symptoms
- Out of memory errors
- Slow due to swapping
- Process killed

### Causes & Solutions

**1. Loading All Documents at Once**

```python
# BAD: Load everything into memory
all_docs = []
for file in files:
    docs = loader.load(file)
    all_docs.extend(docs)  # Memory grows

# GOOD: Process in batches
def process_in_batches(files: list, batch_size: int = 100):
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        docs = []
        for file in batch_files:
            docs.extend(loader.load(file))
        
        # Process batch
        vector_store.add_documents(docs)
        
        # Memory freed after each batch
        del docs
```

**2. Storing Embeddings in Memory**

```python
# BAD: In-memory vector store for large datasets
vector_store = Chroma.from_documents(
    documents,  # All in RAM
    embeddings
)

# GOOD: Use persistent storage
from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_data")  # Disk storage
# Or use a proper database server
client = QdrantClient(url="http://localhost:6333")
```

**3. Large Conversation History**

```python
def manage_memory(messages: list, max_messages: int = 50) -> list:
    """Keep conversation history manageable."""
    
    if len(messages) <= max_messages:
        return messages
    
    # Keep system message + recent messages
    system = [m for m in messages if m["role"] == "system"]
    recent = messages[-max_messages:]
    
    return system + recent
```

---

## Database Performance

### Symptoms
- Slow vector searches
- Indexing takes forever
- Queries timeout

### Causes & Solutions

**1. No Index**

```python
# For Qdrant, ensure HNSW index is built
info = client.get_collection("documents")
if info.status != "green":
    # Wait for indexing
    client.wait_for_collection("documents")
```

**2. Suboptimal Index Parameters**

```python
from qdrant_client.models import HnswConfigDiff

# Tune HNSW parameters
client.update_collection(
    collection_name="documents",
    hnsw_config=HnswConfigDiff(
        m=16,  # More edges = better recall, more memory
        ef_construct=100  # Higher = better index, slower build
    )
)

# For search, adjust ef parameter
results = client.search(
    collection_name="documents",
    query_vector=query_emb,
    limit=10,
    search_params={"ef": 128}  # Higher = better recall, slower
)
```

**3. No Payload Indexes**

```python
from qdrant_client.models import PayloadSchemaType

# Index fields you filter on
client.create_payload_index(
    collection_name="documents",
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)

client.create_payload_index(
    collection_name="documents",
    field_name="date",
    field_schema=PayloadSchemaType.DATETIME
)
```

**4. Not Using Batches**

```python
# BAD: Insert one at a time
for doc in documents:
    vector_store.add_documents([doc])

# GOOD: Batch inserts
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    vector_store.add_documents(batch)
```

---

## Agent Performance

### Symptoms
- Agents take many iterations
- Redundant tool calls
- Slow decision making

### Causes & Solutions

**1. Too Many Tools**

```python
# BAD: 20 tools confuses the model
tools = [tool1, tool2, ..., tool20]

# GOOD: Group related tools or limit selection
# Option 1: Use fewer, more general tools
general_search = create_general_search_tool()

# Option 2: Dynamically select relevant tools
def select_tools(query: str) -> list:
    if "weather" in query.lower():
        return [weather_tool]
    if "calculate" in query.lower():
        return [calculator_tool]
    return [search_tool]  # Default
```

**2. No Early Termination**

```python
def should_continue(state) -> str:
    # Stop if we have a good answer
    if state.get("confidence", 0) > 0.9:
        return END
    
    # Stop if iterations exceeded
    if state.get("iterations", 0) >= 5:
        return END
    
    # Continue if tool calls pending
    if state["messages"][-1].tool_calls:
        return "tools"
    
    return END
```

**3. Verbose Reasoning**

```python
# Tell model to be concise
system_prompt = """...

Keep your reasoning brief. Don't explain obvious decisions.
Get to the answer efficiently."""
```

---

## Profiling and Monitoring

### Time Profiling

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{name}: {elapsed:.2f}s")

# Usage
with timer("Embedding"):
    embedding = embeddings.embed_query(query)

with timer("Search"):
    results = vector_store.similarity_search(query, k=5)

with timer("LLM"):
    response = llm.invoke(prompt)
```

### Memory Profiling

```python
import tracemalloc

tracemalloc.start()

# Your code here
results = process_documents(docs)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")

tracemalloc.stop()
```

### Cost Tracking Dashboard

```python
class UsageTracker:
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0
        self.calls = 0
    
    def track(self, response):
        usage = response.usage
        self.total_tokens += usage.total_tokens
        self.calls += 1
        
        # Estimate cost
        cost = (usage.prompt_tokens * 0.00001 + 
                usage.completion_tokens * 0.00003)
        self.total_cost += cost
    
    def report(self):
        return {
            "total_tokens": self.total_tokens,
            "total_cost": f"${self.total_cost:.4f}",
            "calls": self.calls,
            "avg_tokens_per_call": self.total_tokens / max(self.calls, 1)
        }

tracker = UsageTracker()
```

---

## Quick Optimization Checklist

1. **Response time**
   - [ ] Streaming enabled
   - [ ] Parallel API calls where possible
   - [ ] Context size optimized
   - [ ] Appropriate number of retrieved docs

2. **Costs**
   - [ ] Model routing by complexity
   - [ ] Response caching
   - [ ] Embedding caching
   - [ ] Token limits set
   - [ ] Usage monitoring

3. **Memory**
   - [ ] Batch processing for large datasets
   - [ ] Persistent vector storage
   - [ ] Conversation history limits

4. **Database**
   - [ ] Indexes created
   - [ ] Parameters tuned
   - [ ] Batch operations used

5. **Agents**
   - [ ] Reasonable number of tools
   - [ ] Early termination conditions
   - [ ] Efficient prompts

---

## Performance Baseline Metrics

Typical targets for production systems:

| Metric | Target | Measurement |
|--------|--------|-------------|
| Time to first token | < 500ms | Streaming latency |
| Full response | < 5s | End-to-end for simple queries |
| Vector search | < 100ms | Retrieval latency |
| Embedding | < 200ms | Per query |
| Agent iteration | < 2s | Per tool call cycle |

---

## Related Guides

- **RAG Problems**: When retrieval is the bottleneck
- **Agent Debugging**: When agents are inefficient
