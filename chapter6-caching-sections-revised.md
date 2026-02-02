## Performance Optimization Through Caching

Here's a reality check: LLM calls are expensive. Every token costs money, and every request takes time. If you're regenerating the same content repeatedly—producing the same explanation, extracting the same memory patterns, summarizing the same conversation—you're burning money and making users wait for no good reason.

Caching solves this. Store the results of expensive operations, and serve them instantly when the same request comes in again. This isn't just a geeky optimization tweak. For many applications, it's the difference between viable and unaffordable.

### The Cost of Redundant Generation

Consider the background memory extraction we discussed earlier. Every time a conversation ends, you're calling an LLM to analyze the transcript and extract memories. But what if multiple users have similar conversations? What if your research assistant handles dozens of requests about the same topic? Without caching, you're paying for duplicate analysis every single time.

Let's say memory extraction costs $0.02 per conversation in LLM tokens. You have 1,000 users, and 30% of their conversations cover similar ground—common questions, standard onboarding flows, frequently discussed topics. Without caching, you're paying full price for every extraction. With caching, you extract once and reuse the results when similar patterns appear. Those 300 redundant extractions become database lookups costing essentially nothing.

### Content-Addressed Caching Strategies

Content-addressed caching uses the input content itself as the cache key. Hash the input, use that hash to look up cached results. If the hash matches, you have a cache hit, so return the stored result. If not, generate fresh content and store it under that hash.

```python
import hashlib
import json

def get_content_hash(content: str, params: dict) -> str:
    """Generate a deterministic hash for cache lookup."""
    cache_key_data = {
        "content": content,
        "params": params
    }
    key_string = json.dumps(cache_key_data, sort_keys=True)
    return hashlib.sha256(key_string.encode()).hexdigest()
```

The beauty of content-addressed caching is that identical inputs always produce the same hash. It doesn't matter which user requests it or when. If the input matches, the cached result applies.

### Database-Backed Performance Wins

Your PostgreSQL database doubles as a cache store. Store generated content with its content hash, then check the database before calling the LLM. Cache hits are simple SELECT queries, which are orders of magnitude faster than LLM generation.

```python
def get_or_generate_explanation(topic: str, context: str, user_level: str) -> str:
    """Return cached explanation if available, otherwise generate new one."""
    
    # Check cache first
    content_hash = get_content_hash(topic + context, {"level": user_level})
    cached = db.query(ExplanationCache).filter_by(content_hash=content_hash).first()
    
    if cached:
        cached.access_count += 1
        cached.last_accessed = datetime.utcnow()
        db.commit()
        return cached.explanation
    
    # Cache miss - generate new explanation
    explanation = generate_explanation_with_llm(topic, context, user_level)
    
    # Store for future requests
    db.add(ExplanationCache(
        id=str(uuid.uuid4()),
        content_hash=content_hash,
        topic=topic,
        user_level=user_level,
        explanation=explanation
    ))
    db.commit()
    
    return explanation
```

This pattern works for any expensive generation: explanations, summaries, memory extractions, analysis reports. Anywhere you're calling an LLM with potentially repeatable input, caching can help.

## Caching Generated Content

Let's go deeper into the mechanics of caching generated content. The pattern is straightforward, but the details matter for reliability and performance.

### Hashing Inputs for Cache Keys

Your cache key needs to capture everything that affects the output. If you're generating explanations, that includes the topic, the source material, and any parameters like expertise level or explanation style. Miss any input and you'll serve stale or incorrect cached results.

```python
def build_cache_key(topic: str, source_docs: list[str], params: dict) -> str:
    """Build a comprehensive cache key."""
    
    # Sort everything for deterministic ordering
    key_components = {
        "topic": topic,
        "sources": sorted(source_docs),
        "params": dict(sorted(params.items()))
    }
    
    key_string = json.dumps(key_components, sort_keys=True)
    return hashlib.sha256(key_string.encode()).hexdigest()
```

A common mistake is including non-deterministic elements in your cache key, such as timestamps, random IDs, or mutable objects. These cause constant cache misses even when the meaningful input hasn't changed. Be deliberate about what goes into the hash.

### Storing Generated Outputs

Once you've generated content, store it with enough metadata to be useful later. The content itself, obviously, but also when it was created, what inputs produced it, and maybe statistics like token usage.

```python
class GeneratedContent(Base):
    __tablename__ = "generated_content"
    
    id = Column(String, primary_key=True)
    content_hash = Column(String, unique=True, index=True)
    content_type = Column(String)  # "explanation", "summary", "extraction", etc.
    content = Column(JSON)
    input_summary = Column(String)  # For debugging
    created_at = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime)
```

The access tracking (count and last accessed time) helps with cache management. You can identify popular content worth keeping and stale content worth evicting. Even with generous storage limits, tracking usage patterns helps optimize your cache strategy.

### Measuring Performance Improvements

How do you know caching is actually helping? Instrument your cache with metrics. Track hits versus misses. Measure latency with and without cache. Calculate cost savings from avoided LLM calls.

```python
class CacheMetrics:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.total_latency_saved_ms = 0
        self.estimated_cost_saved = 0.0
    
    def record_hit(self, latency_saved_ms: int, cost_saved: float):
        self.hits += 1
        self.total_latency_saved_ms += latency_saved_ms
        self.estimated_cost_saved += cost_saved
    
    def record_miss(self):
        self.misses += 1
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
```

Good caching can yield dramatic improvements. Hit rates of 60-80% are common for applications with repeat queries. At those rates, you're cutting LLM costs by more than half while serving most responses instantly.

### When to Cache

Not everything benefits from caching. Cache when the operation is expensive (LLM calls), when inputs repeat (common topics, standard questions), and when outputs are deterministic or semi-deterministic (same input produces same or similar output).

Don't cache when freshness matters (current events, time-sensitive data), when personalization makes outputs unique (heavily user-specific responses), or when storage costs exceed generation costs (tiny operations, huge outputs).

The sweet spot for caching is expensive operations with moderate output size and repeatable inputs. Concept explanations hit all three: LLM calls are expensive, explanations are moderate in size, and many users ask about the same topics. Memory extraction is another good candidate—conversation patterns repeat, and the extracted insights can be reused when similar conversations occur.
