
## Database and Memory Model

Before building our memory store, we need the database infrastructure. We'll use SQLAlchemy as our ORM, which gives us clean Python abstractions over SQL. For the examples in this chapter, we use SQLite—it requires no setup and runs anywhere. In production, you'd swap in PostgreSQL with a single connection string change (we cover that later in the chapter).

```python
from sqlalchemy import create_engine, Column, String, DateTime, JSON, Index
from sqlalchemy.orm import Session, sessionmaker, declarative_base
from datetime import datetime

Base = declarative_base()
engine = create_engine("sqlite:///:memory:")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

The Memory model uses a namespace/key/value structure. Namespaces categorize memories (preferences, facts, goals, episodes). Keys identify specific memories within a category. Values store the actual data as JSON, giving you flexibility without schema migrations:

```python
class Memory(Base):
    """User memory storage organized by namespace.

    Namespaces let you categorize memories for efficient retrieval:
    - preferences: User preferences and settings
    - facts: Known facts about the user
    - episodes: Summaries of past interactions
    - goals: User's stated objectives
    """
    __tablename__ = "memories"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(50), nullable=False, index=True)
    namespace = Column(String(100), nullable=False, index=True)
    key = Column(String(200), nullable=False)
    value = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Composite index for efficient lookups
    __table_args__ = (Index("ix_memory_user_namespace", "user_id", "namespace"),)
```

The composite index on (user_id, namespace) makes lookups fast when retrieving all memories of a particular type for a user.

## Memory System Architecture

Now that we understand why memory matters, let's look at how to build it. We'll create a simple but powerful memory system using SQLAlchemy. No external memory libraries required. This approach gives you full control over your data and makes debugging straightforward.

### The Memory Store Pattern

At the core of any memory system is a store abstraction that handles the basic operations: put, get, search, and delete. Rather than using a third-party library, we'll build a lightweight store class that wraps SQLAlchemy queries. This keeps dependencies minimal and makes the code easy to understand and modify.

The API is straightforward. You put memories into the store with a namespace, key, and value. You get memories back by namespace and key. You search within a namespace to find relevant memories. Here's the pattern:

```python
from datetime import datetime
from sqlalchemy.orm import Session
import uuid

class MemoryStore:
    """Simple memory store backed by SQLAlchemy."""

    def __init__(self, db: Session):
        self.db = db

    def put(self, user_id: str, namespace: str, key: str, value: dict) -> Memory:
        """Store or update a memory."""
        existing = (
            self.db.query(Memory)
            .filter_by(user_id=user_id, namespace=namespace, key=key)
            .first()
        )

        if existing:
            existing.value = value
            existing.updated_at = datetime.utcnow()
            self.db.commit()
            return existing

        memory = Memory(
            id=str(uuid.uuid4()),
            user_id=user_id,
            namespace=namespace,
            key=key,
            value=value,
        )
        self.db.add(memory)
        self.db.commit()
        return memory

    def get(self, user_id: str, namespace: str, key: str) -> dict | None:
        """Get a specific memory by key."""
        memory = (
            self.db.query(Memory)
            .filter_by(user_id=user_id, namespace=namespace, key=key)
            .first()
        )
        return memory.value if memory else None
```

This pattern—namespace plus key equals value—is simple but surprisingly powerful. The namespace organizes memories into categories. The key identifies a specific memory within that category. The value stores the actual data as JSON.

### Giving Agents Memory Tools

Your agent needs a way to interact with the memory store. The cleanest approach is to wrap store operations as tools that the agent can call. This lets the agent decide when to remember something and when to recall past context.

```python
from langchain_core.tools import tool

# Create store with database session
store = MemoryStore(db_session)

@tool
def remember(namespace: str, key: str, fact: str) -> str:
    """Store an important fact about the user or conversation.

    Args:
        namespace: Category like 'preferences', 'goals', or 'struggles'
        key: Identifier for this specific memory
        fact: The information to remember
    """
    store.put(
        user_id=current_user_id,
        namespace=namespace,
        key=key,
        value={"content": fact, "timestamp": datetime.utcnow().isoformat()}
    )
    return f"I'll remember that: {fact}"

@tool
def recall(namespace: str) -> str:
    """Recall memories from a specific category.

    Args:
        namespace: Category to search, like 'preferences' or 'goals'
    """
    memories = store.search(current_user_id, namespace=namespace, limit=5)

    if not memories:
        return f"No memories found in {namespace}."

    return "\n".join([
        f"- {m['key']}: {m['value'].get('content', m['value'])}"
        for m in memories
    ])
```

With these tools, your agent can decide on its own when to store information and when to search for it. The agent might store a preference when a user mentions one, or search for past context when answering a follow-up question.

### Namespace Organization

Namespaces are how you organize memories in the store. Think of them as a hierarchical folder structure for user data. Each namespace groups related memories together, making retrieval efficient and logical.

For multi-user applications, the user ID is your first level of organization. Within each user's memories, namespaces categorize the content:

```python
# Common namespace conventions
"preferences"   # Learning style, communication preferences, settings
"facts"         # Known facts about the user (profession, background)
"goals"         # What they're trying to achieve
"struggles"     # Topics or concepts they find difficult
"sessions"      # Summaries of past interactions
```

When searching, you scope your query to a specific namespace. This keeps searches focused and fast. Looking for what a user struggles with? Search the "struggles" namespace. Need their preferences? Search "preferences". This organization also helps with data management—you can easily clear all session summaries without touching preferences.

### Search and Retrieval Patterns

The most common retrieval pattern is searching within a namespace, ordered by recency. Recent memories are usually more relevant than old ones:

```python
def search(self, user_id: str, namespace: str | None = None, limit: int = 10) -> list[dict]:
    """Search memories, ordered by recency."""
    query = self.db.query(Memory).filter_by(user_id=user_id)

    if namespace:
        query = query.filter_by(namespace=namespace)

    query = query.order_by(Memory.updated_at.desc()).limit(limit)

    return [
        {
            "namespace": m.namespace,
            "key": m.key,
            "value": m.value,
            "updated_at": m.updated_at.isoformat() if m.updated_at else None,
        }
        for m in query.all()
    ]
```

For more sophisticated retrieval, you can add filtering by key patterns, date ranges, or even full-text search on the JSON content. SQLAlchemy's text matching makes this straightforward:

```python
def search_by_content(self, user_id: str, search_term: str, limit: int = 10) -> list[dict]:
    """Search memories by content within the JSON value."""
    from sqlalchemy import cast, String

    query = (
        self.db.query(Memory)
        .filter_by(user_id=user_id)
        .filter(cast(Memory.value, String).ilike(f"%{search_term}%"))
        .order_by(Memory.updated_at.desc())
        .limit(limit)
    )

    return [{"key": m.key, "value": m.value} for m in query.all()]
```

This approach uses SQL pattern matching on the JSON content. It's not as sophisticated as vector similarity search, but it's fast, requires no additional infrastructure, and works well for keyword-based recall. For applications that need semantic search, you can add vector embeddings later; the namespace/key/value structure remains the same.

## Types of Memory

Not all memories are created equal. Just as human memory has different types—facts you know, experiences you've had, skills you've learned—AI memory systems benefit from similar distinctions. Understanding these types helps you design systems that remember the right things in the right ways.

> Humans use memories to remember facts (semantic memory), experiences (episodic memory), and rules (procedural memory). AI agents can use memory in the same ways.
>
> —LangChain Docs, https://docs.langchain.com/oss/python/concepts/memory

### Semantic Memory: Facts and Knowledge

Semantic memory stores facts and knowledge about the user or domain. These are declarative statements: "The user works at a fintech startup." "They prefer Python over JavaScript." "Their company uses PostgreSQL." "They're preparing for AWS certification."

This type of memory drives personalization. When you know facts about a user, you can tailor your responses accordingly. Technical explanations for the engineer. Business-focused summaries for the executive. Examples using their preferred programming language.

Semantic memories tend to be relatively stable. Once you learn that a user prefers detailed explanations, that preference probably won't change next week. This stability makes semantic memory efficient: you store it once and reference it many times.

```python
# Examples of semantic memory
store.put(
    user_id=current_user_id,
    namespace="facts",
    key="profession",
    value={"content": "Senior software engineer at a healthcare company"}
)

store.put(
    user_id=current_user_id,
    namespace="preferences",
    key="explanation_style",
    value={"content": "Prefers code examples over abstract descriptions"}
)
```

### Episodic Memory: Past Experiences

Episodic memory records specific interactions and experiences. "Last Tuesday, we discussed transformer architectures and the user struggled with attention mechanisms." "Three weeks ago, they asked for help debugging a memory leak." "Yesterday, we created a study plan for their certification exam."

While semantic memory captures what you know about someone, episodic memory captures what you've done together. This enables learning from experience. If a particular explanation didn't land well last time, you can try a different approach. If a user asked about a topic before, you can build on that foundation rather than starting over.

Episodic memories are timestamped and contextual. They capture not just the content but the circumstances: when it happened, what was being discussed, how it went. This temporal aspect lets you weight recent experiences more heavily than distant ones.

```python
# Examples of episodic memory
store.put(
    user_id=current_user_id,
    namespace="episodes",
    key=f"session_{session_id}",
    value={
        "content": "Discussed RAG implementation patterns",
        "timestamp": "2025-01-08T14:30:00Z",
        "outcome": "User understood chunking but needed more help with retrieval",
        "topics": ["RAG", "chunking", "vector search"]
    }
)
```

### Procedural Memory: Learned Behaviors

Procedural memory encodes how the agent should behave. These aren't facts about the user, they're patterns and rules the agent has learned to follow. "Always cite sources when answering technical questions." "Ask clarifying questions when queries are ambiguous." "Break down complex explanations into steps."

This type of memory shapes agent behavior rather than agent knowledge. Over time, an agent can learn that certain approaches work better with certain users. Maybe this user responds well to analogies. Maybe that user gets frustrated by too many clarifying questions. Procedural memory captures these behavioral adaptations.

In practice, procedural memory often emerges from analyzing episodic memories. You notice patterns across many interactions and extract rules that should guide future behavior.

### When to Use Each Type

The three memory types serve complementary purposes, and most production systems use all three. Semantic memory powers personalization with stable user facts. Episodic memory enables learning from experience with timestamped interaction records. Procedural memory shapes behavior with learned patterns and rules.

For a customer support agent, semantic memory might store the customer's product tier and technical environment. Episodic memory would track previous support tickets and their resolutions. Procedural memory might capture that this customer prefers brief, action-oriented responses rather than lengthy explanations.

For a research assistant, semantic memory stores the researcher's domain expertise and current projects. Episodic memory records past research sessions and findings. Procedural memory encodes preferences such as "always include citations" or "summarize before diving into details."

The key is matching memory type to information type. Facts go in semantic memory. Events go in episodic memory. Behaviors go in procedural memory. When you keep these distinct, your memory system stays organized and your retrievals stay relevant.

## Memory Formation Strategies

Knowing what to remember is only half the challenge. The other half is knowing when to create memories. There are two fundamental approaches: forming memories during the conversation (hot path), or extracting memories afterward (background). Each has distinct trade-offs, and many production systems use both.

### Hot Path: Active Memory Formation

In the hot path approach, memory formation happens while the agent is actively responding. The agent decides in real-time that something is worth remembering and stores it immediately. This might be triggered by explicit user statements ("Remember that I prefer Python") or by the agent recognizing important information during the conversation.

The implementation typically involves giving your agent a memory tool that it can invoke during conversations.

```python
@tool
def remember_hotpath(fact: str) -> str:
    """
    Store an important fact about the user or conversation.
    Use when the user shares preferences, goals, or important context.
    """
    store.put(
        user_id=current_user_id,
        namespace="facts",
        key=str(uuid.uuid4()),
        value={
            "content": fact,
            "type": "semantic",
            "timestamp": datetime.now().isoformat()
        }
    )
    return f"I'll remember that: {fact}"
```

The advantage of hot path memory is immediacy. The agent has full context when deciding what to remember. It understands the nuance of the conversation, what seems important, what the user emphasized. The memory is stored with that rich contextual understanding.

The downside is latency. Every memory operation adds time to the response. The agent needs to spend tokens reasoning about what to remember, then execute the tool call to store it. For real-time conversational applications, this overhead can be noticeable.

### Background: Passive Memory Extraction

The background approach separates memory formation from the conversation itself. After a conversation ends, a separate process analyzes the transcript and extracts memories. This extraction can be more thorough because it's not competing with response-time constraints.

```python
def extract_memories_background(conversation: list[dict]):
    """Run after conversation ends to extract memories."""

    prompt = """Analyze this conversation and extract:
    1. User preferences mentioned
    2. Important facts shared
    3. Topics discussed
    4. Any commitments or follow-ups needed

    Return as a JSON array of objects with "type" and "content" fields.

    Conversation:
    {conversation}
    """

    extraction = llm.invoke(prompt.format(conversation=json.dumps(conversation)))
    memories = parse_extraction(extraction)

    for memory in memories:
        store.put(
            user_id=current_user_id,
            namespace=memory.get("type", "facts"),
            key=str(uuid.uuid4()),
            value=memory
        )
```

Background extraction has clear advantages. It adds zero latency to conversations. It can analyze the full conversation holistically, identifying patterns that might not be obvious in any single message. It can use larger, more capable models for extraction without impacting response time.

The trade-off is delayed memory. Information from a conversation won't be available until the extraction process runs. If a user mentions their preference early in a session, the agent can't use that memory later in the same session because it hasn't been extracted yet.

### Trade-offs and Decision Framework

When should you use each approach? Hot path memory excels when information needs to be available immediately within the same conversation. If a user says "I'm a visual learner," you want that preference influencing responses right away, not in the next session.

Background extraction excels for comprehensive analysis and long-term memory. After a tutoring session, you might want to analyze: What concepts did we cover? What did the student struggle with? How did the session go overall? These summary-level insights benefit from holistic analysis after the conversation completes.

Latency sensitivity matters too. Customer support chat that needs sub-second responses? Background extraction keeps things snappy. Research assistant where users expect thoughtful pauses? Hot path memory is more acceptable.

### Hybrid Strategies

Most production systems use both approaches. Hot path for explicit memory requests and high-priority information. Background for comprehensive extraction and analysis. The combination gives you the best of both worlds: immediate memory when it matters, plus thorough extraction for long-term learning.

A practical hybrid might work like this: During the conversation, the agent stores memories only when the user explicitly asks ("Remember that...") or when critical information surfaces ("My company deadline is Friday"). After the conversation, a background job extracts additional insights, preferences, and patterns that weren't explicitly stored.

This approach keeps conversations fast while still building comprehensive user understanding over time. The hot path handles urgent memory needs; the background handles everything else.

## Memory Storage and Retrieval

Getting memories into the store is only half the equation. Getting them back out—at the right time, in the right context—is equally important. A memory system that stores everything but retrieves nothing useful is just a database with extra steps.

### Storing Memories with Metadata

When you store a memory, include rich metadata beyond just the content. Timestamps let you weight by recency. Type labels distinguish semantic from episodic. Source information helps with debugging. Topic tags enable efficient filtering.

```python
# Rich memory storage with metadata
store.put(
    user_id=current_user_id,
    namespace="semantic",
    key=str(uuid.uuid4()),
    value={
        "content": "User is preparing for AWS Solutions Architect exam",
        "type": "semantic",
        "source": "explicit_mention",
        "timestamp": datetime.now().isoformat(),
        "topics": ["AWS", "certification", "career"],
        "confidence": 1.0
    }
)
```

This metadata becomes essential during retrieval. You might want only recent memories, or only memories about certain topics, or only high-confidence facts. The richer your metadata, the more precise your retrieval can be.

### Semantic Search for Relevant Memories

The most common retrieval pattern is content-based search: given the current conversation context, find memories that might be relevant. In production you'd use vector similarity search for this; here we use keyword matching via `search_by_content` as a simpler stand-in that requires no additional infrastructure.

```python
def get_relevant_memories(query: str, user_id: str, k: int = 5):
    """Retrieve memories semantically related to the query."""
    # In production, use vector similarity search
    # Here we use keyword matching as a simplified stand-in
    results = store.search_by_content(user_id, query, limit=k)
    return [result["value"] for result in results]
```

Text matching handles exact keyword overlap well. For variations in phrasing—where the user asks about "machine learning" and you want to retrieve memories stored under "ML" or "neural networks"—you'd add vector embeddings. The namespace/key/value structure remains the same; only the search method changes.

### Recency and Relevance Weighting

Pure semantic similarity isn't always optimal. A memory from yesterday is often more relevant than a memory from six months ago, even if the older one scores slightly higher on similarity. Combining recency with relevance produces better results.

```python
def get_memories_with_recency(query: str, user_id: str):
    """Get memories weighted by both relevance and recency."""
    results = store.search(user_id, limit=20)  # Get more candidates for reranking

    scored = []
    now = datetime.utcnow()

    for r in results:
        updated = r.get("updated_at")
        if updated:
            timestamp = datetime.fromisoformat(updated)
        else:
            timestamp = now
        age_hours = (now - timestamp).total_seconds() / 3600
        recency_weight = 1 / (1 + age_hours / 24)  # Decay over days
        scored.append((recency_weight, r))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [r["value"] for _, r in scored[:5]]
```

This decay function can be tuned to your use case. A customer support system might decay memories more slowly; issues from a month ago might still be relevant. A news-focused assistant might decay aggressively; yesterday's context matters more than last week's.

### Memory Consolidation Over Time

As memory accumulates, you'll want strategies for consolidation. Rather than keeping every individual memory forever, you can summarize patterns, merge similar memories, and archive or delete stale information.

Think of it like human memory. You don't remember every individual conversation from last year, but you remember the general patterns and important events. Memory consolidation transforms many specific memories into fewer, more useful summary memories.

A practical consolidation might run weekly: analyze all episodic memories from the past week, extract patterns ("User tends to ask follow-up questions about implementation details"), store those patterns as new semantic memories, and archive the original episodes. The raw detail is preserved if needed, but the working memory stays focused on high-value insights.

## Memory Integration Patterns

You've got memories in your store. Now the question becomes: how do you get them into your agent's context at the right time? There are several integration patterns, each with its own strengths.

### Injection at Start

The simplest pattern is to load relevant memories at the start of every conversation and inject them into the system prompt. Before the agent sees the user's message, it already has context about who it's talking to.

```python
def build_system_prompt(user_id: str, current_query: str):
    """Build a personalized system prompt with relevant memories."""
    
    memories = get_relevant_memories(current_query, user_id)
    
    memory_context = "\n".join([
        f"- {m.get('content', m)}" for m in memories
    ])
    
    return f"""You are a helpful assistant.

What you know about this user:
{memory_context}

Use this context to personalize your responses."""
```

This approach is reliable and predictable. The agent always has memory context available. There's no need for the agent to decide when to search because the relevant information is already there.

The trade-off is token usage. You're including memory context in every request, even when it's not needed. For users with extensive memory, this can eat into your context budget. You'll want to limit how many memories you inject and be thoughtful about relevance thresholds.

### On-Demand Retrieval

An alternative is giving the agent a memory search tool and letting it decide when to retrieve. The agent might search memory at the start of a conversation, or when it encounters something it might have seen before, or when the user references something from the past.

```python
@tool
def recall_on_demand(topic: str) -> str:
    """
    Search memory for information about a topic.
    Use when you need to remember something about the user
    or from past conversations.
    """
    memories = store.search_by_content(current_user_id, topic, limit=5)

    if not memories:
        return "No relevant memories found."

    return "\n".join([f"- {m['value'].get('content', m['value'])}" for m in memories])
```

On-demand retrieval is more token-efficient. The agent only loads memory when it actually needs it. But it requires the agent to be good at recognizing when memory would help. If the agent doesn't think to search, relevant memories go unused.

### Proactive Memory

A more sophisticated pattern is proactive memory, where the system surfaces relevant memories without being asked. Based on the current conversation context, the system identifies memories that might be useful and injects them automatically.

```python
def get_proactive_memories(conversation_context: str, user_id: str):
    """Find memories the user might not ask about but would help."""

    related = store.search_by_content(user_id, conversation_context, limit=5)

    # In production, filter by relevance score (e.g., score > 0.75)
    return [r["value"] for r in related]
```

Proactive memory can create delightful experiences. The agent "remembers" relevant context without being prompted. "Last time we discussed this topic, you mentioned being interested in the production deployment aspects. Want me to focus there again?"

The risk is over-triggering. If you surface too many memories, or memories that aren't actually relevant, the experience feels surveillance-like rather than helpful. High relevance thresholds help, but you'll need to tune based on user feedback.

### Adapting Behavior Based on History

Beyond surfacing content, memory can also adapt how the agent behaves. If you know this user struggled with a concept before, spend more time on it. If you know they prefer brief responses, don't write essays. Memory enables dynamic personalization at the behavior level, not just the content level.

This kind of behavioral adaptation typically comes from procedural memories. "This user responds well to analogies." "This user gets frustrated when asked too many clarifying questions." "This user is an expert, so skip the basics." These patterns shape how the agent approaches interactions, making each conversation feel tailored.

## Moving to PostgreSQL for Production

The SQLite in-memory database we've been using is perfect for learning and prototyping, but your carefully curated memories vanish the moment your process exits. For production applications, you need persistent storage that survives beyond a single session. PostgreSQL is the standard choice, and since we're using SQLAlchemy, switching requires only a connection string change.

### Why PostgreSQL Everywhere

There's a philosophy in software development called "dev-prod parity" and it's the idea that your development environment should match production as closely as possible. When you use one database locally and a different one in production, you're inviting subtle bugs. Queries that work fine in your local SQLite setup might behave differently in your production PostgreSQL. Data types don't map perfectly. You're essentially testing against a different system than you're deploying to.

By running PostgreSQL both locally and in a produ tion environment such as Vercel, we eliminate this entire class of problems. Your local database is a real PostgreSQL instance. Your production database is a real PostgreSQL instance. The only difference is the connection string.

This approach also teaches you skills that transfer directly to professional work. PostgreSQL is the most popular database for production applications. Learning it now—including local setup—prepares you for real-world development environments.

### Setting Up Local PostgreSQL

On macOS, Homebrew makes PostgreSQL installation simple. Install it and start the service:

```bash
# Install PostgreSQL
brew install postgresql@16

# Add to PATH (add this to your ~/.zshrc for persistence)
export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"

# If you add the path to ~/.zshrc, run the following or open a new terminal
source ~/.zshrc

# Start PostgreSQL as a background service
brew services start postgresql@16
```

This installs PostgreSQL 16 and runs it as a background service that starts automatically when you log in. Your data persists in Homebrew's data directory.

Once PostgreSQL is running, create a database for your application:

```python
# Create a database (use your app's name)
createdb myapp
```

That's it. PostgreSQL on macOS uses "peer authentication" by default, meaning your macOS username connects without a password. This is perfect for local development.

On Windows, download the official installer from postgresql.org. On Linux, your distribution's package manager has you covered—apt install postgresql on Ubuntu/Debian, dnf install postgresql-server on Fedora.

To verify your database is running, connect with psql:

```bash
# Connect to local PostgreSQL
psql myapp
```

### Database Schema Design

For agent applications with memory, you typically need tables for users, memories, and any domain-specific data. Here's a schema that covers the common patterns:

```sql
-- Users table with preferences
CREATE TABLE users (
    id VARCHAR(50) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    preferences JSON DEFAULT '{}'
);

-- Memories table for agent context (organized by namespace)
CREATE TABLE memories (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(50) REFERENCES users(id) NOT NULL,
    namespace VARCHAR(100) NOT NULL,
    key VARCHAR(200) NOT NULL,
    value JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX ix_memory_user_namespace ON memories(user_id, namespace);

-- Sessions table for conversation history
CREATE TABLE sessions (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(50) REFERENCES users(id) NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    summary TEXT,
    metadata JSON DEFAULT '{}'
);
CREATE INDEX ix_session_user ON sessions(user_id);

-- Generated content cache (content-addressed)
CREATE TABLE content_cache (
    id VARCHAR(36) PRIMARY KEY,
    content_hash VARCHAR(64) UNIQUE NOT NULL,
    content_type VARCHAR(50) NOT NULL,
    content JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP
);
CREATE INDEX ix_cache_hash ON content_cache(content_hash);
```

The memories table uses a namespace + key structure instead of a single content field. This lets you organize memories by category—preferences, facts, goals, episodes—and look them up efficiently. The value column stores the actual memory data as JSON, giving you flexibility without schema migrations.

The content cache table uses content-addressed storage: hash the inputs, use the hash as a lookup key. If the hash exists, return the cached content. This pattern works for any expensive generation—explanations, summaries, recommendations, or any LLM output you don't want to regenerate unnecessarily.

### Connecting to PostgreSQL

The SQLAlchemy setup we defined at the start of this chapter used SQLite in-memory. To switch to PostgreSQL, you only need to change the engine creation and add connection pool settings. Everything else—the Memory model, MemoryStore class, and all memory tools—stays exactly the same:

```python
import os

POSTGRES_URL = os.environ.get("POSTGRES_URL")
if not POSTGRES_URL:
    raise RuntimeError(
        "POSTGRES_URL environment variable is required. "
        "Set it in your .env file, e.g.: POSTGRES_URL=postgresql://localhost/myapp"
    )

# Vercel uses 'postgres://' but SQLAlchemy requires 'postgresql://'
DATABASE_URL = POSTGRES_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

For local development, add `POSTGRES_URL=postgresql://localhost/myapp` to your `.env` file. The `pool_pre_ping=True` setting verifies connections before using them, which is important for serverless environments where functions can be idle for extended periods. The URL translation from `postgres://` to `postgresql://` handles a Vercel quirk—they use the older URL scheme, but SQLAlchemy 2.0 requires the explicit `postgresql://` prefix.

### Setting Up Vercel Postgres

For production, Vercel Postgres provides a managed PostgreSQL database with a generous free tier. Setup takes about a minute:

1. Go to your Vercel project dashboard
1. Navigate to the Storage tab
1. Click Create Database and select Postgres
Vercel automatically populates the POSTGRES_URL environment variable

1. Redeploy your application
That's it. Both environments use POSTGRES_URL. Locally it points to your development database, and on Vercel it contains the full connection string with credentials. Same code, same database engine, different instances.

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

Your database doubles as a cache store. Store generated content with its content hash, then check the database before calling the LLM. Cache hits are simple SELECT queries, which are orders of magnitude faster than LLM generation.

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
