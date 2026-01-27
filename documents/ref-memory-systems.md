# Memory Systems

## The Memory Problem

LLMs are stateless. Each request starts fresh with no memory of previous interactions. The context window provides temporary memory within a conversation, but it has limits—typically 8K to 200K tokens. Once a conversation exceeds this, older content is lost.

For agents that need to maintain state across sessions, remember user preferences, or learn from past interactions, explicit memory systems are required.

## Types of Memory

Memory in AI systems mirrors human memory in useful ways.

### Semantic Memory

Stores facts and knowledge. "The user prefers detailed explanations." "They're studying for AWS certification." "Their company uses Python 3.11."

Semantic memory drives personalization. It captures what you know about a user or domain that should inform future interactions.

### Episodic Memory

Records specific interactions and experiences. "Last Tuesday, we discussed transformer architectures." "The user struggled with the attention mechanism explanation."

Episodic memory enables learning from experience. You can reference past conversations, avoid repeating mistakes, and build on previous work.

### Procedural Memory

Encodes behaviors and patterns. "Always cite sources when answering technical questions." "Ask clarifying questions when queries are ambiguous."

Procedural memory shapes how the agent behaves rather than what it knows.

## Memory Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AGENT                                   │
├─────────────────────────────────────────────────────────────┤
│  Context Window (Short-Term)                                │
│  - Current conversation                                      │
│  - Retrieved memories                                        │
│  - Active task state                                        │
├─────────────────────────────────────────────────────────────┤
│  Memory Store (Long-Term)                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Semantic   │  │  Episodic   │  │ Procedural  │         │
│  │   Facts     │  │   Events    │  │  Behaviors  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## StudyBuddy v6 Memory Implementation

StudyBuddy v6 uses a custom SQLAlchemy-based memory store backed by PostgreSQL. This avoids external dependencies and gives us full control over the memory schema.

### Database Schema

Memories are stored with a namespace-key-value structure:

```python
class Memory(Base):
    """User memory storage for learning preferences, struggles, goals."""

    __tablename__ = "memories"

    id = Column(String(36), primary_key=True)
    user_id = Column(String(50), ForeignKey("users.id"), nullable=False, index=True)
    namespace = Column(String(100), nullable=False, index=True)
    key = Column(String(200), nullable=False)
    value = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

### Namespace Conventions

Memories are organized by namespace for easy categorization:

- `preferences`: Learning style, favorite topics, explanation depth
- `struggles`: Topics the user finds difficult
- `goals`: What the user is studying for
- `sessions`: Summaries of past study sessions

### The MemoryStore Class

```python
class MemoryStore:
    """Simple memory store backed by SQLAlchemy.

    Usage:
        store = MemoryStore(db_session)
        store.put("default", "preferences", "learning_style", {"type": "visual"})
        memories = store.search("default", "preferences")
    """

    def __init__(self, db: Session):
        self.db = db

    def put(self, user_id: str, namespace: str, key: str, value: dict) -> Memory:
        """Store or update a memory."""
        # Check for existing memory
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
        else:
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

    def search(self, user_id: str, namespace: str | None = None, limit: int = 10) -> list[dict]:
        """Search memories in a namespace."""
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

    def delete(self, user_id: str, namespace: str, key: str) -> bool:
        """Delete a specific memory."""
        memory = (
            self.db.query(Memory)
            .filter_by(user_id=user_id, namespace=namespace, key=key)
            .first()
        )

        if memory:
            self.db.delete(memory)
            self.db.commit()
            return True
        return False
```

## Memory Formation

### Hot Path Memory

Memory created during the conversation while the agent is actively responding.

```python
# During tutoring, remember what the user struggles with
def remember_struggle(user_id: str, topic: str, context: str, db: Session):
    """Record that a user is struggling with a topic."""
    store = MemoryStore(db)
    store.put(
        user_id=user_id,
        namespace="struggles",
        key=topic.lower().replace(" ", "_"),
        value={
            "topic": topic,
            "context": context,
            "recorded_at": datetime.utcnow().isoformat(),
        }
    )
```

**Pros:** Immediate, agent can decide what's important in context.
**Cons:** Adds latency, uses tokens for memory operations.

### Background Memory

Memory extracted from conversations after they complete.

```python
def extract_memories_from_conversation(conversation_text: str, llm) -> list[dict]:
    """Extract memorable information from conversation text.

    Uses LLM to identify information worth remembering:
    - User preferences
    - Learning goals
    - Struggle areas
    - Important context
    """
    system_prompt = """Extract memorable information from this conversation that would be useful to remember for future tutoring sessions.

Look for:
- Learning preferences (visual learner, prefers examples, etc.)
- Goals (studying for certification, learning for job, etc.)
- Struggle areas (topics they find confusing)
- Background info (experience level, related knowledge)

Return a JSON array of memories to store:
[
    {
        "namespace": "preferences|goals|struggles|background",
        "key": "short_identifier",
        "value": {"description": "...", "context": "..."}
    }
]

If nothing memorable, return an empty array: []"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Conversation:\n{conversation_text}"),
    ]

    response = llm.invoke(messages)
    return json.loads(response.content)
```

**Pros:** Doesn't add latency to conversations, can do deeper analysis.
**Cons:** May miss nuance, extraction quality varies.

## Memory Storage with PostgreSQL

StudyBuddy v6 uses PostgreSQL for persistence, which works on both local development and Vercel serverless.

### Connection Setup

```python
# PostgreSQL connection (required)
POSTGRES_URL = os.environ.get("POSTGRES_URL")

if not POSTGRES_URL:
    raise RuntimeError(
        "POSTGRES_URL environment variable is required. "
        "Set it in your .env file, e.g.: POSTGRES_URL=postgresql://localhost/studybuddy"
    )

# Fix URL scheme for SQLAlchemy 2.0 (Vercel uses postgres://)
DATABASE_URL = POSTGRES_URL.replace("postgres://", "postgresql://", 1)
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,  # Verify connections before using
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

### Using the Memory Store

```python
from api.database.connection import get_db
from api.services.memory_store import MemoryStore

# In an endpoint or agent
with get_db() as db:
    store = MemoryStore(db)

    # Store a learning preference
    store.put(
        user_id="default",
        namespace="preferences",
        key="explanation_style",
        value={"style": "detailed", "prefers_examples": True}
    )

    # Retrieve user's preferences
    prefs = store.search("default", "preferences")

    # Get a specific memory
    style = store.get("default", "preferences", "explanation_style")
```

## Memory Integration Patterns

### Injection at Start

Load relevant memories into system prompt:

```python
def build_personalized_prompt(user_id: str, db: Session):
    store = MemoryStore(db)

    # Get user's learning preferences
    preferences = store.search(user_id, "preferences")
    struggles = store.search(user_id, "struggles")

    memory_context = ""
    if preferences:
        memory_context += "User preferences:\n"
        for p in preferences:
            memory_context += f"- {p['key']}: {p['value']}\n"

    if struggles:
        memory_context += "\nTopics user struggles with:\n"
        for s in struggles:
            memory_context += f"- {s['key']}\n"

    return f"""You are a helpful tutor.

{memory_context}

Use this context to personalize your explanations."""
```

### Using Memories in Spaced Repetition

StudyBuddy v6 combines memory with spaced repetition to identify struggle areas:

```python
def get_study_stats(db: Session, user_id: str) -> dict:
    """Get study statistics including struggle areas from review history."""

    # Find cards with low ease factor (struggled cards)
    struggle_cards = (
        db.query(CardReview)
        .filter_by(user_id=user_id)
        .filter(CardReview.ease_factor < 2.0)
        .all()
    )

    # Group by topic
    struggle_topics = {}
    for review in struggle_cards:
        topic = review.flashcard.topic
        struggle_topics[topic] = struggle_topics.get(topic, 0) + 1

    # Sort by struggle count
    sorted_struggles = sorted(struggle_topics.items(), key=lambda x: -x[1])

    return {
        "struggle_areas": [topic for topic, _ in sorted_struggles[:5]],
        # ... other stats
    }
```

## Privacy and Data Governance

Memory systems store personal data. Consider:

### User Control

```python
def delete_user_memory(user_id: str, namespace: str, key: str, db: Session) -> bool:
    """Allow users to delete specific memories."""
    store = MemoryStore(db)
    return store.delete(user_id, namespace, key)

def list_user_memories(user_id: str, db: Session) -> list[dict]:
    """Show all stored memories for transparency."""
    store = MemoryStore(db)
    return store.search(user_id, limit=100)
```

### Data Retention

```python
def cleanup_old_memories(max_age_days: int = 90, db: Session):
    """Automatic cleanup of old memories."""
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)

    db.query(Memory).filter(Memory.updated_at < cutoff).delete()
    db.commit()
```

## Graceful Degradation

Memory systems can fail. Handle gracefully:

```python
def get_memories_safe(user_id: str, namespace: str, db: Session) -> list[dict]:
    """Get memories with fallback on failure."""
    try:
        store = MemoryStore(db)
        return store.search(user_id, namespace, limit=10)
    except Exception as e:
        logger.warning(f"Memory retrieval failed: {e}")
        return []  # Continue without memories
```

## Memory vs. RAG

Memory and RAG both provide context, but serve different purposes:

| Aspect | Memory | RAG |
|--------|--------|-----|
| Content | User-specific, interaction history | Document corpus, knowledge base |
| Scope | Per-user, personal | Shared across users |
| Updates | Continuous, from interactions | Batch, from document changes |
| Query type | "What do I know about this user?" | "What information exists about X?" |

StudyBuddy v6 uses both: RAG for course content, memory for user personalization.

## Best Practices

1. **Be selective** about what to remember. Not everything is worth storing.

2. **Use namespaces** to organize memories logically.

3. **Provide user transparency**. Let users see and manage their memories.

4. **Set retention policies**. Old memories should expire or be consolidated.

5. **Handle failures gracefully**. Agents should work (less well) without memory.

6. **Keep it simple**. A SQLAlchemy model with namespace-key-value is often enough.

## Related Concepts

- **Agents**: Primary users of memory systems
- **RAG**: Complementary knowledge retrieval
- **Spaced Repetition**: Uses memory to track learning progress
- **PostgreSQL**: Production-grade storage backend
