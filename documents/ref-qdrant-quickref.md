# Qdrant Quick Reference

## Overview

Qdrant is an open-source vector database optimized for similarity search. It supports filtering, payload storage, and multiple distance metrics. Excellent for RAG applications with its LangChain integration.

## Installation

```bash
uv pip install qdrant-client langchain-qdrant
```

## Client Setup

### In-Memory (Development)

```python
from qdrant_client import QdrantClient

client = QdrantClient(":memory:")
```

### Local Server

```bash
# Start with Docker
docker run -p 6333:6333 qdrant/qdrant
```

```python
client = QdrantClient(host="localhost", port=6333)
```

### Qdrant Cloud (Production)

```python
client = QdrantClient(
    url="https://your-cluster.qdrant.io",
    api_key="your-api-key"
)
```

## Collections

### Create Collection

```python
from qdrant_client.models import Distance, VectorParams

client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=1536,  # Must match embedding dimension
        distance=Distance.COSINE
    )
)
```

### With Multiple Vector Types

```python
from qdrant_client.models import VectorParams, Distance

client.create_collection(
    collection_name="multi_vector",
    vectors_config={
        "content": VectorParams(size=1536, distance=Distance.COSINE),
        "title": VectorParams(size=384, distance=Distance.COSINE)
    }
)
```

### List and Delete

```python
# List collections
collections = client.get_collections()

# Delete collection
client.delete_collection("documents")

# Check if exists
exists = client.collection_exists("documents")
```

## Adding Vectors

### Basic Insert

```python
from qdrant_client.models import PointStruct

points = [
    PointStruct(
        id=1,
        vector=[0.1, 0.2, ...],  # 1536 dimensions
        payload={"text": "Document content", "source": "file.txt"}
    ),
    PointStruct(
        id=2,
        vector=[0.3, 0.4, ...],
        payload={"text": "Another document", "source": "file2.txt"}
    )
]

client.upsert(collection_name="documents", points=points)
```

### Batch Insert

```python
# For large datasets, batch inserts
batch_size = 100
for i in range(0, len(all_points), batch_size):
    batch = all_points[i:i + batch_size]
    client.upsert(collection_name="documents", points=batch)
```

### With UUID IDs

```python
import uuid

point = PointStruct(
    id=str(uuid.uuid4()),
    vector=embedding,
    payload={"text": content}
)
```

## Searching

### Basic Search

```python
results = client.search(
    collection_name="documents",
    query_vector=[0.1, 0.2, ...],
    limit=5
)

for result in results:
    print(f"Score: {result.score}")
    print(f"Text: {result.payload['text']}")
```

### With Filtering

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="technical")
            )
        ]
    ),
    limit=5
)
```

### Complex Filters

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

# AND conditions
filter = Filter(
    must=[
        FieldCondition(key="category", match=MatchValue(value="tech")),
        FieldCondition(key="year", range=Range(gte=2023))
    ]
)

# OR conditions
filter = Filter(
    should=[
        FieldCondition(key="category", match=MatchValue(value="tech")),
        FieldCondition(key="category", match=MatchValue(value="science"))
    ]
)

# NOT conditions
filter = Filter(
    must_not=[
        FieldCondition(key="status", match=MatchValue(value="archived"))
    ]
)

# Combined
filter = Filter(
    must=[
        FieldCondition(key="public", match=MatchValue(value=True))
    ],
    should=[
        FieldCondition(key="category", match=MatchValue(value="tech")),
        FieldCondition(key="category", match=MatchValue(value="science"))
    ],
    must_not=[
        FieldCondition(key="status", match=MatchValue(value="draft"))
    ]
)
```

### Search with Score Threshold

```python
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=10,
    score_threshold=0.7  # Only results with score >= 0.7
)
```

## LangChain Integration

### Setup

```python
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
client = QdrantClient(":memory:")

# Create collection first
from qdrant_client.models import Distance, VectorParams
client.create_collection(
    collection_name="langchain_docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# Create vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name="langchain_docs",
    embedding=embeddings
)
```

### Add Documents

```python
from langchain_core.documents import Document

docs = [
    Document(page_content="First document", metadata={"source": "a.txt"}),
    Document(page_content="Second document", metadata={"source": "b.txt"})
]

vector_store.add_documents(docs)
```

### Search

```python
# Similarity search
results = vector_store.similarity_search("query", k=5)

# With scores
results = vector_store.similarity_search_with_score("query", k=5)
for doc, score in results:
    print(f"{score}: {doc.page_content}")

# MMR search (diversity)
results = vector_store.max_marginal_relevance_search(
    "query",
    k=5,
    fetch_k=20,
    lambda_mult=0.7
)
```

### As Retriever

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# With filtering
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"category": "technical"}
    }
)

# Use in chain
docs = retriever.invoke("What is RAG?")
```

## Payload Management

### Update Payload

```python
client.set_payload(
    collection_name="documents",
    payload={"category": "updated"},
    points=[1, 2, 3]  # Point IDs
)
```

### Delete Payload Keys

```python
client.delete_payload(
    collection_name="documents",
    keys=["old_field"],
    points=[1, 2, 3]
)
```

### Retrieve Points

```python
points = client.retrieve(
    collection_name="documents",
    ids=[1, 2, 3],
    with_payload=True,
    with_vectors=True
)
```

## Indexing

### Payload Indexing (for filtering)

```python
from qdrant_client.models import PayloadSchemaType

client.create_payload_index(
    collection_name="documents",
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)

# For numeric fields
client.create_payload_index(
    collection_name="documents",
    field_name="year",
    field_schema=PayloadSchemaType.INTEGER
)
```

### HNSW Configuration

```python
from qdrant_client.models import HnswConfigDiff

client.update_collection(
    collection_name="documents",
    hnsw_config=HnswConfigDiff(
        m=16,  # Number of edges per node
        ef_construct=100  # Search width during indexing
    )
)
```

## Snapshots and Backup

### Create Snapshot

```python
# Create snapshot
snapshot_info = client.create_snapshot(collection_name="documents")

# List snapshots
snapshots = client.list_snapshots(collection_name="documents")

# Recover from snapshot
client.recover_snapshot(
    collection_name="documents",
    location=snapshot_path
)
```

## Performance Tips

### Batch Operations

```python
# Always batch inserts for large datasets
BATCH_SIZE = 100

for i in range(0, len(points), BATCH_SIZE):
    batch = points[i:i + BATCH_SIZE]
    client.upsert(collection_name="docs", points=batch, wait=False)

# Wait for indexing to complete
client.wait_collection_ready("docs")
```

### Optimize Search

```python
# Use score threshold to reduce results
results = client.search(
    collection_name="docs",
    query_vector=embedding,
    limit=10,
    score_threshold=0.5
)

# Create payload indexes for filtered searches
client.create_payload_index(
    collection_name="docs",
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)
```

### Memory vs Disk

```python
# For large collections, use disk storage
from qdrant_client.models import OptimizersConfigDiff

client.update_collection(
    collection_name="large_docs",
    optimizers_config=OptimizersConfigDiff(
        memmap_threshold=20000  # Store vectors on disk after this many
    )
)
```

## Common Patterns

### Upsert with Deduplication

```python
import hashlib

def get_content_id(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()

def upsert_document(content: str, metadata: dict):
    doc_id = get_content_id(content)
    embedding = get_embedding(content)
    
    client.upsert(
        collection_name="docs",
        points=[PointStruct(
            id=doc_id,
            vector=embedding,
            payload={"content": content, **metadata}
        )]
    )
```

### Hybrid Search (with BM25)

```python
# Qdrant handles vector search
# Combine with external BM25 for hybrid
from rank_bm25 import BM25Okapi

def hybrid_search(query: str, k: int = 5):
    # Vector search
    vector_results = client.search(
        collection_name="docs",
        query_vector=get_embedding(query),
        limit=k * 2
    )
    
    # BM25 search (on same corpus)
    bm25_results = bm25_search(query, k * 2)
    
    # Combine with RRF
    return reciprocal_rank_fusion([vector_results, bm25_results])[:k]
```

## Troubleshooting

### Collection Not Found

```python
# Check if collection exists before operations
if not client.collection_exists("docs"):
    client.create_collection(...)
```

### Dimension Mismatch

```python
# Ensure embedding dimension matches collection config
collection_info = client.get_collection("docs")
expected_dim = collection_info.config.params.vectors.size
# Your embeddings must be this size
```

### Slow Searches

1. Create payload indexes for filtered fields
2. Adjust HNSW parameters
3. Use score threshold to limit results
4. Consider disk storage for large collections

## Related Concepts

- **Vector Databases**: General concepts
- **Embeddings**: What gets stored in Qdrant
- **RAG Fundamentals**: Primary use case
- **LangChain**: Integration framework
