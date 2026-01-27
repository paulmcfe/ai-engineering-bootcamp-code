# Vector Databases

## What Is a Vector Database?

A vector database is a specialized database optimized for storing high-dimensional vectors and performing similarity searches. The core operation—given a query vector, find the most similar vectors—needs to be fast even with millions or billions of stored vectors.

Traditional databases aren't built for this. You could store vectors in PostgreSQL and compute similarity scores for every vector on every query, but that's prohibitively slow at scale. Vector databases use specialized indexing structures that make similarity search orders of magnitude faster.

## Why Vector Databases Matter

In AI applications, you frequently need to find "similar" items: documents related to a query, images that look alike, products a user might like. These similarity operations are fundamental to RAG, recommendation systems, and semantic search.

Without proper indexing, similarity search is O(n)—you compare against every vector. Vector databases reduce this to approximately O(log n) through clever indexing, making real-time search possible even with massive datasets.

## Popular Vector Database Solutions (2025)

### Qdrant

Open-source vector database with excellent performance and a clean API. Supports filtering, payload storage, and multiple distance metrics. Can run in-memory for development or persistent for production. Strong LangChain integration.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# In-memory for development
client = QdrantClient(":memory:")

# Persistent local storage
client = QdrantClient(path="./qdrant_data")

# Remote server
client = QdrantClient(host="localhost", port=6333)

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)
```

### Pinecone

Fully managed vector database as a service. No infrastructure to manage—just use their API. Scales automatically and handles operations complexity. Trade-off is cost and vendor lock-in.

### Chroma

Lightweight, open-source embedding database. Great for prototyping and small projects. Simple API, runs in-memory or with SQLite persistence. Less performant at scale than Qdrant or Pinecone.

### Weaviate

Open-source with hybrid search capabilities (vector + keyword). Good for applications that need both semantic and traditional search. More complex setup than alternatives.

### pgvector

PostgreSQL extension for vector similarity search. Good choice if you're already using PostgreSQL and want to avoid adding another database. Less optimized than purpose-built solutions but simpler operations.

## Indexing Algorithms

### Brute Force (Flat Index)

Compare query against every stored vector. Guarantees finding the true nearest neighbors but doesn't scale. Use only for small datasets (< 10,000 vectors) or when perfect accuracy is required.

### HNSW (Hierarchical Navigable Small World)

The most popular algorithm for approximate nearest neighbor search. Builds a multi-layer graph where vectors are nodes and edges connect similar vectors.

The algorithm navigates from coarse (top layers) to fine (bottom layers), like using highway exits to reach a local destination. Offers excellent speed-accuracy trade-off and is the default choice for most applications.

```python
# Qdrant uses HNSW by default
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    hnsw_config={"m": 16, "ef_construct": 100}  # HNSW parameters
)
```

### IVF (Inverted File Index)

Clusters vectors during indexing, then only searches relevant clusters during query. Good for very large datasets (billions of vectors) where HNSW memory requirements become prohibitive.

### Product Quantization (PQ)

Compresses vectors to reduce memory usage. Trades some accuracy for significant memory savings. Often combined with IVF for large-scale deployments.

## Speed vs. Accuracy Trade-off

Approximate nearest neighbor algorithms don't guarantee finding the true nearest neighbors. They find vectors that are probably close, with configurable accuracy.

Key parameters:
- **ef_search** (HNSW): Higher values search more thoroughly but slower
- **nprobe** (IVF): Number of clusters to search
- **candidates** (general): How many candidates to consider before final ranking

For most RAG applications, approximate search is fine. You're retrieving the top 5-10 documents anyway—if the algorithm returns the 11th best instead of the 10th, it rarely matters.

## Metadata Filtering

Modern vector databases support filtering by metadata alongside vector similarity. This is crucial for production applications.

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Search with metadata filter
results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="technical")
            ),
            FieldCondition(
                key="year",
                range={"gte": 2023}
            )
        ]
    ),
    limit=10
)
```

Common filtering use cases:
- Restricting search to specific document types
- Filtering by date range
- User-specific content (multi-tenancy)
- Access control based on permissions

## LangChain Integration

LangChain provides a unified interface for vector stores:

```python
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient

# Setup
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
client = QdrantClient(":memory:")

# Create vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name="documents",
    embedding=embeddings
)

# Add documents
from langchain.schema import Document

docs = [
    Document(page_content="First document", metadata={"source": "doc1.txt"}),
    Document(page_content="Second document", metadata={"source": "doc2.txt"})
]
vector_store.add_documents(docs)

# Search
results = vector_store.similarity_search("query text", k=5)

# Create retriever for use in chains
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
```

## Choosing a Vector Database

**Use Qdrant when:**
- You need production-grade performance
- You want self-hosted or cloud options
- Filtering and payloads are important
- You value good documentation and DX

**Use Pinecone when:**
- You want fully managed infrastructure
- You don't want to handle operations
- Budget allows for managed service costs

**Use Chroma when:**
- You're prototyping or learning
- Dataset is small (< 100k vectors)
- Simplicity is more important than performance

**Use pgvector when:**
- You're already using PostgreSQL
- You want to minimize infrastructure
- Performance requirements are moderate

## Best Practices

1. **Size your index appropriately**. HNSW uses significant memory. Plan for approximately 1.5x your vector data size.

2. **Batch insertions** when indexing large datasets. Individual inserts are slow; batches of 100-1000 vectors are optimal.

3. **Use metadata strategically**. Store enough to filter effectively but don't bloat payloads with unnecessary data.

4. **Monitor query latency**. If searches slow down, consider adjusting index parameters or scaling infrastructure.

5. **Plan for reindexing**. When you change embedding models or chunking strategies, you'll need to rebuild the entire index.

## Related Concepts

- **Embeddings**: The vectors stored in vector databases
- **Similarity Search**: The core operation vector databases optimize
- **RAG**: Primary use case for vector databases in LLM applications
- **Chunking**: How documents are split before embedding and storage
