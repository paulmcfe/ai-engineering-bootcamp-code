# Recipe: Similarity Search

## Goal

Search a vector database for documents similar to a query.

## Quick Start

```python
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

# Assuming vector_store is already set up and indexed
results = vector_store.similarity_search("What is machine learning?", k=5)

for doc in results:
    print(f"Content: {doc.page_content[:100]}...")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print("---")
```

## Search Methods

### Basic Similarity Search

```python
# Returns Document objects
results = vector_store.similarity_search(
    query="How do neural networks work?",
    k=5
)
```

### With Relevance Scores

```python
# Returns (Document, score) tuples
results = vector_store.similarity_search_with_score(
    query="How do neural networks work?",
    k=5
)

for doc, score in results:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content[:100]}...")
    print("---")
```

### Maximum Marginal Relevance (MMR)

Balances relevance with diversity to avoid redundant results:

```python
results = vector_store.max_marginal_relevance_search(
    query="machine learning",
    k=5,           # Number of results to return
    fetch_k=20,    # Number to fetch before reranking
    lambda_mult=0.7  # 0=max diversity, 1=max relevance
)
```

### With Metadata Filtering

```python
# Filter by exact match
results = vector_store.similarity_search(
    query="machine learning",
    k=5,
    filter={"category": "technical"}
)

# Complex filters (Qdrant syntax)
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

results = vector_store.similarity_search(
    query="machine learning",
    k=5,
    filter=Filter(
        must=[
            FieldCondition(key="year", range=Range(gte=2023)),
            FieldCondition(key="status", match=MatchValue(value="published"))
        ]
    )
)
```

## Using Retrievers

### Basic Retriever

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Invoke retriever
docs = retriever.invoke("What is RAG?")
```

### Configurable Retriever

```python
# Similarity search retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# MMR retriever
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7}
)

# With score threshold
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 10}
)
```

### Multi-Query Retriever

Generates multiple query variations for better coverage:

```python
from langchain.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    llm=llm
)

# Automatically generates query variations
docs = multi_retriever.invoke("How does attention work in transformers?")
```

### Contextual Compression

Extracts only relevant parts from retrieved documents:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store.as_retriever(search_kwargs={"k": 10})
)

# Returns compressed, relevant excerpts
docs = compression_retriever.invoke("What is the attention mechanism?")
```

## Advanced Patterns

### Hybrid Search (Semantic + Keyword)

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25 keyword retriever
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# Vector retriever
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Combine
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.4, 0.6]  # Adjust based on your needs
)

docs = hybrid_retriever.invoke("machine learning algorithms")
```

### Search with Reranking

```python
import cohere

co = cohere.Client()

def search_with_rerank(query: str, k: int = 5) -> list:
    """Search then rerank for better results."""
    
    # Initial broad search
    initial = vector_store.similarity_search(query, k=k * 4)
    
    if not initial:
        return []
    
    # Rerank with Cohere
    docs_text = [doc.page_content for doc in initial]
    
    rerank_response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs_text,
        top_n=k
    )
    
    # Return reranked documents
    return [initial[r.index] for r in rerank_response.results]
```

### Query Expansion

```python
def expand_query(query: str, llm) -> list[str]:
    """Generate query variations."""
    
    prompt = f"""Generate 3 alternative search queries for:
    
Original: {query}

Create variations using:
1. Synonyms
2. More specific terms
3. More general terms

Return as JSON list of strings."""

    response = llm.invoke(prompt)
    import json
    variations = json.loads(response.content)
    
    return [query] + variations

def search_with_expansion(query: str, k: int = 5) -> list:
    """Search with query expansion."""
    
    queries = expand_query(query, llm)
    all_results = []
    seen_content = set()
    
    for q in queries:
        results = vector_store.similarity_search(q, k=k)
        for doc in results:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                all_results.append(doc)
    
    # Return top k unique results
    return all_results[:k]
```

### Self-Query (Natural Language Filters)

```python
from langchain.retrievers import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="category",
        description="Document category: technical, tutorial, reference",
        type="string"
    ),
    AttributeInfo(
        name="date",
        description="Publication date in YYYY-MM-DD format",
        type="string"
    ),
    AttributeInfo(
        name="author",
        description="Document author name",
        type="string"
    )
]

self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_store,
    document_contents="Technical documentation about AI and machine learning",
    metadata_field_info=metadata_field_info
)

# Natural language with implicit filters
docs = self_query_retriever.invoke(
    "Recent tutorials about RAG written by John"
)
# Automatically extracts: category=tutorial, topic=RAG, author=John, date=recent
```

## Direct Qdrant Search

For more control, use the Qdrant client directly:

```python
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings

client = QdrantClient(":memory:")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def direct_search(query: str, k: int = 5, score_threshold: float = 0.0):
    """Search Qdrant directly."""
    
    # Get query embedding
    query_vector = embeddings.embed_query(query)
    
    # Search
    results = client.search(
        collection_name="documents",
        query_vector=query_vector,
        limit=k,
        score_threshold=score_threshold
    )
    
    return [
        {
            "content": r.payload.get("page_content", ""),
            "metadata": r.payload.get("metadata", {}),
            "score": r.score
        }
        for r in results
    ]
```

## Performance Tips

### Batch Searches

```python
async def batch_search(queries: list[str], k: int = 5):
    """Search multiple queries efficiently."""
    
    import asyncio
    
    async def search_one(query):
        return await vector_store.asimilarity_search(query, k=k)
    
    results = await asyncio.gather(*[search_one(q) for q in queries])
    return dict(zip(queries, results))
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_search(query: str, k: int = 5) -> tuple:
    """Cache search results."""
    results = vector_store.similarity_search(query, k=k)
    # Convert to tuple for hashability
    return tuple((doc.page_content, doc.metadata) for doc in results)
```

## Common Issues

### No Results

```python
def robust_search(query: str, k: int = 5) -> list:
    """Search with fallback strategies."""
    
    # Try exact search
    results = vector_store.similarity_search(query, k=k)
    
    if results:
        return results
    
    # Try with query expansion
    expanded = expand_query(query, llm)
    for q in expanded:
        results = vector_store.similarity_search(q, k=k)
        if results:
            return results
    
    # Last resort: return most general results
    return vector_store.similarity_search("", k=k)
```

### Low Quality Results

```python
def quality_filtered_search(query: str, k: int = 5, min_score: float = 0.5):
    """Filter out low-quality results."""
    
    results = vector_store.similarity_search_with_score(query, k=k * 2)
    
    # Filter by score
    filtered = [(doc, score) for doc, score in results if score >= min_score]
    
    # Return top k
    return [doc for doc, _ in filtered[:k]]
```

## Related Recipes

- **Document Indexing**: Preparing documents for search
- **RAG Pipeline**: Using search results for generation
- **Creating Agents**: Agents that use search tools
