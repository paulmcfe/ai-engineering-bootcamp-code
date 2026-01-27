# Retrieval Patterns

## Overview

Retrieval patterns determine how you find and fetch relevant information for RAG systems. The right pattern depends on your data, queries, and quality requirements. This guide covers common patterns from simple to advanced.

## Basic Patterns

### Simple Similarity Search

The most straightforward pattern: embed query, find nearest vectors.

```python
def simple_search(query: str, k: int = 5) -> list:
    """Basic vector similarity search."""
    results = vector_store.similarity_search(query, k=k)
    return results
```

**Pros:** Fast, simple, works well for focused queries.
**Cons:** May miss relevant content with different wording.

### Similarity with Score Threshold

Filter results by minimum similarity score:

```python
def threshold_search(query: str, threshold: float = 0.7, k: int = 10) -> list:
    """Search with minimum similarity threshold."""
    results = vector_store.similarity_search_with_score(query, k=k)
    
    # Filter by threshold (lower distance = more similar for some metrics)
    filtered = [(doc, score) for doc, score in results if score >= threshold]
    
    return [doc for doc, _ in filtered]
```

**Use when:** You'd rather return nothing than irrelevant results.

### Metadata Filtering

Narrow search using document metadata:

```python
def filtered_search(query: str, filters: dict, k: int = 5) -> list:
    """Search with metadata filters applied."""
    
    # Qdrant filter example
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    
    filter_conditions = []
    for key, value in filters.items():
        filter_conditions.append(
            FieldCondition(key=key, match=MatchValue(value=value))
        )
    
    results = vector_store.similarity_search(
        query,
        k=k,
        filter=Filter(must=filter_conditions)
    )
    
    return results

# Usage
results = filtered_search(
    "machine learning basics",
    filters={"category": "tutorials", "difficulty": "beginner"}
)
```

**Use when:** You have structured metadata to constrain search.

## Hybrid Patterns

### Hybrid Search (Dense + Sparse)

Combine semantic search with keyword matching:

```python
def hybrid_search(query: str, k: int = 5, alpha: float = 0.5) -> list:
    """Combine dense vector and sparse BM25 search."""
    
    # Dense (semantic) search
    dense_results = vector_store.similarity_search_with_score(query, k=k*2)
    
    # Sparse (BM25) search
    sparse_results = bm25_index.search(query, k=k*2)
    
    # Combine with reciprocal rank fusion
    combined = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        weights=[alpha, 1-alpha]
    )
    
    return combined[:k]

def reciprocal_rank_fusion(result_lists: list, weights: list, k: int = 60) -> list:
    """Combine multiple ranked lists using RRF."""
    scores = {}
    
    for results, weight in zip(result_lists, weights):
        for rank, (doc, _) in enumerate(results):
            doc_id = doc.metadata.get("id", hash(doc.page_content))
            if doc_id not in scores:
                scores[doc_id] = {"doc": doc, "score": 0}
            scores[doc_id]["score"] += weight * (1 / (k + rank + 1))
    
    sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_results]
```

**Use when:** Queries mix conceptual questions with specific terms.

### Multi-Query Retrieval

Generate multiple queries to improve coverage:

```python
def multi_query_search(query: str, k: int = 5) -> list:
    """Generate multiple query variations and combine results."""
    
    # Generate query variations
    variations_prompt = f"""Generate 3 different ways to search for information about:
"{query}"

Return as JSON array of strings."""
    
    response = llm.invoke(variations_prompt)
    queries = [query] + json.loads(response)  # Include original
    
    # Search with each variation
    all_results = []
    for q in queries:
        results = vector_store.similarity_search(q, k=k)
        all_results.extend(results)
    
    # Deduplicate and rank
    return deduplicate_and_rank(all_results, query)

def deduplicate_and_rank(results: list, original_query: str) -> list:
    """Remove duplicates, rank by relevance to original query."""
    seen = set()
    unique = []
    
    for doc in results:
        content_hash = hash(doc.page_content)
        if content_hash not in seen:
            seen.add(content_hash)
            unique.append(doc)
    
    # Re-rank by similarity to original query
    query_embedding = embeddings.embed_query(original_query)
    scored = []
    for doc in unique:
        doc_embedding = embeddings.embed_query(doc.page_content)
        score = cosine_similarity(query_embedding, doc_embedding)
        scored.append((doc, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored]
```

**Use when:** Queries might be phrased differently than document content.

## Advanced Patterns

### Reranking

Use a cross-encoder to rerank initial results:

```python
def search_with_reranking(query: str, k: int = 5, initial_k: int = 20) -> list:
    """Retrieve broadly, then rerank with cross-encoder."""
    
    # Initial broad retrieval
    candidates = vector_store.similarity_search(query, k=initial_k)
    
    # Rerank with Cohere
    import cohere
    co = cohere.Client()
    
    rerank_response = co.rerank(
        query=query,
        documents=[doc.page_content for doc in candidates],
        top_n=k,
        model="rerank-english-v3.0"
    )
    
    # Return reranked results
    reranked = []
    for result in rerank_response.results:
        reranked.append(candidates[result.index])
    
    return reranked
```

**Use when:** Precision matters more than latency.

### RAG Fusion

Multiple queries with reciprocal rank fusion:

```python
def rag_fusion(query: str, k: int = 5, num_queries: int = 4) -> list:
    """RAG-Fusion: multi-query with RRF combination."""
    
    # Generate diverse queries
    prompt = f"""Generate {num_queries} different search queries to find information for:
"{query}"

Make queries diverse - different angles, synonyms, related concepts.
Return as JSON array."""
    
    queries = json.loads(llm.invoke(prompt))
    
    # Search with each
    all_ranked_results = []
    for q in queries:
        results = vector_store.similarity_search_with_score(q, k=k*2)
        all_ranked_results.append(results)
    
    # RRF combination
    return reciprocal_rank_fusion(all_ranked_results, weights=[1]*len(queries))[:k]
```

**Use when:** Comprehensive coverage is important.

### Parent Document Retrieval

Retrieve chunks but return larger parent documents:

```python
def parent_document_search(query: str, k: int = 3) -> list:
    """Retrieve by chunk, return parent document."""
    
    # Search at chunk level
    chunks = vector_store.similarity_search(query, k=k*3)
    
    # Get unique parent documents
    parent_ids = set()
    parents = []
    
    for chunk in chunks:
        parent_id = chunk.metadata.get("parent_id")
        if parent_id and parent_id not in parent_ids:
            parent_ids.add(parent_id)
            parent_doc = document_store.get(parent_id)
            parents.append(parent_doc)
            
            if len(parents) >= k:
                break
    
    return parents
```

**Use when:** Context around the matching chunk is important.

### Self-Query Retrieval

Convert natural language to structured queries:

```python
def self_query_search(query: str) -> list:
    """Parse natural language into structured query."""
    
    # Define available filters
    metadata_schema = """
Available filters:
- category: string (e.g., "tutorial", "reference", "blog")
- date: date (e.g., "2024-01-01")
- author: string
- difficulty: string ("beginner", "intermediate", "advanced")
"""
    
    # LLM extracts filters
    extraction_prompt = f"""Given this query, extract search parameters:

Query: "{query}"

{metadata_schema}

Return JSON with:
- search_query: the semantic search portion
- filters: dict of metadata filters to apply

Example: "beginner Python tutorials from 2024"
→ {{"search_query": "Python tutorials", "filters": {{"difficulty": "beginner", "date": {{"gte": "2024-01-01"}}}}}}
"""
    
    parsed = json.loads(llm.invoke(extraction_prompt))
    
    return filtered_search(
        parsed["search_query"],
        filters=parsed["filters"]
    )
```

**Use when:** Users express constraints in natural language.

### Contextual Compression

Compress retrieved content to most relevant parts:

```python
def compressed_search(query: str, k: int = 5) -> list:
    """Retrieve then compress to relevant excerpts."""
    
    # Initial retrieval
    docs = vector_store.similarity_search(query, k=k)
    
    # Compress each document
    compressed = []
    for doc in docs:
        compression_prompt = f"""Extract only the parts of this document relevant to the query.

Query: {query}

Document:
{doc.page_content}

Return only the relevant excerpts, preserving important context."""
        
        excerpt = llm.invoke(compression_prompt)
        
        compressed.append(Document(
            page_content=excerpt,
            metadata={**doc.metadata, "original_length": len(doc.page_content)}
        ))
    
    return compressed
```

**Use when:** Retrieved documents are long and partially relevant.

### Hierarchical Retrieval

Two-stage retrieval: summaries first, then details:

```python
def hierarchical_search(query: str, k: int = 3) -> list:
    """Search summaries first, then retrieve full documents."""
    
    # Stage 1: Search document summaries
    summary_results = summary_store.similarity_search(query, k=k*2)
    
    # Stage 2: Get full documents for top matches
    full_docs = []
    for summary in summary_results[:k]:
        doc_id = summary.metadata["document_id"]
        full_doc = document_store.get(doc_id)
        full_docs.append(full_doc)
    
    return full_docs
```

**Use when:** Documents are long and you want efficient initial filtering.

## Pattern Selection Guide

| Pattern | Best For | Trade-offs |
|---------|----------|------------|
| Simple similarity | Focused queries, fast retrieval | May miss alternate phrasings |
| Threshold filtering | High-precision needs | May return fewer results |
| Metadata filtering | Structured constraints | Requires good metadata |
| Hybrid search | Mixed semantic/keyword needs | More complex, slower |
| Multi-query | Query expansion | More LLM calls |
| Reranking | Precision-critical | Adds latency |
| RAG Fusion | Comprehensive coverage | Higher cost |
| Parent document | Context-dependent content | Returns more text |
| Self-query | Natural language constraints | Parsing can fail |
| Compression | Long documents | Adds LLM calls |
| Hierarchical | Large document collections | Requires summary index |

## Combining Patterns

Patterns can be layered:

```python
def production_retrieval(query: str, k: int = 5) -> list:
    """Production retrieval pipeline combining patterns."""
    
    # 1. Parse query for filters
    parsed = extract_filters(query)
    
    # 2. Multi-query with filters
    results = multi_query_filtered_search(
        parsed["search_query"],
        parsed["filters"],
        k=k*4
    )
    
    # 3. Rerank
    reranked = rerank_with_cohere(query, results, k=k*2)
    
    # 4. Compress
    compressed = compress_to_relevant(query, reranked[:k])
    
    return compressed
```

## Best Practices

1. **Start simple.** Add complexity only when needed.

2. **Measure everything.** Track precision, recall, and latency for each pattern.

3. **Match pattern to query type.** Different queries may need different patterns.

4. **Cache expensive operations.** Reranking and compression results can be cached.

5. **Set reasonable limits.** Don't retrieve 100 documents to find 5.

6. **Log retrieval decisions.** Understanding what was retrieved helps debugging.

## Related Concepts

- **RAG Fundamentals**: Basic retrieval-augmented generation
- **Embeddings**: Vector representations for similarity search
- **Vector Databases**: Storage and search infrastructure
- **Evaluation**: Measuring retrieval quality
