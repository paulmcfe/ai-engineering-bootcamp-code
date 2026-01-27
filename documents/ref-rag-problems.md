# Troubleshooting: RAG Problems

## Overview

Common issues with Retrieval-Augmented Generation systems and how to fix them.

---

## No Results Returned

### Symptoms
- Search returns empty results
- "No relevant information found" messages

### Causes & Solutions

**1. Documents Not Indexed**

```python
# Check if documents are in the collection
from qdrant_client import QdrantClient

client = QdrantClient(":memory:")
collection_info = client.get_collection("documents")
print(f"Points in collection: {collection_info.points_count}")
```

Fix: Re-run indexing pipeline.

**2. Query Doesn't Match Document Vocabulary**

The query uses different terms than the documents.

```python
# Try query expansion
def expand_query(query: str) -> list[str]:
    expanded = [query]
    # Add synonyms or related terms
    synonyms = get_synonyms(query)
    expanded.extend(synonyms)
    return expanded

# Search with multiple queries
all_results = []
for q in expand_query(query):
    results = vector_store.similarity_search(q, k=3)
    all_results.extend(results)
```

**3. Wrong Collection Name**

```python
# Verify collection exists
if not client.collection_exists("documents"):
    print("Collection doesn't exist!")
```

**4. Embedding Dimension Mismatch**

```python
# Check collection dimension
info = client.get_collection("documents")
expected_dim = info.config.params.vectors.size
print(f"Collection expects {expected_dim} dimensions")

# Check your embedding dimension
test_embedding = embeddings.embed_query("test")
print(f"Embedding produces {len(test_embedding)} dimensions")
```

Fix: Recreate collection with correct dimensions or use matching embedding model.

---

## Poor Quality Results

### Symptoms
- Retrieved documents aren't relevant
- Wrong information being returned
- Missing obvious matches

### Causes & Solutions

**1. Chunks Too Large or Too Small**

```python
# Too large = diluted relevance
# Too small = missing context

# Experiment with sizes
for chunk_size in [200, 500, 1000]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)
    chunks = splitter.split_documents(docs)
    
    # Test retrieval
    results = test_retrieval(chunks, test_queries)
    print(f"Chunk size {chunk_size}: {results['precision']}")
```

Recommendation: Start with 500 tokens, adjust based on testing.

**2. Wrong Distance Metric**

```python
from qdrant_client.models import Distance

# Cosine is usually best for text embeddings
# Check your collection's distance metric
info = client.get_collection("documents")
print(f"Distance metric: {info.config.params.vectors.distance}")

# Recreate with correct metric
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)
```

**3. No Overlap Between Chunks**

```python
# Without overlap, context at boundaries is lost
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50  # Add 10% overlap
)
```

**4. Missing Metadata for Filtering**

```python
# Add useful metadata during indexing
for doc in documents:
    doc.metadata["category"] = extract_category(doc)
    doc.metadata["date"] = extract_date(doc)

# Then filter during search
results = vector_store.similarity_search(
    query,
    k=5,
    filter={"category": "technical"}
)
```

---

## Hallucinations Despite RAG

### Symptoms
- Model makes up information not in retrieved docs
- Responses contradict the context provided

### Causes & Solutions

**1. Retrieved Context Not Used**

Check if context is actually being included in the prompt.

```python
# Debug: Print the full prompt
def format_prompt(query: str, context: str) -> str:
    prompt = f"""Answer based ONLY on this context:

Context:
{context}

Question: {query}

If the answer isn't in the context, say "I don't have that information."
"""
    print(f"Full prompt:\n{prompt}")  # Debug
    return prompt
```

**2. Prompt Not Strong Enough**

```python
# Weak prompt
prompt = f"Context: {context}\n\nQuestion: {query}"

# Strong prompt
prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided context.

RULES:
1. Only use information from the context below
2. If the answer isn't in the context, say "I cannot answer this based on the available information"
3. Quote relevant parts of the context to support your answer
4. Never make up information

Context:
{context}

Question: {query}

Answer:"""
```

**3. Retrieved Context Not Relevant**

Add relevance checks before using context.

```python
def is_relevant(query: str, doc: str, threshold: float = 0.5) -> bool:
    """Check if document is relevant to query."""
    # Use embedding similarity
    query_emb = embeddings.embed_query(query)
    doc_emb = embeddings.embed_query(doc)
    
    similarity = cosine_similarity(query_emb, doc_emb)
    return similarity >= threshold

# Filter results
relevant_docs = [doc for doc in results if is_relevant(query, doc.page_content)]
```

**4. Model Temperature Too High**

```python
# Lower temperature for more factual responses
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

---

## Slow Performance

### Symptoms
- Long query times
- Timeouts
- Poor user experience

### Causes & Solutions

**1. No Index on Vector Database**

```python
# For Qdrant, HNSW index is created automatically
# But check index status
info = client.get_collection("documents")
print(f"Indexed: {info.status}")

# Force index optimization
client.update_collection(
    collection_name="documents",
    optimizer_config={"indexing_threshold": 0}  # Index immediately
)
```

**2. Retrieving Too Many Documents**

```python
# Don't retrieve more than needed
results = vector_store.similarity_search(query, k=3)  # Not k=100
```

**3. No Embedding Caching**

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

# Cache embeddings to avoid re-computing
store = LocalFileStore("./embedding_cache")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embeddings,
    store,
    namespace="text-embedding-3-small"
)
```

**4. Synchronous Operations**

```python
# Use async for I/O bound operations
async def async_rag(query: str):
    # Parallel embedding and retrieval
    results = await vector_store.asimilarity_search(query, k=5)
    return results
```

---

## Context Window Overflow

### Symptoms
- "Maximum context length exceeded" errors
- Truncated responses

### Causes & Solutions

**1. Too Many Retrieved Documents**

```python
# Limit retrieved documents
results = vector_store.similarity_search(query, k=3)  # Not k=20

# Or truncate total context
def limit_context(docs: list, max_tokens: int = 3000) -> str:
    context = ""
    for doc in docs:
        if len(context) + len(doc.page_content) > max_tokens * 4:  # Rough char estimate
            break
        context += doc.page_content + "\n\n"
    return context
```

**2. Chunks Too Large**

```python
# Use smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # Smaller chunks
    chunk_overlap=30
)
```

**3. Use Contextual Compression**

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# Returns only relevant excerpts
compressed_docs = compression_retriever.invoke(query)
```

---

## Inconsistent Results

### Symptoms
- Same query returns different results
- Results vary between runs

### Causes & Solutions

**1. Non-Deterministic Embeddings**

Most embedding models are deterministic, but check:

```python
# Test embedding consistency
emb1 = embeddings.embed_query("test query")
emb2 = embeddings.embed_query("test query")
assert emb1 == emb2, "Embeddings are not deterministic!"
```

**2. Database Not Persisted**

```python
# Using in-memory mode?
client = QdrantClient(":memory:")  # Data lost on restart!

# Use persistent storage
client = QdrantClient(path="./qdrant_data")  # Local persistence
# Or
client = QdrantClient(url="http://localhost:6333")  # Server mode
```

**3. LLM Temperature > 0**

```python
# For consistent outputs
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

---

## Debugging Checklist

1. **Verify indexing**
   - [ ] Documents loaded correctly
   - [ ] Chunks created with appropriate size
   - [ ] Embeddings generated
   - [ ] Collection has expected point count

2. **Verify retrieval**
   - [ ] Query produces embedding
   - [ ] Similarity search returns results
   - [ ] Results are relevant to query
   - [ ] Metadata filtering works

3. **Verify generation**
   - [ ] Context is included in prompt
   - [ ] Prompt instructs model to use context
   - [ ] Response references context appropriately

4. **Check configurations**
   - [ ] Embedding model matches indexed model
   - [ ] Collection dimensions are correct
   - [ ] Distance metric is appropriate (COSINE for text)

---

## Quick Diagnostics

```python
def diagnose_rag(vector_store, query: str):
    """Run diagnostic checks on RAG system."""
    
    print("=== RAG Diagnostics ===\n")
    
    # 1. Check collection
    print("1. Collection Status:")
    info = vector_store.client.get_collection(vector_store.collection_name)
    print(f"   Points: {info.points_count}")
    print(f"   Status: {info.status}")
    
    # 2. Test embedding
    print("\n2. Embedding Test:")
    emb = vector_store.embedding.embed_query(query)
    print(f"   Dimension: {len(emb)}")
    print(f"   Sample values: {emb[:3]}")
    
    # 3. Test retrieval
    print("\n3. Retrieval Test:")
    results = vector_store.similarity_search_with_score(query, k=3)
    for i, (doc, score) in enumerate(results):
        print(f"   [{i+1}] Score: {score:.4f}")
        print(f"       Content: {doc.page_content[:100]}...")
    
    # 4. Check for issues
    print("\n4. Potential Issues:")
    if info.points_count == 0:
        print("   ⚠️  Collection is empty - index documents first")
    if not results:
        print("   ⚠️  No results returned - check query and documents")
    elif results[0][1] < 0.5:
        print("   ⚠️  Low similarity scores - documents may not match query")
    else:
        print("   ✓ No obvious issues detected")

# Usage
diagnose_rag(vector_store, "What is machine learning?")
```

## Related Guides

- **Agent Debugging**: When agents don't use RAG correctly
- **Performance Optimization**: Speeding up RAG pipelines
