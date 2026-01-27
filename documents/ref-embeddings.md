# Embeddings

## What Are Embeddings?

An embedding is a numerical representation of text as a vector—a list of floating-point numbers that captures semantic meaning. When you embed the sentence "The cat sat on the mat," you get back something like:

```
[0.23, -0.45, 0.67, -0.12, ...]
```

These vectors typically have hundreds or thousands of dimensions. Each dimension captures some aspect of meaning, though individual dimensions aren't human-interpretable.

## Why Embeddings Matter

The key property of embeddings is that similar meanings produce similar vectors. "The cat sat on the mat" and "A feline rested on a rug" will have vectors that are close together in vector space, even though they share no words. This enables semantic search—finding content by meaning rather than keyword matching.

Traditional keyword search fails when users and documents use different terminology. A search for "car insurance" won't find documents about "automobile coverage policies." Embeddings solve this by operating in semantic space where synonyms and related concepts naturally cluster together.

## Common Embedding Models (2025)

| Model | Dimensions | Provider | Best For |
|-------|------------|----------|----------|
| text-embedding-3-small | 1536 | OpenAI | General purpose, cost-effective |
| text-embedding-3-large | 3072 | OpenAI | Higher accuracy, more expensive |
| embed-english-v3.0 | 1024 | Cohere | Multilingual support |
| all-MiniLM-L6-v2 | 384 | HuggingFace | Self-hosted, fast |
| bge-large-en-v1.5 | 1024 | BAAI | Self-hosted, high quality |
| nomic-embed-text-v1.5 | 768 | Nomic | Open source, good quality |

## Key Considerations

**Dimension size** affects both quality and cost. More dimensions capture more nuance but require more storage and compute. For most applications, 1536 dimensions (OpenAI small) is sufficient.

**Model consistency** is critical. You must use the same embedding model for indexing and querying. Vectors from different models exist in different vector spaces and cannot be compared meaningfully. Switching models requires re-embedding your entire corpus.

**Normalization** affects distance calculations. Most embedding models produce normalized vectors (length 1), which means cosine similarity and dot product are equivalent. Check your model's documentation.

**Context length** limits how much text you can embed at once. Most models handle 512-8192 tokens. Longer documents must be chunked before embedding.

## Distance Metrics

Three metrics are commonly used for comparing embeddings:

**Cosine similarity** measures the angle between vectors, ranging from -1 to 1. Most common for text embeddings because it captures directional similarity regardless of magnitude. A score of 1 means identical direction, 0 means perpendicular, -1 means opposite directions.

**Euclidean distance** measures straight-line distance between points. Sensitive to both direction and magnitude. Smaller values indicate more similarity.

**Dot product** combines direction and magnitude. Equivalent to cosine similarity for normalized vectors, but faster to compute since it skips the normalization step.

For text embeddings, cosine similarity is the standard choice. If your model produces normalized vectors, use dot product for better performance.

## Code Example (OpenAI)

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Generate an embedding for a piece of text."""
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

# Embed a single query
query_vector = get_embedding("How do transformers work?")
print(f"Vector has {len(query_vector)} dimensions")

# Embed multiple texts in batch (more efficient)
texts = [
    "Machine learning fundamentals",
    "Neural network architectures", 
    "Training deep learning models"
]
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)
vectors = [item.embedding for item in response.data]
```

## Code Example (LangChain)

```python
from langchain_openai import OpenAIEmbeddings

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Embed a single text
vector = embeddings.embed_query("How do transformers work?")

# Embed multiple documents
texts = ["First document", "Second document", "Third document"]
vectors = embeddings.embed_documents(texts)
```

## Computing Similarity

```python
import numpy as np

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def dot_product(vec1: list[float], vec2: list[float]) -> float:
    """Compute dot product (use for normalized vectors)."""
    return np.dot(vec1, vec2)

# Example usage
query = get_embedding("What is machine learning?")
doc1 = get_embedding("Machine learning is a subset of AI")
doc2 = get_embedding("The weather is nice today")

print(f"Query vs Doc1: {cosine_similarity(query, doc1):.3f}")  # High similarity
print(f"Query vs Doc2: {cosine_similarity(query, doc2):.3f}")  # Low similarity
```

## Best Practices

1. **Batch your embedding calls** when processing multiple texts. API calls have overhead, and batching is more efficient.

2. **Cache embeddings** for documents that don't change. Re-embedding the same content wastes API calls and money.

3. **Preprocess text** before embedding. Remove excessive whitespace, normalize unicode, and consider lowercasing for consistency.

4. **Monitor costs** especially with large corpora. Embedding millions of documents can get expensive with API-based models.

5. **Consider open-source models** for high-volume or privacy-sensitive applications. Self-hosted models have no per-token costs after initial setup.

## Related Concepts

- **Vector Databases**: Where embeddings are stored and searched
- **Similarity Search**: Finding nearest neighbors in vector space
- **RAG**: Using embeddings to retrieve relevant context for LLMs
- **Chunking**: Splitting documents for optimal embedding
