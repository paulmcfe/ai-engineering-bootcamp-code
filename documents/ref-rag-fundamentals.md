# RAG Fundamentals

## What Is RAG?

RAG (Retrieval-Augmented Generation) is a pattern for giving LLMs access to external knowledge at query time. Instead of relying solely on information baked into model weights during training, RAG retrieves relevant documents and includes them in the prompt, grounding the model's response in specific source material.

The name comes from the 2020 paper by Lewis et al. that formalized this approach. The core insight: separate knowledge storage (in a retrieval system) from knowledge use (in a language model). This lets you update knowledge without retraining, provide sources for answers, and work with proprietary data the model was never trained on.

## Why RAG Matters

LLMs have a fundamental limitation: their knowledge is frozen at training time. GPT-5's training data has a cutoff date. It doesn't know about events after that date, doesn't have access to your company's internal documents, and can't reference the latest research papers.

RAG solves this by injecting relevant context at inference time. Ask about your company's Q3 earnings, and the system retrieves the earnings report and includes it in the prompt. The model generates a response based on that specific document, not vague training data.

RAG also provides attribution. When the model's response comes from retrieved documents, you can cite sources. This builds trust and lets users verify information—critical for enterprise and research applications.

## The RAG Pipeline

RAG has two phases: indexing (offline) and querying (online).

### Indexing Phase

```
Documents → Load → Chunk → Embed → Store
```

1. **Load**: Read documents from files, databases, APIs, or web pages
2. **Chunk**: Split documents into smaller pieces (typically 200-1000 tokens)
3. **Embed**: Convert each chunk to a vector using an embedding model
4. **Store**: Save vectors and text in a vector database

This happens once per document (or when documents update). It's an offline process that prepares your knowledge base for search.

### Query Phase

```
Query → Embed → Search → Retrieve → Generate
```

1. **Embed**: Convert user's question to a vector
2. **Search**: Find most similar vectors in the database
3. **Retrieve**: Pull the associated text chunks
4. **Generate**: Include chunks in prompt, generate response

This happens on every user query. Latency matters here—users are waiting.

## Basic Implementation

```python
from openai import OpenAI
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

client = OpenAI()

# Setup vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant = QdrantClient(":memory:")
vector_store = QdrantVectorStore(
    client=qdrant,
    collection_name="docs",
    embedding=embeddings
)

# Indexing
def index_document(text: str, source: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    
    from langchain.schema import Document
    docs = [Document(page_content=chunk, metadata={"source": source}) 
            for chunk in chunks]
    vector_store.add_documents(docs)

# Querying
def answer_question(question: str) -> str:
    # Retrieve relevant chunks
    results = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in results])
    
    # Generate response
    response = client.responses.create(
        model="gpt-4o-mini",
        instructions="Answer based only on the provided context.",
        input=f"Context:\n{context}\n\nQuestion: {question}"
    )
    return response.output_text
```

## Key Components

### Document Loaders

Load content from various sources:

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader
)

# Text files
loader = TextLoader("document.txt")

# PDFs
loader = PyPDFLoader("document.pdf")

# Web pages
loader = WebBaseLoader("https://example.com/page")

# Directory of files
loader = DirectoryLoader("./docs", glob="**/*.md")

documents = loader.load()
```

### Text Splitters

Break documents into chunks:

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter
)

# Character-based (most common)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Token-based (matches model limits)
splitter = TokenTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Structure-aware (for markdown)
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2")]
)

chunks = splitter.split_documents(documents)
```

### Retrievers

Interface for searching:

```python
# Basic retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke("search query")

# With filtering
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
        "filter": {"category": "technical"}
    }
)

# MMR (Maximum Marginal Relevance) for diversity
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)
```

## RAG Prompt Patterns

### Basic Context Injection

```python
prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""
```

### With Citation Instructions

```python
prompt = f"""Answer the question using the provided sources. 
Cite sources using [Source: filename] format.

Sources:
{formatted_sources}

Question: {question}

Answer with citations:"""
```

### With Confidence Indication

```python
prompt = f"""Answer based on the context. If the context doesn't 
contain enough information, say "I don't have enough information 
to answer this fully" and explain what's missing.

Context:
{context}

Question: {question}"""
```

## Common Challenges

### Retrieval Quality

The most common RAG failure mode: retrieved chunks aren't relevant to the question. Causes include:
- Poor chunking (splitting mid-thought)
- Wrong embedding model for your domain
- Query-document vocabulary mismatch
- Insufficient context in chunks

Solutions: better chunking strategies, query expansion, hybrid search (vector + keyword), reranking.

### Context Window Management

Retrieved chunks compete for limited context space. Too many chunks dilute relevance; too few might miss important information.

Solutions: careful chunk sizing, reranking to prioritize best chunks, summarization of retrieved content.

### Hallucination Despite Context

Model ignores retrieved context and generates from training data, producing plausible but unsourced information.

Solutions: stronger prompting ("only use provided context"), lower temperature, explicit citation requirements, verification steps.

### Stale or Inconsistent Data

Index contains outdated information, or different chunks contradict each other.

Solutions: timestamp filtering, source prioritization, conflict detection in prompts.

## Evaluation Metrics

Common metrics for RAG systems:

- **Retrieval Precision**: What fraction of retrieved docs are relevant?
- **Retrieval Recall**: What fraction of relevant docs were retrieved?
- **Answer Faithfulness**: Is the answer grounded in retrieved context?
- **Answer Relevance**: Does the answer address the question?
- **Context Relevance**: Is retrieved context actually useful?

See the RAGAS framework for automated evaluation of these metrics.

## When to Use RAG

**Good fit:**
- Question answering over documents
- Customer support with knowledge bases
- Research assistance
- Internal documentation search
- Any task requiring specific, updateable knowledge

**Poor fit:**
- Tasks requiring reasoning without external knowledge
- Creative generation where grounding isn't needed
- Simple classification or extraction
- Real-time data (use tools/APIs instead)

## Related Concepts

- **Embeddings**: How text becomes searchable vectors
- **Vector Databases**: Where embeddings are stored
- **Chunking**: How documents are split for indexing
- **Agentic RAG**: Adding reasoning to retrieval decisions
- **Evaluation**: Measuring RAG system quality
