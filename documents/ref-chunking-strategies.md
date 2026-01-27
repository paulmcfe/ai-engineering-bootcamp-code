# Chunking Strategies

## Why Chunking Matters

You can't embed an entire book as a single vector. Embeddings work best on focused pieces of text that express coherent ideas. A whole document contains too many diverse topics—the embedding becomes a muddy average that doesn't represent any particular part well.

Chunking splits documents into pieces suitable for embedding and retrieval. Get it right and your RAG system returns precisely relevant content. Get it wrong and you'll retrieve chunks that cut off mid-thought or miss crucial context.

## The Chunking Trade-off

**Smaller chunks** provide more granular retrieval. You get exactly the paragraph that answers the question, not the whole section. But smaller chunks lose context—a sentence might not make sense without surrounding sentences.

**Larger chunks** preserve more context. You get complete thoughts and full explanations. But larger chunks are less precise—you might retrieve 1000 tokens when only 100 are relevant, wasting context window space.

Most applications use chunks between 200-1000 tokens, with 300-500 being a common sweet spot.

## Fixed-Size Chunking

The simplest approach: split text into chunks of N characters or tokens, with optional overlap.

```python
from langchain_text_splitters import CharacterTextSplitter, TokenTextSplitter

# Character-based
splitter = CharacterTextSplitter(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200,    # Overlap between chunks
    separator="\n"        # Prefer splitting at newlines
)

# Token-based (more precise for LLM context limits)
splitter = TokenTextSplitter(
    chunk_size=500,       # Tokens per chunk
    chunk_overlap=50      # Token overlap
)

chunks = splitter.split_text(document_text)
```

**Pros:** Simple, predictable chunk sizes, easy to implement.

**Cons:** Ignores document structure. May split mid-sentence or mid-paragraph.

## Recursive Character Splitting

Tries to split on natural boundaries (paragraphs, sentences) while staying within size limits. Falls back to smaller separators if chunks are still too large.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=[
        "\n\n",    # First try paragraph breaks
        "\n",      # Then line breaks
        ". ",      # Then sentences
        ", ",      # Then clauses
        " ",       # Then words
        ""         # Finally characters
    ]
)

chunks = splitter.split_text(document_text)
```

This is the most commonly used splitter because it balances simplicity with reasonable boundary detection.

## Semantic Chunking

Splits based on meaning rather than character count. Uses embeddings to detect topic shifts.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95
)

chunks = splitter.split_text(document_text)
```

The algorithm embeds each sentence, then identifies points where embedding similarity drops significantly—indicating a topic change. Chunks contain coherent topics regardless of length.

**Pros:** Produces semantically coherent chunks. Topics stay together.

**Cons:** Chunk sizes vary. Requires embedding calls during chunking. More complex.

## Structure-Aware Chunking

Uses document structure (headers, sections) to determine chunk boundaries.

### Markdown

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
)

chunks = splitter.split_text(markdown_text)
# Each chunk includes header hierarchy in metadata
```

### HTML

```python
from langchain_text_splitters import HTMLHeaderTextSplitter

splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[
        ("h1", "Header 1"),
        ("h2", "Header 2"),
    ]
)

chunks = splitter.split_text(html_text)
```

**Pros:** Respects document organization. Headers provide context.

**Cons:** Only works for structured documents. Sections may still be too large.

## Overlap Strategies

Overlap ensures context isn't lost at chunk boundaries. If a concept spans two paragraphs, overlap lets both chunks contain the transition.

```python
# Standard overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50  # 10% overlap
)
```

**Rule of thumb:** 10-20% overlap. Less and you risk losing context. More and you're duplicating too much content, increasing storage and potentially retrieving redundant chunks.

### Sentence-Boundary Overlap

A more sophisticated approach overlaps complete sentences rather than arbitrary characters:

```python
def chunk_with_sentence_overlap(text: str, chunk_size: int, overlap_sentences: int = 2):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence)
        current_length += len(sentence)
        
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            # Keep last N sentences for overlap
            current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences else []
            current_length = sum(len(s) for s in current_chunk)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
```

## Chunking for Specific Content Types

### Code

Code requires special handling—you don't want to split a function in half.

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language
)

# Python-aware splitting
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=100
)

# Splits on class/function boundaries when possible
chunks = splitter.split_text(python_code)
```

Supported languages include Python, JavaScript, TypeScript, Go, Rust, Java, C++, and more.

### Tables

Tables are tricky—splitting rows loses column context. Options:

1. Keep tables as single chunks (if small enough)
2. Convert to text description before chunking
3. Store table metadata separately from content

### PDFs

PDFs often have complex layouts. Consider:

1. Using layout-aware extraction (PyMuPDF, Unstructured)
2. Preserving page boundaries in metadata
3. Handling headers/footers specially

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document.pdf")
pages = loader.load()  # Each page is a document with page number metadata

# Then chunk pages individually
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(pages)
```

## Metadata Enrichment

Always attach metadata to chunks for filtering and attribution:

```python
from langchain.schema import Document

def chunk_with_metadata(text: str, source: str, splitter):
    chunks = splitter.split_text(text)
    
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                # Add more: date, author, category, etc.
            }
        )
        documents.append(doc)
    
    return documents
```

Useful metadata fields:
- **source**: Original file/URL
- **chunk_index**: Position in document
- **page_number**: For PDFs
- **section_header**: From structure-aware splitting
- **date**: Document date or last modified
- **category/type**: Document classification

## Chunk Size Selection

Factors to consider:

1. **Embedding model context limit**: Most models handle 512-8192 tokens
2. **Retrieval granularity**: Smaller = more precise, larger = more context
3. **LLM context budget**: How many chunks can you fit in the prompt?
4. **Content type**: Technical docs might need larger chunks; Q&A might need smaller

### Testing Chunk Sizes

```python
def test_chunk_sizes(documents, queries, sizes=[200, 500, 1000]):
    """Compare retrieval quality across chunk sizes."""
    from langchain_openai import OpenAIEmbeddings
    from langchain_qdrant import QdrantVectorStore
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    results = {}
    
    for size in sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, 
            chunk_overlap=size // 10
        )
        chunks = splitter.split_documents(documents)
        
        # Create temporary vector store
        store = QdrantVectorStore.from_documents(
            chunks, embeddings, location=":memory:"
        )
        
        # Test retrieval
        for query in queries:
            retrieved = store.similarity_search(query, k=3)
            # Evaluate relevance (manual or automated)
            print(f"Size {size}, Query: {query[:50]}...")
            for doc in retrieved:
                print(f"  - {doc.page_content[:100]}...")
    
    return results
```

## Best Practices

1. **Start with RecursiveCharacterTextSplitter** at 500 tokens with 50 token overlap. Adjust based on results.

2. **Preserve context** with overlap. Don't let important ideas get cut off at chunk boundaries.

3. **Include metadata** for every chunk. You'll need it for filtering and citations.

4. **Match chunking to content**. Code needs language-aware splitting. Structured docs need structure-aware splitting.

5. **Test with real queries**. The only way to know if your chunking works is to try retrieving with actual user queries.

6. **Consider your retrieval k**. If you retrieve 5 chunks, each can be larger than if you retrieve 20.

## Related Concepts

- **Embeddings**: Chunks get embedded for similarity search
- **RAG**: Chunking is part of the indexing pipeline
- **Vector Databases**: Store chunked and embedded content
- **Retrieval**: Chunks are what get retrieved
