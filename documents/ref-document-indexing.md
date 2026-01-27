# Recipe: Document Indexing Pipeline

## Goal

Index documents into a vector database for RAG retrieval.

## Prerequisites

```bash
uv pip install langchain langchain-openai langchain-qdrant qdrant-client
```

## Quick Start

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# 1. Load documents
loader = TextLoader("document.txt")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

# 3. Setup vector store
client = QdrantClient(":memory:")
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = QdrantVectorStore(
    client=client,
    collection_name="documents",
    embedding=embeddings
)

# 4. Index
vector_store.add_documents(chunks)
print(f"Indexed {len(chunks)} chunks")
```

## Full Implementation

### Load Multiple File Types

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader
)
from pathlib import Path

def load_documents(path: str) -> list:
    """Load documents from file or directory."""
    
    path = Path(path)
    
    if path.is_file():
        return load_single_file(path)
    elif path.is_dir():
        return load_directory(path)
    else:
        raise ValueError(f"Path not found: {path}")

def load_single_file(path: Path) -> list:
    """Load a single file based on extension."""
    
    loaders = {
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".pdf": PyPDFLoader,
    }
    
    ext = path.suffix.lower()
    if ext not in loaders:
        raise ValueError(f"Unsupported file type: {ext}")
    
    loader = loaders[ext](str(path))
    return loader.load()

def load_directory(path: Path) -> list:
    """Load all supported files from directory."""
    
    all_docs = []
    
    for ext, loader_cls in [(".txt", TextLoader), (".md", UnstructuredMarkdownLoader), (".pdf", PyPDFLoader)]:
        loader = DirectoryLoader(
            str(path),
            glob=f"**/*{ext}",
            loader_cls=loader_cls,
            show_progress=True
        )
        docs = loader.load()
        all_docs.extend(docs)
    
    return all_docs
```

### Chunking with Metadata

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[Document]:
    """Chunk documents while preserving metadata."""
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    
    for doc in docs:
        doc_chunks = splitter.split_documents([doc])
        
        # Add chunk index to metadata
        for i, chunk in enumerate(doc_chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(doc_chunks)
        
        chunks.extend(doc_chunks)
    
    return chunks
```

### Batch Indexing with Progress

```python
from tqdm import tqdm

def index_documents(
    vector_store: QdrantVectorStore,
    chunks: list[Document],
    batch_size: int = 100
) -> int:
    """Index documents in batches with progress bar."""
    
    total_indexed = 0
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Indexing"):
        batch = chunks[i:i + batch_size]
        vector_store.add_documents(batch)
        total_indexed += len(batch)
    
    return total_indexed
```

### Complete Pipeline

```python
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

class DocumentIndexer:
    def __init__(
        self,
        collection_name: str = "documents",
        embedding_model: str = "text-embedding-3-small",
        qdrant_url: str = ":memory:"
    ):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Setup Qdrant
        if qdrant_url == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(url=qdrant_url)
        
        # Create collection if needed
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
        
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings
        )
    
    def index(
        self,
        path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 100
    ) -> dict:
        """Index documents from path."""
        
        # Load
        print(f"Loading documents from {path}...")
        docs = load_documents(path)
        print(f"Loaded {len(docs)} documents")
        
        # Chunk
        print("Chunking documents...")
        chunks = chunk_documents(docs, chunk_size, chunk_overlap)
        print(f"Created {len(chunks)} chunks")
        
        # Index
        print("Indexing chunks...")
        indexed = index_documents(self.vector_store, chunks, batch_size)
        
        return {
            "documents_loaded": len(docs),
            "chunks_created": len(chunks),
            "chunks_indexed": indexed
        }
    
    def get_retriever(self, k: int = 5):
        """Get retriever for searching."""
        return self.vector_store.as_retriever(search_kwargs={"k": k})


# Usage
indexer = DocumentIndexer(collection_name="my_docs")
stats = indexer.index("./documents")
print(stats)

retriever = indexer.get_retriever(k=5)
results = retriever.invoke("What is machine learning?")
```

## Variations

### With Deduplication

```python
import hashlib

def get_content_hash(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()

def deduplicate_chunks(chunks: list[Document]) -> list[Document]:
    """Remove duplicate chunks based on content hash."""
    seen = set()
    unique = []
    
    for chunk in chunks:
        hash_id = get_content_hash(chunk.page_content)
        if hash_id not in seen:
            seen.add(hash_id)
            chunk.metadata["content_hash"] = hash_id
            unique.append(chunk)
    
    return unique
```

### Incremental Indexing

```python
def index_if_new(vector_store, chunks: list[Document]) -> int:
    """Only index chunks that don't already exist."""
    
    new_chunks = []
    
    for chunk in chunks:
        hash_id = get_content_hash(chunk.page_content)
        
        # Check if exists
        existing = vector_store.similarity_search(
            chunk.page_content,
            k=1,
            filter={"content_hash": hash_id}
        )
        
        if not existing:
            chunk.metadata["content_hash"] = hash_id
            new_chunks.append(chunk)
    
    if new_chunks:
        vector_store.add_documents(new_chunks)
    
    return len(new_chunks)
```

### With Custom Metadata Extraction

```python
import re
from datetime import datetime

def extract_metadata(doc: Document) -> Document:
    """Extract additional metadata from document content."""
    
    content = doc.page_content
    
    # Extract title (first heading)
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if title_match:
        doc.metadata["title"] = title_match.group(1)
    
    # Extract date if present
    date_match = re.search(r'\d{4}-\d{2}-\d{2}', content)
    if date_match:
        doc.metadata["date"] = date_match.group(0)
    
    # Word count
    doc.metadata["word_count"] = len(content.split())
    
    # Index timestamp
    doc.metadata["indexed_at"] = datetime.now().isoformat()
    
    return doc
```

## Common Issues

### Memory Errors with Large Datasets

```python
# Process in streaming fashion
def index_large_dataset(indexer, file_paths: list[str]):
    for path in file_paths:
        # Process one file at a time
        indexer.index(path)
        # Memory freed after each file
```

### Slow Indexing

```python
# Increase batch size
indexer.index(path, batch_size=500)

# Use async for I/O bound operations
async def async_index(chunks, vector_store):
    await vector_store.aadd_documents(chunks)
```

### Embedding Costs

```python
# Estimate cost before indexing
def estimate_cost(chunks: list[Document]) -> float:
    total_tokens = sum(len(c.page_content.split()) * 1.3 for c in chunks)
    # text-embedding-3-small: $0.00002 per 1K tokens
    return (total_tokens / 1000) * 0.00002

cost = estimate_cost(chunks)
print(f"Estimated embedding cost: ${cost:.4f}")
```

## Related Recipes

- **Similarity Search**: Querying indexed documents
- **RAG Pipeline**: Using indexed documents for generation
