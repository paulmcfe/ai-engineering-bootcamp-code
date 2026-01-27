# LangChain Quick Reference

## Overview

LangChain is a framework for building applications with large language models. Version 1.0 (October 2025) introduced a cleaner API with `create_agent`, improved component architecture, and better integrations.

## Installation

```bash
# Core packages
uv pip install langchain langchain-core

# Provider integrations
uv pip install langchain-openai      # OpenAI models
uv pip install langchain-anthropic   # Anthropic/Claude models
uv pip install langchain-cohere      # Cohere models

# Vector store integrations
uv pip install langchain-qdrant      # Qdrant
uv pip install langchain-pinecone    # Pinecone
uv pip install langchain-chroma      # Chroma

# Community integrations
uv pip install langchain-community   # Document loaders, etc.
```

## Chat Models

### OpenAI

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=1000
)

# Simple invocation
response = llm.invoke("What is RAG?")
print(response.content)

# With messages
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Explain embeddings in one paragraph.")
]
response = llm.invoke(messages)
```

### Anthropic

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0,
    max_tokens=1000
)

response = llm.invoke("What is RAG?")
```

### Streaming

```python
for chunk in llm.stream("Explain machine learning"):
    print(chunk.content, end="", flush=True)
```

## Embeddings

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Single text
vector = embeddings.embed_query("What is machine learning?")
print(f"Dimensions: {len(vector)}")

# Multiple documents
texts = ["First document", "Second document"]
vectors = embeddings.embed_documents(texts)
```

## Agents (LangChain 1.0)

### Creating Agents

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def search_documents(query: str) -> str:
    """Search the knowledge base for information."""
    results = vector_store.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in results])

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

# Create agent
agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_documents, calculator],
    system_prompt="You are a helpful assistant. Use tools when needed."
)

# Run agent
response = agent.invoke({
    "messages": [{"role": "user", "content": "What is 15% of 240?"}]
})

# Extract final answer
final_message = response["messages"][-1]
print(final_message.content)
```

### Agent with Memory

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_documents],
    system_prompt="You are a helpful assistant.",
    memory=memory
)
```

## Tools

### Defining Tools

```python
from langchain.tools import tool
from pydantic import BaseModel, Field

# Simple tool
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    # Implementation
    return f"Weather in {city}: 72°F, sunny"

# Tool with complex parameters
class SearchParams(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=5, description="Maximum results")
    filter_type: str = Field(default=None, description="Optional filter")

@tool(args_schema=SearchParams)
def advanced_search(query: str, max_results: int = 5, filter_type: str = None) -> str:
    """Search with advanced options."""
    # Implementation
    pass
```

### Built-in Tools

```python
from langchain_community.tools import (
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    PythonREPLTool
)

# Web search
search = DuckDuckGoSearchRun()
result = search.invoke("LangChain latest features")

# Wikipedia
from langchain_community.utilities import WikipediaAPIWrapper
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Python REPL
python_repl = PythonREPLTool()
```

## Document Loaders

```python
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    WebBaseLoader,
    DirectoryLoader,
    CSVLoader
)

# Text files
loader = TextLoader("document.txt")
docs = loader.load()

# PDFs
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# Web pages
loader = WebBaseLoader("https://example.com")
docs = loader.load()

# Directory of files
loader = DirectoryLoader(
    "./documents",
    glob="**/*.md",
    loader_cls=TextLoader
)
docs = loader.load()

# CSV
loader = CSVLoader("data.csv")
docs = loader.load()
```

## Text Splitters

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter
)

# Recursive character splitting (most common)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(docs)

# Token-based splitting
splitter = TokenTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Markdown-aware splitting
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
)
```

## Vector Stores

### Qdrant

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# In-memory
client = QdrantClient(":memory:")
vector_store = QdrantVectorStore(
    client=client,
    collection_name="my_collection",
    embedding=embeddings
)

# Add documents
vector_store.add_documents(docs)

# Search
results = vector_store.similarity_search("query", k=5)

# Search with scores
results = vector_store.similarity_search_with_score("query", k=5)

# As retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
```

### Chroma

```python
from langchain_chroma import Chroma

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

## Retrievers

```python
# Basic retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# With MMR (diversity)
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

# Use retriever
docs = retriever.invoke("What is machine learning?")
```

## Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Simple template
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specializing in {domain}."),
    ("user", "{question}")
])

prompt = template.format(domain="AI", question="What is RAG?")

# With message history
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])
```

## Output Parsers

```python
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser
)
from pydantic import BaseModel

# String output
parser = StrOutputParser()
result = parser.invoke(llm_response)

# JSON output
class Answer(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

parser = JsonOutputParser(pydantic_object=Answer)
```

## Chains (LCEL)

```python
from langchain_core.runnables import RunnablePassthrough

# Simple chain
chain = template | llm | StrOutputParser()
result = chain.invoke({"domain": "AI", "question": "What is RAG?"})

# RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | template
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("What are embeddings?")
```

## Callbacks and Tracing

```python
from langchain_core.callbacks import BaseCallbackHandler

class LoggingHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM started with {len(prompts)} prompts")
    
    def on_llm_end(self, response, **kwargs):
        print(f"LLM finished")

# Use callback
response = llm.invoke("Hello", config={"callbacks": [LoggingHandler()]})

# LangSmith tracing (automatic if LANGSMITH_API_KEY set)
import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-key"
```

## Common Patterns

### RAG Pipeline

```python
# 1. Load and split
loader = DirectoryLoader("./docs", glob="**/*.md")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 2. Index
vector_store = QdrantVectorStore.from_documents(
    chunks, embeddings, location=":memory:"
)

# 3. Retrieve and generate
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

template = ChatPromptTemplate.from_template("""
Answer based on the context:

Context: {context}

Question: {question}
""")

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | template
    | llm
    | StrOutputParser()
)

answer = chain.invoke("What is RAG?")
```

### Conversational Agent

```python
from langchain.agents import create_agent
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=10, return_messages=True)

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_documents, calculator],
    system_prompt="You are a helpful assistant with access to tools.",
    memory=memory
)

# Multi-turn conversation
response1 = agent.invoke({"messages": [{"role": "user", "content": "What is RAG?"}]})
response2 = agent.invoke({"messages": [{"role": "user", "content": "How does it compare to fine-tuning?"}]})
```

## Migration from Legacy APIs

```python
# OLD (deprecated)
from langchain.agents import initialize_agent, create_react_agent

# NEW (LangChain 1.0)
from langchain.agents import create_agent

# OLD
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")

# NEW
agent = create_agent(model="gpt-4o-mini", tools=tools, system_prompt="...")
```

## Related Resources

- **LangGraph**: For complex agent workflows
- **LangSmith**: For observability and evaluation
- **LangServe**: For deploying as APIs
