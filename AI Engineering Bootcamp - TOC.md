# AI Engineering Bootcamp - Complete Table of Contents

---

### Introduction
- What This Book Covers
- Who This Book Is For  
- How to Use This Book
- Prerequisites
- The Journey from Prototype to Production
- Meet StudyBuddy: Your Project for This Book

---
# Part 1: Foundations
---

## Chapter 1: Understanding AI Engineering

### What is AI Engineering?
### The AI Engineering Stack
### LLMs and Their Capabilities
- Language Models are Few-Shot Learners
- Emergent Capabilities
- Limitations and Challenges
### From Prototyping to Production
- The Prototyping Phase
- The Productionizing Phase
- Key Skills for Each Phase
### The Role of Agents in Modern AI Systems
### Key Skills and Tools
- Python and Backend Development
- LLM Orchestration Frameworks
- Vector Databases and Search
- Evaluation and Monitoring
### Your First Vibe Check: Getting Started
- What is a Vibe Check?
- Evaluating LLM Applications Qualitatively
- The AI Engineer Challenge
- Context Engineering Fundamentals
### LLM Prototyping Best Practices
- Prompt Engineering Basics
- Iterative Development
- Quick Feedback Loops
### Introduction to Evaluation
- Why Evals Matter from Day One
- Starting with Vibe Checks
- Building Toward Systematic Evaluation
### Building StudyBuddy v1

---

## Chapter 2: Dense Vector Retrieval

### Introduction to Context Engineering
- Why context matters for LLMs
- The context window problem
- RAG as a solution
- The LLM application stack

### Introduction to Embeddings
- What are embeddings?
- Semantic similarity vs keyword matching
- Embedding models (OpenAI, alternatives)
- How embeddings capture meaning

### Vector Databases
- Architecture and design principles
- Popular solutions (Chroma, FAISS, Pinecone, Qdrant, Weaviate)
- When you need a vector database
- Building a custom vector database (learning exercise)

### Indexing Strategies
- Brute force vs. approximate search
- HNSW (Hierarchical Navigable Small World)
- IVF (Inverted File Index)
- Trade-offs: speed vs. accuracy

### Similarity Search
- Distance metrics (cosine similarity, Euclidean distance, dot product)
- K-Nearest Neighbors (KNN)
- Approximate Nearest Neighbors (ANN)
- Choosing the right metric

### Chunking Strategies
- Why chunking matters
- Fixed-size chunking
- Semantic chunking (paragraph, section, sentence boundaries)
- Overlap and context preservation
- Chunking best practices

### The RAG Pipeline
- Indexing phase: load → chunk → embed → store
- Query phase: embed query → search → retrieve → generate
- End-to-end implementation

### The 12-Factor Agent Methodology
- Principles for production-ready agents
- Configuration and dependencies management
- State management patterns
- Applying 12-factor principles to AI systems

### Building StudyBuddy v2
- Adding document upload capability
- Implementing RAG from scratch (learning exercise)
- Uploading study materials (use book chapters as examples)
- Grounding explanations in actual content
- Testing retrieval quality
- Understanding what's happening under the hood

---

## Chapter 3: The Agent Loop

### Understanding the Agent Pattern
- What makes something an agent?
- Agents vs. workflows
- The ReAct pattern (Reasoning + Acting)
- Agent loop fundamentals

### Core Components of an Agent
- Perception: understanding the current situation
- Planning: deciding what to do next
- Action: using tools and taking steps
- Observation: seeing results and adapting

### Planning and Reasoning
- ReAct pattern implementation
- Chain-of-thought prompting
- Plan-and-execute strategies
- When to plan vs. when to act

### Introduction to LangChain 1.0
- LangChain's philosophy and design
- Component architecture
- Why abstractions matter
- The new create_agent API

### Tool Use and Function Calling
- Defining tools in LangChain
- Tool selection mechanisms
- Parameter extraction
- Error handling in tool execution

### Context Engineering in Agents
- Managing agent context
- What goes in the context window
- Context optimization strategies
- Avoiding context overload

### Working with Qdrant
- Setting up Qdrant (in-memory and server modes)
- Integration with LangChain
- Creating and managing collections
- Retriever configuration

### LangChain Middleware
- What is middleware?
- Common middleware patterns
- Custom middleware development
- When to use middleware

### Reflection and Iteration
- Self-critique mechanisms
- Error recovery patterns
- Deciding when to stop (termination conditions)
- Learning from mistakes

### Building StudyBuddy v3
- Rebuilding with LangChain's create_agent
- Adding tools: search_materials, get_more_context
- Implementing the ReAct loop
- Agent decides when to search vs. answer from knowledge
- Switching from custom vector DB to Qdrant
- Testing agentic behavior
- Observing the reasoning process

---

## Chapter 4: Agentic RAG

### Beyond Basic RAG
- Limitations of static RAG
- What makes RAG "agentic"
- When you need agentic RAG

### Looking Under the Hood
- Understanding create_agent internals
- When to use high-level vs. low-level APIs
- The case for LangGraph

### Introduction to LangGraph
- LangGraph 1.0 philosophy
- Graph-based agent orchestration
- State machines for agents
- Thinking in graphs: nodes, edges, state

### The Agentic RAG Architecture
- State machine design for RAG
- Node definitions (planning, retrieval, synthesis, reflection)
- Edge definitions (conditional routing, loops)
- Control flow patterns

### Query Planning and Analysis
- Understanding user intent
- Query decomposition into sub-questions
- Complexity analysis
- Generating search strategies

### Dynamic Retrieval Strategies
- Deciding when to retrieve
- Deciding what to retrieve
- Deciding how much to retrieve
- Multi-source retrieval patterns

### Result Synthesis and Response Generation
- Combining multiple sources
- Handling conflicting information
- Citation and attribution
- Confidence scoring

### Handling Ambiguity and Uncertainty
- Query ambiguity detection
- Asking clarifying questions
- Acknowledging uncertainty in answers
- Graceful degradation

### Observability and Monitoring with LangSmith
- Setting up tracing
- Viewing and analyzing traces
- Understanding agent execution flow
- Performance monitoring
- Debugging with traces

### Building StudyBuddy v4
- Rebuilding with LangGraph for fine-grained control
- Implementing query analysis node
- Dynamic retrieval strategies based on question complexity
- Reflection and confidence scoring
- Adding first flashcard generation capability (agent creates cards when explaining important concepts)
- Setting up LangSmith observability
- Testing the complete agentic RAG flow

---
# Part 2: Advanced Agents
---

## Chapter 5: Multi-Agent Applications

### When to Use Multiple Agents
- The case for multi-agent systems
- When NOT to build multi-agents
- Workflows vs. agents: understanding the distinction
- Decision framework: single agent, multi-agent, or workflow

### Context Optimization
- The context rot problem
- How multiple agents manage context
- Context isolation benefits
- Splitting concerns across agents

### Multi-Agent Architectures
- Centralized orchestration (supervisor pattern)
- Decentralized coordination (swarm pattern)
- Hierarchical structures
- Choosing the right architecture

### Agent Roles and Specialization
- Manager agents (supervisors, coordinators)
- Worker agents (specialists)
- Designing agent personalities and boundaries
- Avoiding role confusion

### Coordination and Communication
- Message passing and handoff mechanisms
- Task delegation strategies
- Shared context management
- Conflict resolution and error handling

### Multi-Agent Patterns and Use Cases
- Report generation teams
- Research and analysis pipelines
- Code generation workflows
- Customer support systems
- Content creation workflows
- Parallel task execution patterns

### Visualization and Debugging with LangSmith Studio
- What is LangSmith Studio v2?
- Graph mode vs. Chat mode
- Interactive agent exploration
- Debugging multi-agent interactions
- Testing and iteration workflows

### Building StudyBuddy v5
- Multi-agent architecture with supervisor pattern using langgraph-supervisor
- Agent team:
  - **Tutor Agent:** Explains concepts conversationally (gpt-4o)
  - **Card Generator Agent:** Creates flashcards from explanations and materials (gpt-4o-mini)
  - **Quality Checker Agent:** Validates card clarity and usefulness
  - **Learning Coordinator (Supervisor):** Orchestrates all agents
- Coordinating learning mode via the supervisor agent
- Background card generation with quality checking pipeline
- Error recovery: handling rejected cards and agent failures
- Testing multi-agent coordination
- Using LangSmith Studio to visualize agent interactions

---

## Chapter 6: Agent Memory

### The Memory Problem in AI Agents
- Why persistent memory matters
- Memory vs. RAG: understanding the difference

### Memory System Architecture
- The memory store pattern
- Giving agents memory tools
- Namespace organization
- Search and retrieval patterns

### Types of Memory
- Semantic memory: facts and knowledge
- Episodic memory: past experiences and interactions
- Procedural memory: learned behaviors and patterns
- When to use each type

### Memory Formation Strategies
- Hot path: active memory formation during conversations
- Background: passive memory extraction after conversations
- Trade-offs and decision framework
- Hybrid strategies

### Memory Storage and Retrieval
- Storing memories with metadata
- Semantic search for relevant memories
- Recency and relevance weighting
- Memory consolidation over time

### Memory Integration Patterns
- Injection at start
- On-demand retrieval
- Proactive memory
- Adapting behavior based on history

### Persistent State with PostgreSQL
- Why PostgreSQL everywhere (dev-prod parity)
- Setting up local PostgreSQL (Homebrew)
- Database schema design
- Configuring SQLAlchemy
- Working with SQLAlchemy models
- Setting up Vercel Postgres

### Performance Optimization Through Caching
- The cost of redundant generation
- Content-addressed caching strategies
- Database-backed performance wins

### Caching Generated Content
- Hashing inputs for cache keys
- Storing generated outputs
- Measuring performance improvements
- When to cache

### Building StudyBuddy v6
- Where we left off
- What we're adding
- **Persistence strategy**: 
  - PostgreSQL everywhere (same database in dev and production)
  - Local PostgreSQL via Homebrew
  - Vercel Postgres for production
  - Using SQLAlchemy for database abstraction
- **Database configuration**:
  - Environment-based connection strings
  - POSTGRES_URL requirement (fail fast, no silent fallbacks)
  - Connection pooling for serverless
- **Custom memory store**:
  - SQLAlchemy-backed MemoryStore class
  - Namespace/key/value pattern
  - No external memory libraries (full control)
- **Content-addressed flashcard caching**:
  - Hash topic + context for cache keys
  - Check database before generating new cards
  - Instant retrieval on cache hits
- **The Scheduler Agent and SM-2 algorithm**:
  - Spaced repetition for optimal learning
  - Tracking ease factors and intervals
  - Review scheduling based on performance
- **Tracking learning history**:
  - What concepts you've studied
  - Performance analytics over time
  - Struggle areas identification
- Memory connecting both modes: tutoring informs practice, practice informs tutoring
- **Testing persistence**:
  - Verify data survives server restart
  - Test cache hits vs. misses
  - Database inspection with psql
- What's next

---

## Chapter 7: Deep Agents

### What Makes an Agent "Deep"?
- Beyond simple reasoning
- The deep agents framework
- When you need deep agents vs. simple agents

### The Four Key Elements of Deep Agents
1. Planning and task decomposition
2. Context management over long time horizons
3. Subagent spawning and delegation
4. Long-term memory integration

### Planning and Task Decomposition
- Breaking down complex problems
- Creating subgoals and prerequisite chains
- Forward planning strategies
- Backward chaining from goals
- State space search

### Context Management in Deep Agents
- Managing long-running contexts
- Context window optimization techniques
- Selective information retention
- Context compression strategies
- When to summarize vs. retain details

### Subagent Spawning and Delegation
- When to spawn subagents
- Delegating subtasks effectively
- Managing subagent lifecycles
- Collecting and integrating results
- Parallel vs. sequential delegation

### Multi-Step Reasoning
- Sequential reasoning chains
- Parallel reasoning paths
- Balancing reasoning depth vs. breadth
- When to stop going deeper

### Self-Reflection and Metacognition
- Evaluating progress toward goals
- Strategy adjustment based on results
- Learning from mistakes
- Meta-reasoning about reasoning

### Backtracking and Recovery
- Detecting dead ends
- Rolling back decisions
- Exploring alternative paths
- Maintaining exploration history

### The Claude Agent SDK
- Introduction to Anthropic's approach
- Key concepts and patterns
- When to consider Claude's SDK

### Building StudyBuddy v7
- Adding deep agent capabilities with the **Curriculum Planner Agent**
- Task decomposition: Breaking complex learning goals into chapters, sections, and prerequisite chains
- Learning path optimization using backward chaining from goals
- Subagent spawning: Curriculum planner delegates to card generator for each section
- Progress checkpoints: Define milestones that trigger knowledge assessments
- Context summarization: Compress long study sessions to maintain context across interactions
- Managing learning context over multiple sessions with selective retention
- Testing curriculum generation and learning path planning
- Backtracking when a learning approach isn't working

---
# Part 3: Evaluation and Optimization
---

## Chapter 8: Evaluation Infrastructure

### Deep Research Patterns (Condensed from original Chapter 8)
- Research as an agentic capability
- Three-step research: scope, research, write
- When deep research matters
- How the Tutor Agent can use research internally

### Why Evaluation Matters
- The testing challenge for AI systems
- Why you need evaluation datasets
- Evaluation as the foundation for improvement

### The Challenge of Creating Test Data
- Manual test creation is slow
- Real data may not cover edge cases
- The need for synthetic data

### Introduction to Synthetic Data
- What is synthetic data?
- Benefits and limitations
- When to use synthetic data
- Quality considerations and validation

### The RAGAS Framework
- Introduction to RAGAS (RAG ASsessment)
- Testset generation capabilities
- Integration with LangSmith
- Why RAGAS for agent evaluation

### The Knowledge Graph Approach
- Understanding the knowledge graph method
- Building knowledge graphs from documents
- Generating questions from graph structure
- Ensuring coverage and diversity

### Generating Test Cases
- LLM-generated examples
- Question-answer pair generation
- Adversarial examples and edge cases
- Covering failure modes

### Data Quality and Coverage
- Diversity in test data
- Realistic scenarios
- Distribution matching
- Validating synthetic data quality

### Metrics-Driven Development
- The philosophy of metrics-driven development
- Measuring to improve
- Iteration loops
- Continuous improvement

### Loading Datasets into LangSmith
- Dataset management in LangSmith
- Organization and versioning
- Running evaluations at scale

### Building StudyBuddy v8
- **Synthetic test data generation** using RAGAS knowledge graph approach:
  - Question-answer pairs from reference documents
  - Flashcard quality validation sets (clear, ambiguous, incorrect examples)
  - Edge cases and difficult concepts at varying difficulty levels
- **Knowledge graph construction** from documents to ensure comprehensive topic coverage
- Building evaluation datasets for:
  - Tutoring explanation quality and accuracy
  - Flashcard clarity, usefulness, and pedagogical value
  - Retrieval accuracy (preparing for v9 improvements)
- **LangSmith dataset integration**: Store and version evaluation datasets
- Setting up systematic evaluation pipelines in LangSmith
- Creating baseline performance metrics for comparison
- **Evaluation dashboard**: Display quality scores and identify weak content areas
- Metrics-driven development: Using evals to drive improvement

---

## Chapter 9: Advanced Retrieval & Evaluation

### Agentic RAG Evaluation
- Evaluation metrics for agentic RAG
- RAGAS core metrics (faithfulness, relevance, context precision/recall)
- Agent-specific metrics (tool usage, reasoning quality)
- Human evaluation strategies

### Beyond Basic Vector Search
- Limitations of dense vector retrieval alone
- The need for advanced techniques
- When to add complexity

### Hybrid Search
- Combining dense and sparse retrieval
- BM25 and traditional keyword search
- Score fusion strategies (Reciprocal Rank Fusion)
- When hybrid search helps

### Reranking
- Why reranking improves results
- Cross-encoders vs. bi-encoders
- Reranking models (Cohere, others)
- Performance trade-offs
- When to rerank

### Query Expansion
- Synonyms and related terms
- Query rewriting with LLMs
- Multi-query strategies
- Avoiding query drift

### RAG-Fusion
- Concept and motivation
- Combining multiple query strategies
- Implementation with LangChain
- When RAG-Fusion helps

### Semantic Chunking
- Beyond fixed-size chunks
- Content-aware chunking strategies
- Preserving semantic boundaries
- Implementation approaches

### Contextual Compression
- Reducing retrieved context
- Relevance filtering
- Extractive summarization
- Balancing completeness vs. conciseness

### Hierarchical Retrieval
- Document summaries for initial retrieval
- Progressive retrieval strategies
- Two-stage and multi-stage retrieval
- When hierarchy helps

### Systematic Comparison of Retrievers
- Defining comparison criteria
- Evaluation methodology
- Benchmarking different approaches
- Using RAGAS for retriever comparison
- Making data-driven decisions

### Building StudyBuddy v9
- **Hybrid search**: Combining dense vector search with BM25 keyword search using Reciprocal Rank Fusion
- **Reranking pipeline**: Adding Cohere cross-encoder reranker to improve retrieval precision
- **RAG-Fusion**: Generating multiple query variations and fusing results for better recall
- **Semantic chunking**: Replacing fixed-size chunks with content-aware chunking that respects section boundaries
- Running evaluations using the datasets from v8
- Measuring retrieval improvements with RAGAS metrics (faithfulness, relevance, context precision/recall)
- **Retrieval comparison tool**: Systematic side-by-side comparison of retrieval strategies
- Documenting quantitative performance gains
- Balancing retrieval quality vs. latency trade-offs

---
# Part 4: Production Systems
---

## Chapter 10: Full Stack Applications

### Industry Use Cases (Condensed from original Chapter 12)
- The state of production LLM applications
- Common use cases across industries
- How StudyBuddy patterns generalize
- Lessons from real-world deployments

### Architecture Overview
- Full-stack AI application architecture
- Separation of concerns
- Frontend, backend, and agent layers
- State management strategies

### Frontend Components
- Modern chat interfaces (not just terminal)
- Dashboard and analytics views
- Flashcard review interface design
- Progress tracking visualizations
- Streaming responses
- User feedback mechanisms
- Mobile-responsive design

### Backend Architecture
- FastAPI for agent services
- Request/response handling
- WebSocket integration for streaming
- Background job processing
- Queue management

### State Management
- Client-side state (React, Vue, etc.)
- Server-side state persistence
- State synchronization
- Handling offline scenarios

### Agent Integration
- Calling agent services from frontend
- Handling long-running operations
- Error handling and retries
- Progress updates during generation

### User Experience Considerations
- Loading states and skeletons
- Progressive disclosure of complexity
- Transparency about AI behavior
- Explainability and citations
- Accessibility

### Database Design for Agent Apps
- User data and authentication
- Progress tracking schema
- Flashcard storage
- Learning analytics
- Session management

### Persistent Vector Storage with pgvector
- The problem with in-memory vector stores (re-indexing on every restart)
- Why persistence matters when users upload their own documents
- pgvector: adding vector capabilities to PostgreSQL
- Migrating from Qdrant in-memory to pgvector
- Program-scoped vector collections (isolating each learning program's documents)
- Trade-offs: pgvector vs. dedicated vector databases

### Building StudyBuddy v10
- **Generalizing StudyBuddy to learn ANY subject** (the major feature):
  - **Document upload**: Upload PDF, Markdown, or text files to create custom knowledge bases
  - **Custom topic lists**: Upload or AI-generate topic-list.md for any subject
  - **Learning program management**: Create, switch between, and manage multiple programs (Spanish, History, Cooking, etc.)
  - **AI curriculum generation**: "Create a learning program for [topic]" generates complete curriculum
- **Persistent vector storage with pgvector**:
  - Moving from in-memory Qdrant to PostgreSQL-backed vectors
  - No more re-indexing on restart—uploaded documents persist
  - Per-program vector isolation using metadata filtering
- Building proper full-stack UI:
  - **Progress dashboard** with charts showing learning progress across all programs
  - Program selector and management interface
  - Document upload with drag-and-drop
  - Analytics visualizations per program
- **Streaming responses**: WebSocket-based streaming for long generation tasks
- FastAPI file handling and background jobs
- User-scoped data isolation (preparing for v12's multi-user)
- Connecting v7's curriculum planner + v9's advanced retrieval for user documents
- Testing the complete user experience across multiple learning programs

---

## Chapter 11: MCP Connectors

### Introduction to Model Context Protocol (MCP)
- What is MCP and why it matters
- The standardization problem MCP solves
- The MCP specification overview
- Resources, tools, and prompts

### MCP Architecture
- Client-server model
- Protocol communication
- Resource management
- Security considerations

### Client-Side MCP Usage
- Connecting to MCP servers
- LangChain MCP adapters
- Using MCP for enhanced retrieval
- Authentication and authorization

### Building MCP Connectors
- Connector structure and lifecycle
- Implementing resource access
- Implementing tool exposure
- Error handling and retries

### Integration Patterns
- File system access (local files, cloud storage)
- Database connections
- API integrations (REST, GraphQL)
- Custom tool creation
- Composing multiple MCP servers

### Testing MCP Connectors
- Local testing strategies
- Integration testing
- Mock servers for development

### Pros and Cons of Client-Side MCP
- Benefits of standardization
- Performance considerations
- Security trade-offs
- When MCP makes sense

### Building StudyBuddy v11
- **MCP client integration** with LangChain adapters for external data sources
- **GitHub connector (full implementation)**:
  - Import README files, docs, and markdown from repos as learning materials
  - OAuth authentication flow
  - Incremental sync (track changes, re-index only updates)
- **Notion connector (demo)**: Stub implementation showing the pattern with mock data
- **Google Drive connector (demo)**: Stub showing OAuth flow pattern without full implementation
- **Connector management UI**: Add, configure, and view sync status for external sources
- External sources become content within Learning Programs from v10
- Handling authentication and authorization patterns
- Testing integrations and graceful degradation when services unavailable

---

## Chapter 12: Production Deployment

### Production Architecture Considerations
- Development vs. production differences
- Scaling challenges
- Reliability and uptime
- Security requirements

### Packaging and Building Agents
- Agent configuration management
- Dependency management for production
- Environment setup and validation
- Docker containerization

### Deploying with LangSmith Deployment
- Introduction to LangSmith Deployment (formerly LangGraph Platform)
- LangSmith Deployment components
- Creating production API endpoints
- Configuration and secrets management
- Cost: ~$30-40/month for deployment

### Agent Server Design
- Request handling patterns
- Queue management for long-running tasks
- Concurrency models
- Worker pool management
- Rate limiting

### Agent Orchestration in Production
- Managing multiple agent instances
- Load distribution strategies
- Failover and redundancy
- Health checks and monitoring

### State Management in Production
- Persistent state storage
- Distributed state challenges
- State recovery after failures
- Backup and restore strategies

### Authentication and Multi-User Support
- User authentication and authorization
- Session management
- User isolation and data privacy
- API key management

### Monitoring and Logging
- Metrics collection (latency, throughput, errors)
- Distributed tracing with LangSmith
- Log aggregation and analysis
- Alerting and on-call

### Scaling Considerations
- Horizontal scaling (more instances)
- Vertical scaling (bigger instances)
- Auto-scaling strategies
- Cost optimization at scale

### Building StudyBuddy v12
- **User authentication**: Full JWT auth with registration, login, password reset, email verification
- **Multi-user data isolation**: User-scoped Qdrant collections, learning programs, and progress
- **LangSmith Deployment**: Package and deploy agents as production API endpoints (~$30-40/month)
- **Production monitoring**: Metrics collection, health checks, alerting setup
- **Rate limiting**: Per-user request limits to manage costs and prevent abuse
- **Background job queue**: Redis-backed async processing for long-running tasks
- Docker containerization for deployment
- PostgreSQL mandatory (no SQLite fallback)
- Load testing with multiple concurrent users
- Setting up production observability with LangSmith tracing
- Documenting the deployment process and cost analysis

---

## Chapter 13: Advanced Infrastructure

### LLM Servers (Condensed from original Chapter 16)
- Model serving fundamentals
- Choosing open-source models (LLM leaderboards, MTEB)
- Deploying to remote production endpoints (Together AI)
- Inference optimization (quantization, batching, KV cache)
- Cost considerations for self-hosting vs. APIs

### MCP Servers and Agent-to-Agent (Condensed from original Chapter 17)
- Setting up your own MCP server
- Exposing StudyBuddy's capabilities via MCP
- Agent-to-Agent communication (A2A protocol)
- Agent cards for capability discovery
- Remote-to-client agent coordination

### Using Open-Source LLMs
- Latest open-source models (Llama, Mixtral, etc.)
- Open-source embedding models
- Model selection criteria (quality, cost, latency)
- When to use open-source vs. proprietary

### Deploying Open Models
- Together AI for LLM serving
- Deploying custom embedding models
- API configuration and management
- Performance benchmarking

### Popular LLM Servers
- vLLM for high-throughput serving
- Text Generation Inference (TGI)
- Ollama for local development
- OpenAI-compatible server APIs

### Agent-to-Agent Communication Patterns
- Request-response patterns
- Pub-sub for event-driven coordination
- Event sourcing for audit trails
- Multi-agent coordination at scale

### Security in Distributed Systems
- Authentication between agents
- Authorization and access control
- Encrypted communication
- Audit logging

### Building StudyBuddy v13
- **Open-source LLM support**: Use Llama, Mixtral via Together AI as alternatives to OpenAI
- **Model selection**: Per-task model configuration (open models for card generation, OpenAI for complex reasoning)
- Deploying open-source embedding models
- Benchmarking open vs. proprietary model quality and latency
- **StudyBuddy MCP Server**: Expose capabilities as MCP tools for other agents:
  - `generate_flashcards`: Create flashcards for a topic
  - `explain_concept`: Get tutoring explanation
  - `get_curriculum`: Retrieve curriculum structure
  - `search_knowledge`: Search indexed materials
- **Agent-to-agent communication**: Allow external agents to request StudyBuddy services
- **Cost comparison dashboard**: Compare token costs between providers
- Documenting cost savings achievable with open-source models

---

## Chapter 14: Production Hardening

### Safety and Guardrails
- Key categories of guardrails (input, output, behavioral)
- Input validation and sanitization
- Output filtering and moderation
- Content safety for educational contexts
- Prompt injection prevention

### Implementing Guardrails
- Rule-based systems (regex, keyword filtering)
- ML-based detection (toxicity classifiers)
- Hybrid approaches
- Custom LangGraph guardrail implementations

### The Importance of Caching
- Performance benefits (reduced latency)
- Cost reduction (fewer API calls)
- User experience improvements
- When caching helps most

### Semantic Caching
- What is semantic caching?
- Caching based on meaning, not exact matches
- Implementation strategies
- Cache key design for similar queries
- Hit rate optimization

### Response Caching
- Traditional response caching
- Caching LLM responses
- Cache invalidation strategies
- Time-to-live (TTL) policies
- Balancing freshness vs. performance

### Prompt Caching
- How prompt caching reduces latency
- Cost optimization with prompt caching
- Best practices for cache-friendly prompts
- Provider-specific implementations (OpenAI, Anthropic)

### Embedding Caching with CacheBackedEmbeddings
- Setting up CacheBackedEmbeddings
- Storage backends (local, Redis, etc.)
- Performance improvements
- Cost savings from avoiding re-embedding

### Cost Optimization Strategies
- Token usage optimization
- Model selection for cost vs. quality
- Request batching
- Combining multiple caching strategies
- Usage monitoring and budgets

### Rate Limiting and Throttling
- Protecting against abuse
- Per-user rate limits
- Tiered access levels
- Graceful degradation under load

### Production Checklist
- Pre-deployment verification
- Security audit checklist
- Performance benchmarks
- Monitoring and alerting setup
- Incident response procedures
- Backup and disaster recovery

### Building StudyBuddy v14
- **This is the final, production-ready, deployment-hardened version**
- **Content guardrails**:
  - Input validation (prompt injection detection, inappropriate content filtering)
  - Output filtering (age-appropriate explanations, educational safety)
  - LangGraph guardrail node implementations
- **Comprehensive caching**:
  - **Semantic response cache**: Cache similar queries using embedding similarity (not just exact matches)
  - **Prompt caching**: Provider-specific implementations (Anthropic's cache, OpenAI's caching)
  - **CacheBackedEmbeddings**: Avoid re-embedding unchanged documents
- **Cost analytics dashboard**: Token usage tracking, cost attribution by feature, budget alerts
- **Production checklist UI**: Readiness status across security, performance, monitoring categories
- Performance tuning (latency, throughput optimization)
- Security hardening (authentication, authorization, input validation)
- Final monitoring and alerting setup
- Load testing under production conditions
- Documentation for operations and maintenance
- **This is what you share with the world**

---

### Appendix A: Setting Up Your Dev Environment

- Required Software and Tools
- Code Editor Setup (Cursor or VS Code)
- Getting to Know Your Terminal
- Git and GitHub Setup
- Setting Up Python with uv
- Getting API Keys (OpenAI, Anthropic, Hugging Face)
- Setting Up Docker
- Jupyter Notebooks
- Project Structure Best Practices
- First Integration Test

### Appendix B: Twelve Projects to Build, Ship, and Share
