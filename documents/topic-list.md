# AI Engineering Bootcamp - Complete Table of Contents

---

# Chapter 1: Understanding AI Engineering

## What is AI Engineering?
## The AI Engineering Stack
## LLMs and Their Capabilities
- Language Models are Few-Shot Learners
- Emergent Capabilities
- Limitations and Challenges
## From Prototyping to Production
- The Prototyping Phase
- The Productionizing Phase
- Key Skills for Each Phase
## The Role of Agents in Modern AI Systems
## Key Skills and Tools
- Python and Backend Development
- LLM Orchestration Frameworks
- Vector Databases and Search
- Evaluation and Monitoring
## Your First Vibe Check: Getting Started
- What is a Vibe Check?
- Evaluating LLM Applications Qualitatively
- The AI Engineer Challenge
- Context Engineering Fundamentals
## LLM Prototyping Best Practices
- Prompt Engineering Basics
- Iterative Development
- Quick Feedback Loops
## Introduction to Evaluation
- Why Evals Matter from Day One
- Starting with Vibe Checks
- Building Toward Systematic Evaluation

---

# Chapter 2: Dense Vector Retrieval

## Introduction to Context Engineering
- Why context matters for LLMs
- The context window problem
- RAG as a solution
- The LLM application stack

## Introduction to Embeddings
- What are embeddings?
- Semantic similarity vs keyword matching
- Embedding models (OpenAI, alternatives)
- How embeddings capture meaning

## Vector Databases
- Architecture and design principles
- Popular solutions (Chroma, FAISS, Pinecone, Qdrant, Weaviate)
- When you need a vector database
- Building a custom vector database (learning exercise)

## Indexing Strategies
- Brute force vs. approximate search
- HNSW (Hierarchical Navigable Small World)
- IVF (Inverted File Index)
- Trade-offs: speed vs. accuracy

## Similarity Search
- Distance metrics (cosine similarity, Euclidean distance, dot product)
- K-Nearest Neighbors (KNN)
- Approximate Nearest Neighbors (ANN)
- Choosing the right metric

## Chunking Strategies
- Why chunking matters
- Fixed-size chunking
- Semantic chunking (paragraph, section, sentence boundaries)
- Overlap and context preservation
- Chunking best practices

## The RAG Pipeline
- Indexing phase: load → chunk → embed → store
- Query phase: embed query → search → retrieve → generate
- End-to-end implementation

## The 12-Factor Agent Methodology
- Principles for production-ready agents
- Configuration and dependencies management
- State management patterns
- Applying 12-factor principles to AI systems

---

# Chapter 3: The Agent Loop

## Understanding the Agent Pattern
- What makes something an agent?
- Agents vs. workflows
- The ReAct pattern (Reasoning + Acting)
- Agent loop fundamentals

## Core Components of an Agent
- Perception: understanding the current situation
- Planning: deciding what to do next
- Action: using tools and taking steps
- Observation: seeing results and adapting

## Planning and Reasoning
- ReAct pattern implementation
- Chain-of-thought prompting
- Plan-and-execute strategies
- When to plan vs. when to act

## Introduction to LangChain 1.0
- LangChain's philosophy and design
- Component architecture
- Why abstractions matter
- The new create_agent API

## Tool Use and Function Calling
- Defining tools in LangChain
- Tool selection mechanisms
- Parameter extraction
- Error handling in tool execution

## Context Engineering in Agents
- Managing agent context
- What goes in the context window
- Context optimization strategies
- Avoiding context overload

## Working with Qdrant
- Setting up Qdrant (in-memory and server modes)
- Integration with LangChain
- Creating and managing collections
- Retriever configuration

## Reflection and Iteration
- Self-critique mechanisms
- Error recovery patterns
- Deciding when to stop (termination conditions)
- Learning from mistakes

---

# Chapter 4: Agentic RAG

## Beyond Basic RAG
- Limitations of static RAG
- What makes RAG "agentic"
- When you need agentic RAG

## Looking Under the Hood
- Understanding create_agent internals
- When to use high-level vs. low-level APIs
- The case for LangGraph

## Introduction to LangGraph
- LangGraph 1.0 philosophy
- Graph-based agent orchestration
- State machines for agents
- Thinking in graphs: nodes, edges, state

## The Agentic RAG Architecture
- State machine design for RAG
- Node definitions (planning, retrieval, synthesis, reflection)
- Edge definitions (conditional routing, loops)
- Control flow patterns

## Query Planning and Analysis
- Understanding user intent
- Query decomposition into sub-questions
- Complexity analysis
- Generating search strategies

## Dynamic Retrieval Strategies
- Deciding when to retrieve
- Deciding what to retrieve
- Deciding how much to retrieve
- Multi-source retrieval patterns

## Result Synthesis and Response Generation
- Combining multiple sources
- Handling conflicting information
- Citation and attribution
- Confidence scoring

## Handling Ambiguity and Uncertainty
- Query ambiguity detection
- Asking clarifying questions
- Acknowledging uncertainty in answers
- Graceful degradation

## Observability and Monitoring with LangSmith
- Setting up tracing
- Viewing and analyzing traces
- Understanding agent execution flow
- Performance monitoring
- Debugging with traces

---

# Chapter 5: Multi-Agent Applications

## Why Multiple Agents?
- The case for multi-agent systems
- When NOT to build multi-agents
- Decision framework: single vs. multiple agents

## Context Optimization
- The context rot problem
- How multiple agents manage context
- Context isolation benefits
- Splitting concerns across agents

## Multi-Agent Architectures
- Centralized orchestration (supervisor pattern)
- Decentralized coordination (swarm pattern)
- Hierarchical structures
- Choosing the right architecture

## Workflows vs. Agents
- Understanding the distinction
- When to use workflows instead of agents
- Combining workflows and agents
- Hybrid approaches

## Agent Communication
- Message passing patterns
- Shared context management
- Event-driven coordination
- Handoff mechanisms

## Agent Roles and Specialization
- Manager agents (supervisors, coordinators)
- Worker agents (specialists)
- Designing agent personalities and boundaries
- Avoiding role confusion

## Coordination and Orchestration
- Task delegation strategies
- Resource management
- Conflict resolution
- Error handling in multi-agent systems

## Common Multi-Agent Patterns
- Report generation teams
- Research and analysis pipelines
- Multi-step processing workflows
- Parallel task execution

## Visualization and Debugging with LangSmith Studio
- What is LangSmith Studio v2?
- Graph mode vs. Chat mode
- Interactive agent exploration
- Debugging multi-agent interactions
- Testing and iteration workflows

## Use Cases and Applications
- Customer support systems
- Research assistants
- Code generation pipelines
- Content creation workflows

---

# Chapter 6: Agent Memory


## The Memory Problem in AI Agents
- Context window limitations
- Why persistent memory matters
- Memory vs. RAG: understanding the difference

## Memory System Architecture
- LangGraph's native memory store (BaseStore)
- The langmem library
- Namespace organization
- Built-in vector search for memories

## Types of Memory
- Semantic memory: facts and knowledge
- Episodic memory: past experiences and interactions
- Procedural memory: learned behaviors and patterns
- When to use each type

## Memory Formation Strategies
- Hot path: active memory formation during conversations
- Background: passive memory extraction after conversations
- Trade-offs of each approach
- Hybrid strategies

## Memory Storage and Retrieval
- Storing memories with metadata
- Semantic search for relevant memories
- Recency and relevance weighting
- Memory consolidation over time

## Memory Integration Patterns
- Injecting memory into agent context
- Memory-informed tool selection
- Adapting behavior based on history
- User preferences and personalization

## Production Considerations
- Persistent storage (InMemoryStore vs. PostgreSQL)
- Privacy and data governance
- User access to their memories
- Deletion rights and data retention
- Graceful degradation when memory fails

---

# Chapter 7: Deep Agents

## What Makes an Agent "Deep"?
- Beyond simple reasoning
- The deep agents framework
- When you need deep agents vs. simple agents

## The Four Key Elements of Deep Agents
1. Planning and task decomposition
2. Context management over long time horizons
3. Subagent spawning and delegation
4. Long-term memory integration

## Planning and Task Decomposition
- Breaking down complex problems
- Creating subgoals and prerequisite chains
- Forward planning strategies
- Backward chaining from goals
- State space search

## Context Management in Deep Agents
- Managing long-running contexts
- Context window optimization techniques
- Selective information retention
- Context compression strategies
- When to summarize vs. retain details

## Subagent Spawning and Delegation
- When to spawn subagents
- Delegating subtasks effectively
- Managing subagent lifecycles
- Collecting and integrating results
- Parallel vs. sequential delegation

## Multi-Step Reasoning
- Sequential reasoning chains
- Parallel reasoning paths
- Balancing reasoning depth vs. breadth
- When to stop going deeper

## Self-Reflection and Metacognition
- Evaluating progress toward goals
- Strategy adjustment based on results
- Learning from mistakes
- Meta-reasoning about reasoning

## Backtracking and Recovery
- Detecting dead ends
- Rolling back decisions
- Exploring alternative paths
- Maintaining exploration history

## The Claude Agent SDK
- Introduction to Anthropic's approach
- Key concepts and patterns
- When to consider Claude's SDK

---

# Chapter 8: Evaluation Infrastructure

## Deep Research Patterns
- Research as an agentic capability
- Three-step research: scope, research, write
- When deep research matters
- How the Tutor Agent can use research internally

## Why Evaluation Matters
- The testing challenge for AI systems
- Why you need evaluation datasets
- Evaluation as the foundation for improvement

## The Challenge of Creating Test Data
- Manual test creation is slow
- Real data may not cover edge cases
- The need for synthetic data

## Introduction to Synthetic Data
- What is synthetic data?
- Benefits and limitations
- When to use synthetic data
- Quality considerations and validation

## The RAGAS Framework
- Introduction to RAGAS (RAG ASsessment)
- Testset generation capabilities
- Integration with LangSmith
- Why RAGAS for agent evaluation

## The Knowledge Graph Approach
- Understanding the knowledge graph method
- Building knowledge graphs from documents
- Generating questions from graph structure
- Ensuring coverage and diversity

## Generating Test Cases
- LLM-generated examples
- Question-answer pair generation
- Adversarial examples and edge cases
- Covering failure modes

## Data Quality and Coverage
- Diversity in test data
- Realistic scenarios
- Distribution matching
- Validating synthetic data quality

## Metrics-Driven Development
- The philosophy of metrics-driven development
- Measuring to improve
- Iteration loops
- Continuous improvement

## Loading Datasets into LangSmith
- Dataset management in LangSmith
- Organization and versioning
- Running evaluations at scale

---

# Chapter 9: Advanced Retrieval & Evaluation

## Agentic RAG Evaluation
- Evaluation metrics for agentic RAG
- RAGAS core metrics (faithfulness, relevance, context precision/recall)
- Agent-specific metrics (tool usage, reasoning quality)
- Human evaluation strategies

## Beyond Basic Vector Search
- Limitations of dense vector retrieval alone
- The need for advanced techniques
- When to add complexity

## Hybrid Search
- Combining dense and sparse retrieval
- BM25 and traditional keyword search
- Score fusion strategies (Reciprocal Rank Fusion)
- When hybrid search helps

## Reranking
- Why reranking improves results
- Cross-encoders vs. bi-encoders
- Reranking models (Cohere, others)
- Performance trade-offs
- When to rerank

## Query Expansion
- Synonyms and related terms
- Query rewriting with LLMs
- Multi-query strategies
- Avoiding query drift

## RAG-Fusion
- Concept and motivation
- Combining multiple query strategies
- Implementation with LangChain
- When RAG-Fusion helps

## Semantic Chunking
- Beyond fixed-size chunks
- Content-aware chunking strategies
- Preserving semantic boundaries
- Implementation approaches

## Contextual Compression
- Reducing retrieved context
- Relevance filtering
- Extractive summarization
- Balancing completeness vs. conciseness

## Hierarchical Retrieval
- Document summaries for initial retrieval
- Progressive retrieval strategies
- Two-stage and multi-stage retrieval
- When hierarchy helps

## Systematic Comparison of Retrievers
- Defining comparison criteria
- Evaluation methodology
- Benchmarking different approaches
- Using RAGAS for retriever comparison
- Making data-driven decisions

---

# Chapter 10: Full Stack Applications

## Industry Use Cases
- The state of production LLM applications
- Common use cases across industries
- How StudyBuddy patterns generalize
- Lessons from real-world deployments

## Architecture Overview
- Full-stack AI application architecture
- Separation of concerns
- Frontend, backend, and agent layers
- State management strategies

## Frontend Components
- Modern chat interfaces (not just terminal)
- Dashboard and analytics views
- Flashcard review interface design
- Progress tracking visualizations
- Streaming responses
- User feedback mechanisms
- Mobile-responsive design

## Backend Architecture
- FastAPI for agent services
- Request/response handling
- WebSocket integration for streaming
- Background job processing
- Queue management

## State Management
- Client-side state (React, Vue, etc.)
- Server-side state persistence
- State synchronization
- Handling offline scenarios

## Agent Integration
- Calling agent services from frontend
- Handling long-running operations
- Error handling and retries
- Progress updates during generation

## User Experience Considerations
- Loading states and skeletons
- Progressive disclosure of complexity
- Transparency about AI behavior
- Explainability and citations
- Accessibility

## Database Design for Agent Apps
- User data and authentication
- Progress tracking schema
- Flashcard storage
- Learning analytics
- Session management

---

# Chapter 11: MCP Connectors

## Introduction to Model Context Protocol (MCP)
- What is MCP and why it matters
- The standardization problem MCP solves
- The MCP specification overview
- Resources, tools, and prompts

## MCP Architecture
- Client-server model
- Protocol communication
- Resource management
- Security considerations

## Client-Side MCP Usage
- Connecting to MCP servers
- LangChain MCP adapters
- Using MCP for enhanced retrieval
- Authentication and authorization

## Building MCP Connectors
- Connector structure and lifecycle
- Implementing resource access
- Implementing tool exposure
- Error handling and retries

## Integration Patterns
- File system access (local files, cloud storage)
- Database connections
- API integrations (REST, GraphQL)
- Custom tool creation
- Composing multiple MCP servers

## Pros and Cons of Client-Side MCP
- Benefits of standardization
- Performance considerations
- Security trade-offs
- When MCP makes sense

## Testing MCP Connectors
- Local testing strategies
- Integration testing
- Mock servers for development

---

# Chapter 12: Production Deployment

## Production Architecture Considerations
- Development vs. production differences
- Scaling challenges
- Reliability and uptime
- Security requirements

## Packaging and Building Agents
- Agent configuration management
- Dependency management for production
- Environment setup and validation
- Docker containerization

## Deploying with LangSmith Deployment
- Introduction to LangSmith Deployment
- LangSmith Deployment components
- Creating production API endpoints
- Configuration and secrets management
- Cost: ~$30-40/month for deployment

## Agent Server Design
- Request handling patterns
- Queue management for long-running tasks
- Concurrency models
- Worker pool management
- Rate limiting

## Agent Orchestration in Production
- Managing multiple agent instances
- Load distribution strategies
- Failover and redundancy
- Health checks and monitoring

## State Management in Production
- Persistent state storage
- Distributed state challenges
- State recovery after failures
- Backup and restore strategies

## Authentication and Multi-User Support
- User authentication and authorization
- Session management
- User isolation and data privacy
- API key management

## Monitoring and Logging
- Metrics collection (latency, throughput, errors)
- Distributed tracing with LangSmith
- Log aggregation and analysis
- Alerting and on-call

## Scaling Considerations
- Horizontal scaling (more instances)
- Vertical scaling (bigger instances)
- Auto-scaling strategies
- Cost optimization at scale

---

# Chapter 13: Advanced Infrastructure

## LLM Servers
- Model serving fundamentals
- Choosing open-source models (LLM leaderboards, MTEB)
- Deploying to remote production endpoints (Together AI)
- Inference optimization (quantization, batching, KV cache)
- Cost considerations for self-hosting vs. APIs

## MCP Servers and Agent-to-Agent
- Setting up your own MCP server
- Exposing StudyBuddy's capabilities via MCP
- Agent-to-Agent communication (A2A protocol)
- Agent cards for capability discovery
- Remote-to-client agent coordination

## Using Open-Source LLMs
- Latest open-source models (Llama, Mixtral, etc.)
- Open-source embedding models
- Model selection criteria (quality, cost, latency)
- When to use open-source vs. proprietary

## Deploying Open Models
- Together AI for LLM serving
- Deploying custom embedding models
- API configuration and management
- Performance benchmarking

## Popular LLM Servers
- vLLM for high-throughput serving
- Text Generation Inference (TGI)
- Ollama for local development
- OpenAI-compatible server APIs

## Agent-to-Agent Communication Patterns
- Request-response patterns
- Pub-sub for event-driven coordination
- Event sourcing for audit trails
- Multi-agent coordination at scale

## Security in Distributed Systems
- Authentication between agents
- Authorization and access control
- Encrypted communication
- Audit logging

---

# Chapter 14: Production Hardening

## Safety and Guardrails
- Key categories of guardrails (input, output, behavioral)
- Input validation and sanitization
- Output filtering and moderation
- Content safety for educational contexts
- Prompt injection prevention

## Implementing Guardrails
- Rule-based systems (regex, keyword filtering)
- ML-based detection (toxicity classifiers)
- Hybrid approaches
- Custom LangGraph guardrail implementations

## The Importance of Caching
- Performance benefits (reduced latency)
- Cost reduction (fewer API calls)
- User experience improvements
- When caching helps most

## Semantic Caching
- What is semantic caching?
- Caching based on meaning, not exact matches
- Implementation strategies
- Cache key design for similar queries
- Hit rate optimization

## Response Caching
- Traditional response caching
- Caching LLM responses
- Cache invalidation strategies
- Time-to-live (TTL) policies
- Balancing freshness vs. performance

## Prompt Caching
- How prompt caching reduces latency
- Cost optimization with prompt caching
- Best practices for cache-friendly prompts
- Provider-specific implementations (OpenAI, Anthropic)

## Embedding Caching with CacheBackedEmbeddings
- Setting up CacheBackedEmbeddings
- Storage backends (local, Redis, etc.)
- Performance improvements
- Cost savings from avoiding re-embedding

## Cost Optimization Strategies
- Token usage optimization
- Model selection for cost vs. quality
- Request batching
- Combining multiple caching strategies
- Usage monitoring and budgets

## Rate Limiting and Throttling
- Protecting against abuse
- Per-user rate limits
- Tiered access levels
- Graceful degradation under load

## Production Checklist
- Pre-deployment verification
- Security audit checklist
- Performance benchmarks
- Monitoring and alerting setup
- Incident response procedures
- Backup and disaster recovery
