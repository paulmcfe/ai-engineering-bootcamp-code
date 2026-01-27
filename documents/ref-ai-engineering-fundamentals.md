# AI Engineering Fundamentals

## What is AI Engineering?

AI Engineering is the discipline of building applications powered by large language models. While it shares DNA with both machine learning engineering and traditional software engineering, it's distinct from both.

Traditional ML engineering requires deep expertise in model architecture, training pipelines, and mathematical foundations. You're working with data scientists to prepare training data, tune hyperparameters, and optimize model performance. AI Engineering, by contrast, treats models as APIs. You're not training GPT-5, you're building applications that use it. The skills shift from model internals to prompt design, context management, and system architecture.

AI Engineering also differs from traditional software engineering, though it builds heavily on those foundations. The key difference is non-determinism. Traditional code does exactly what you tell it, every time. LLM-powered applications are probabilistic—the same input can produce different outputs, and you need to design systems that handle this gracefully.

The AI Engineer role emerged around 2023 as organizations realized they needed people who could bridge the gap between powerful foundation models and practical applications. These engineers understand enough about LLMs to use them effectively, enough about software engineering to build production systems, and enough about product development to create experiences that actually work for users.

## The AI Engineering Stack

The AI Engineering stack consists of four layers:

**Foundation Models** sit at the bottom—the large language models themselves (GPT-4, Claude, Llama). Most AI Engineers don't train these; they consume them as services. Understanding model capabilities, limitations, and pricing helps you choose the right model for each task.

**API and Inference Layer** is where you interact with models. This includes direct API access through providers like OpenAI and Anthropic, plus inference platforms for open-source models. Rate limits, batching strategies, and cost optimization matter here.

**Orchestration Layer** is where AI Engineers spend most of their time. Raw API calls only get you so far. Real applications need chains of calls, retrieval systems, agent loops, and complex workflows. Frameworks like LangChain and LangGraph live here, as do patterns like RAG and ReAct.

**Application Layer** sits at the top—the actual products you're building. Chatbots, copilots, search systems, content generators, and autonomous agents.

## LLMs: Capabilities and Limitations

Large language models learn new tasks from examples provided in the prompt, without fine-tuning. This in-context learning is why prompt engineering matters—the examples and instructions you provide shape behavior. As models scale, they develop emergent capabilities like chain-of-thought reasoning, code generation, and complex instruction-following.

But LLMs have significant limitations. They hallucinate confidently, stating false things as fact. Context windows constrain how much information fits in a single prompt. The same prompt can produce different outputs. API calls are slow and expensive compared to traditional compute. These limitations don't make LLMs unusable—they shape how you design applications around them.

> For deeper coverage, see: `ref-prompt-engineering.md`

## From Prototyping to Production

One of the unique aspects of AI Engineering is how dramatically different the prototyping and production phases feel.

### Prototyping

Prototyping with LLMs is magical. You can go from idea to working demo in hours. Write a prompt, call an API, and you've got something that answers questions or generates content. This speed creates a trap—demos that seem brilliant often hide fundamental issues that emerge at scale.

During prototyping, focus on exploration and validation. Does this task work with an LLM? What prompt approach gives the best results? How do users react? You're testing hypotheses rapidly, not building durable systems. Jupyter notebooks, playground interfaces, and simple scripts are your tools.

### Production

Production is different. That prompt which worked in your notebook fails on unexpected inputs. Costs become untenable at scale. Users complain about latency.

Productionizing requires systematic attention to reliability, observability, and evaluation. You need error handling, retries, monitoring, and logging. Simple request-response patterns give way to async processing, queuing, caching, and streaming. Hardcoded prompts evolve into template systems with dynamic context injection.

### Skills for Each Phase

For prototyping: prompt engineering, basic Python, API familiarity, and intuition for what LLMs can do.

For production: software engineering fundamentals (API design, error handling, testing, monitoring, deployment), evaluation frameworks, and the discipline to build comprehensive test suites.

The best AI Engineers develop strength in both areas—moving quickly during prototyping without creating technical debt that makes production painful.

## Agents: The Frontier

Agents are LLM-powered systems that take actions, not just generate text. They combine an LLM with external tools—instead of generating text about searching the web, an agent actually searches, processes results, and incorporates them into its response.

The ReAct pattern (Reasoning + Acting) formalized this: the model alternates between reasoning steps (thinking through what to do) and action steps (executing tools). More sophisticated agents add memory systems, multi-agent coordination, and planning capabilities.

Agents come with new challenges: stuck loops, cascading errors, unexpected actions, and debugging complexity that exceeds simple LLM calls.

> For deeper coverage, see: `ref-agents-and-agency.md`, `ref-react-pattern.md`, `ref-tool-use.md`

## Vibe Checks

A vibe check is running examples through your system to see if outputs feel right. It's informal and intuitive—you're developing a feel for quality before building elaborate evaluation suites.

Good vibe checks require structure. Test a range of inputs: easy cases, hard cases, edge cases. Typical users, adversarial users. Questions with clear answers and questions requiring nuance. Take notes on what's working and failing. These observations become test cases in your systematic evaluation later.

Vibe checks aren't a replacement for rigorous evaluation—you need automated evaluation before shipping. But they build the intuition that informs those evaluations and provide quick sanity checks during development.

## Context Engineering

If prompt engineering is about crafting instructions, context engineering is about controlling what information the model has access to. When you send a prompt, you're filling a context window with everything the model should consider: system instructions, examples, retrieved documents, conversation history, and the user query.

Context engineering means making deliberate decisions about what goes into that window. What documents do you retrieve? How much conversation history? What system instructions? How do you structure it all?

Think of context as the model's working memory. Everything it knows for a request is either baked into weights from training or provided in context. You can't change the weights, but you control context completely.

## Prototyping Best Practices

**Start with clarity.** Be specific. Instead of "summarize this text," try "summarize in 2-3 sentences, focusing on main conclusions."

**Use examples.** Few-shot prompting—showing input-output pairs before your actual input—improves consistency dramatically.

**Think step by step.** For complex reasoning, "Let's think step by step" significantly improves accuracy.

**Start simple.** Begin with the most straightforward prompt. Add complexity only to address specific failures.

**Change one thing at a time.** When debugging, resist rewriting everything. Change one aspect, observe, iterate.

**Keep notes.** Track what you've tried to avoid going in circles.

**Build small test sets early.** A handful of representative inputs you rerun after changes accelerates development.

> For deeper coverage, see: `ref-prompt-engineering.md`

## Why Evaluation Matters

Here's a common scenario: a team builds an LLM feature, tests manually, ships it, then discovers it fails on inputs they never tested. Complaints roll in. They patch it, breaking other cases. The cycle continues.

Evaluation catches regressions, proves improvements are real, and builds confidence across diverse inputs. Teams with good evaluation move faster—they make changes confidently and catch problems before users do.

Systematic evaluation involves test datasets (curated inputs with quality criteria), automated metrics (from simple format checks to LLM-as-judge evaluations), and infrastructure to run evaluations regularly and track results over time.

The path from vibe checks to systematic evaluation is ongoing. Your understanding of quality evolves, test datasets grow, and metrics become more sophisticated as you learn what matters for your application.

> For deeper coverage, see: `ref-evaluation-metrics.md`

## Key Tools

**Python** dominates AI Engineering. Most libraries and frameworks are Python-first. You need fluency with classes, async programming, type hints, and the package ecosystem.

**FastAPI** has emerged as the framework of choice—fast, modern, and natural with async patterns.

**LangChain** provides abstractions for common patterns: chains, RAG, tool use. **LangGraph** enables sophisticated workflows with state machines and persistent state.

**Vector databases** (Pinecone, Qdrant, Chroma) store embeddings for semantic search, enabling RAG applications.

**LangSmith** and similar tools provide observability—seeing every call, tracing chains, identifying failures.

> For tool-specific details, see: `ref-langchain-quickref.md`, `ref-langgraph-quickref.md`, `ref-qdrant-quickref.md`, `ref-langsmith-quickref.md`
