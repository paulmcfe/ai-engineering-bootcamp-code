# Prompt Engineering

## What Is Prompt Engineering?

Prompt engineering is the practice of designing inputs to language models to get desired outputs. It's how you communicate with LLMs—not just what you ask, but how you ask it.

Good prompts are clear, specific, and structured. They provide context, examples, and constraints that guide the model toward useful responses. Poor prompts are vague, ambiguous, or missing critical information.

## Why Prompts Matter

LLMs are trained to predict likely continuations of text. Your prompt sets up the context that determines what "likely" means. A prompt like "Write about dogs" could go anywhere. A prompt like "Write a 200-word blog post about training golden retrievers to fetch, aimed at first-time dog owners" constrains the output space dramatically.

The same model can produce wildly different outputs based on prompting. Prompt engineering is often the highest-leverage improvement you can make to an AI system—more impactful than model choice or complex architectures.

## Prompt Structure

### System Prompt

Defines who the model is and how it should behave. Persists across the conversation.

```python
system_prompt = """You are a helpful coding assistant specializing in Python.

Guidelines:
- Write clean, well-documented code
- Explain your reasoning
- Suggest best practices
- Ask clarifying questions when requirements are unclear

Constraints:
- Only use standard library unless user specifies otherwise
- Prefer readability over cleverness
- Include error handling"""
```

### User Prompt

The actual request or question. Changes with each turn.

```python
user_prompt = "Write a function to parse CSV files and return a list of dictionaries."
```

### Assistant Response

What the model generates. Can be pre-filled to guide output format.

```python
# Pre-fill to enforce format
assistant_prefix = "```python\n"
```

## Core Techniques

### Be Specific

Vague prompts get vague responses.

```
❌ "Tell me about databases"
✅ "Explain the difference between SQL and NoSQL databases, focusing on when to use each, with examples"
```

### Provide Context

Give the model the information it needs.

```
❌ "Fix this bug"
✅ "Fix this bug: the function returns None when the input list is empty. 
    Expected: return an empty list instead.
    
    def process(items):
        if items:
            return [x * 2 for x in items]"
```

### Specify Format

Tell the model how to structure its response.

```python
prompt = """List the top 5 machine learning frameworks.

Format each as:
- **Name**: Brief description
- Pros: List 2-3 advantages
- Cons: List 1-2 disadvantages
- Best for: Primary use case"""
```

### Use Examples (Few-Shot)

Show the model what you want through examples.

```python
prompt = """Convert natural language to SQL.

Examples:
User: Show all users from California
SQL: SELECT * FROM users WHERE state = 'California'

User: Count orders from last month  
SQL: SELECT COUNT(*) FROM orders WHERE order_date >= DATE_SUB(NOW(), INTERVAL 1 MONTH)

User: Find the average price of products in the electronics category
SQL:"""
```

### Chain of Thought

Ask the model to reason step by step.

```python
prompt = """Solve this problem step by step:

A store sells apples for $1.50 each and oranges for $2.00 each. 
If Maria buys 4 apples and some oranges, spending exactly $14.00, 
how many oranges did she buy?

Think through this step by step:
1. First, calculate the cost of apples
2. Then, determine remaining budget
3. Finally, calculate number of oranges"""
```

### Role Assignment

Give the model a persona that shapes its responses.

```python
prompt = """You are a senior software architect with 20 years of experience.
A junior developer asks: "Should I use microservices or a monolith for my new project?"

Provide advice as an experienced architect would, considering trade-offs and asking clarifying questions."""
```

## Advanced Techniques

### Self-Consistency

Generate multiple responses and aggregate.

```python
def self_consistent_answer(query: str, n: int = 5) -> str:
    responses = []
    for _ in range(n):
        response = llm.invoke(query, temperature=0.7)
        responses.append(response)
    
    # Take majority answer or consensus
    return find_consensus(responses)
```

### Constrained Generation

Use structured output to enforce format.

```python
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str
    rating: int  # 1-5
    summary: str
    pros: list[str]
    cons: list[str]

response = client.responses.create(
    model="gpt-4o-mini",
    input="Review the movie Inception",
    response_format={"type": "json_object"}
)
```

### Decomposition

Break complex tasks into subtasks.

```python
def complex_analysis(topic: str) -> str:
    # Step 1: Gather facts
    facts = llm.invoke(f"List 10 key facts about {topic}")
    
    # Step 2: Identify perspectives
    perspectives = llm.invoke(f"What are different viewpoints on {topic}?")
    
    # Step 3: Synthesize
    synthesis = llm.invoke(f"""Given these facts:
{facts}

And these perspectives:
{perspectives}

Write a balanced analysis of {topic}.""")
    
    return synthesis
```

### Iterative Refinement

Generate, critique, and improve.

```python
def refine_output(query: str, iterations: int = 2) -> str:
    response = llm.invoke(query)
    
    for _ in range(iterations):
        critique = llm.invoke(f"""Critique this response:
{response}

Identify weaknesses and suggest improvements.""")
        
        response = llm.invoke(f"""Improve this response based on the critique:

Original: {response}
Critique: {critique}

Write an improved version.""")
    
    return response
```

## Prompts for Specific Tasks

### Summarization

```python
prompt = """Summarize the following text in 3 bullet points.
Focus on: main argument, key evidence, and conclusion.

Text:
{document}

Summary:"""
```

### Classification

```python
prompt = """Classify the following customer message into one category.

Categories:
- billing: Payment, charges, refunds
- technical: Product issues, bugs, errors  
- account: Login, password, profile
- general: Everything else

Message: "{customer_message}"

Classification:"""
```

### Extraction

```python
prompt = """Extract the following information from the text:
- Person names
- Organizations
- Dates
- Locations

Text: "{text}"

Return as JSON:
{{"names": [], "organizations": [], "dates": [], "locations": []}}"""
```

### Code Generation

```python
prompt = """Write a Python function with the following specifications:

Function name: {name}
Purpose: {description}
Parameters: {params}
Returns: {returns}
Example usage: {example}

Requirements:
- Include docstring
- Add type hints
- Handle edge cases
- Include basic error handling"""
```

## Common Mistakes

### Too Vague

```
❌ "Make this better"
✅ "Improve this paragraph by: 1) making it more concise, 2) using active voice, 3) adding a specific example"
```

### No Examples for Complex Tasks

```
❌ "Convert data to the right format"
✅ "Convert this data to JSON format. Example:
    Input: name=John, age=30
    Output: {\"name\": \"John\", \"age\": 30}"
```

### Conflicting Instructions

```
❌ "Be concise but thorough. Keep it short but don't miss anything."
✅ "Provide a 2-3 paragraph summary covering the main points."
```

### Missing Constraints

```
❌ "Write some code to do this"
✅ "Write Python 3.11 code using only standard library. Include error handling and comments."
```

## Debugging Prompts

### Check Understanding

Add "Before answering, explain your understanding of the task."

### Request Reasoning

Add "Explain your reasoning step by step."

### Ask for Confidence

Add "Rate your confidence in this answer (1-10) and explain why."

### Test Edge Cases

Run prompts with unusual inputs to find weaknesses.

### Compare Variations

Test multiple phrasings to find what works best.

```python
prompt_variations = [
    "Summarize this text",
    "Write a brief summary of this text",
    "What are the main points of this text?",
    "TL;DR this text in 2-3 sentences"
]

for prompt in prompt_variations:
    response = llm.invoke(f"{prompt}\n\n{text}")
    evaluate_quality(response)
```

## Temperature and Parameters

**Temperature** controls randomness:
- 0.0: Deterministic, always picks most likely token
- 0.7: Balanced creativity and coherence
- 1.0+: More random, creative, sometimes incoherent

**Max tokens** limits response length. Set based on expected output size.

**Top-p (nucleus sampling)** alternative to temperature for controlling diversity.

```python
# Factual task: low temperature
response = client.responses.create(
    model="gpt-4o-mini",
    input="What is the capital of France?",
    temperature=0.0
)

# Creative task: higher temperature  
response = client.responses.create(
    model="gpt-4o-mini",
    input="Write a short poem about coding",
    temperature=0.8
)
```

## Prompt Templates

Use templates for consistency and reusability:

```python
from string import Template

qa_template = Template("""Answer the question based on the context.

Context:
$context

Question: $question

Answer:""")

prompt = qa_template.substitute(
    context=retrieved_docs,
    question=user_question
)
```

Or with LangChain:

```python
from langchain.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specializing in {domain}."),
    ("user", "{query}")
])

prompt = template.format(domain="Python", query="How do decorators work?")
```

## Best Practices

1. **Start simple, add complexity.** Begin with basic prompts, add structure as needed.

2. **Be explicit about format.** Don't assume the model knows your preferred output structure.

3. **Use examples.** Few-shot prompting is remarkably effective.

4. **Test systematically.** Run prompts against diverse inputs.

5. **Version your prompts.** Track changes like code—prompts are part of your system.

6. **Consider edge cases.** What happens with empty input? Very long input? Unusual requests?

7. **Iterate based on failures.** When outputs are wrong, analyze why and adjust.

## Related Concepts

- **Chain of Thought**: Reasoning technique in prompts
- **Few-Shot Learning**: Using examples in prompts
- **Agents**: Systems built on sophisticated prompting
- **Tool Use**: Prompts that enable function calling
- **RAG**: Prompts that incorporate retrieved context
