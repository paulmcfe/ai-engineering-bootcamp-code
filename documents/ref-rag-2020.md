# Paper Summary: Retrieval-Augmented Generation (RAG)

## Citation

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. *Advances in Neural Information Processing Systems*, 33, 9459-9474.

## One-Sentence Summary

RAG combines parametric memory (a pretrained language model) with non-parametric memory (a retrieval index) to generate responses grounded in retrieved documents.

## The Problem

Language models store knowledge in their parameters, learned during pretraining. This approach has fundamental limitations:

1. **Knowledge is static.** Once trained, the model can't learn new information without retraining.
2. **Knowledge is implicit.** You can't easily inspect or update what the model "knows."
3. **Hallucination risk.** The model may generate plausible-sounding but incorrect information.
4. **Scale limitations.** Storing more knowledge requires larger models.

## The Solution

RAG augments a language model with a retrieval mechanism:

```
Query → Retriever → Relevant Documents → Generator → Response
```

The key insight: instead of asking the model to recall everything from its parameters, give it relevant documents at generation time. The model's job becomes synthesizing and reasoning over retrieved information rather than pure recall.

## Architecture

RAG consists of two components:

**1. Retriever (DPR - Dense Passage Retrieval)**
- Encodes documents and queries into dense vectors
- Uses FAISS for efficient similarity search
- Retrieves top-k relevant documents for each query

**2. Generator (BART)**
- Sequence-to-sequence model
- Conditions generation on both the query and retrieved documents
- Two variants:
  - **RAG-Sequence**: Uses same documents for entire output
  - **RAG-Token**: Can use different documents for each token

## Training

RAG is trained end-to-end:
- The retriever and generator are trained jointly
- No direct supervision for which documents to retrieve
- The model learns to retrieve documents that help answer questions

This is key: the system learns what to retrieve based on what helps generation, not based on human-labeled relevance.

## Key Results

RAG outperformed pure parametric models on knowledge-intensive tasks:

| Task | Previous SOTA | RAG |
|------|---------------|-----|
| Natural Questions | 44.5 | 44.5 |
| TriviaQA | 68.1 | 56.1 |
| MS-MARCO | 33.8 | 33.6 |

More importantly, RAG showed:
- Better factual accuracy
- Ability to update knowledge by changing the index
- More interpretable outputs (you can see what was retrieved)

## Why This Matters for AI Engineering

RAG established the foundational pattern used in most production LLM applications today:

1. **Retrieval-then-generate** became the standard architecture for knowledge-grounded systems
2. **Vector similarity search** became the primary retrieval mechanism
3. **Dense embeddings** became how we represent text for search

Every RAG system you build follows this paper's core insight: augment generation with retrieval.

## Practical Implications

**What the paper got right:**
- Combining retrieval with generation dramatically improves factuality
- End-to-end training can learn useful retrieval without explicit labels
- The architecture scales with index size, not model size

**What's evolved since:**
- Modern embeddings (OpenAI, Cohere) outperform DPR
- Larger context windows reduce need for aggressive retrieval
- Chunking and indexing strategies have become more sophisticated
- Reranking and multi-stage retrieval improve quality

## Connection to StudyBuddy

StudyBuddy v2 implements the core RAG pattern:
1. Index study materials (the retrieval index)
2. Embed user questions (query encoding)
3. Retrieve relevant chunks (similarity search)
4. Generate explanations grounded in retrieved content

The difference: StudyBuddy uses a more modular approach (separate embedding and generation models) rather than end-to-end training.

## Key Quotes

> "RAG models combine the best of both worlds: they can access knowledge from a non-parametric memory while maintaining the generation capabilities of parametric models."

> "Unlike purely parametric approaches, the knowledge can be updated or augmented by simply modifying the index."

## Further Reading

- **Dense Passage Retrieval** (Karpukhin et al., 2020): The retriever architecture
- **BART** (Lewis et al., 2020): The generator architecture
- **REALM** (Guu et al., 2020): Related approach with pretrained retriever

## BibTeX

```bibtex
@article{lewis2020retrieval,
  title={Retrieval-augmented generation for knowledge-intensive nlp tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K{\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rockt{\"a}schel, Tim and others},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={9459--9474},
  year={2020}
}
```
