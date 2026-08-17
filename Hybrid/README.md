# 🚀 Advanced Hybrid RAG Pipeline: Turkish Penal Code (TCK)

A State-of-the-Art (SOTA) Retrieval-Augmented Generation (RAG) pipeline designed to query the Turkish Penal Code (TCK). This project demonstrates production-ready architectural patterns, combining multiple retrieval strategies, advanced chunking methodologies, and caching mechanisms to maximize context precision and reduce latency.

## 🏗️ System Architecture

This pipeline moves beyond standard LangChain tutorials by implementing an enterprise-grade ingestion and retrieval flow:

### 1. Ingestion & Chunking (Data Engine)
* **Parent-Child Chunking (Auto-Merging):** Preserves semantic context by splitting documents into small "Child" chunks for precise vector searching, while feeding the larger encompassing "Parent" chunk to the LLM.
* **Agentic & Late Chunking:** Modular support for logical boundary detection and Jina AI's late chunking coordination.

### 2. Retrieval Strategy (Hybrid + Reranking)
* **Dense Vector Search:** Qdrant Vector Store with `multilingual-e5-small` embeddings for semantic matching.
* **Sparse BM25 Search:** Exact-keyword matching engine, crucial for legal documents and specific penal code articles.
* **Hybrid Ensemble:** Combines Dense and Sparse results using Reciprocal Rank Fusion (RRF) (40% Dense, 60% Sparse).
* **Cross-Encoder Re-ranking:** Filters the top 10 hybrid results down to the absolute best 3 using `BAAI/bge-reranker-base`.

### 3. Latency Optimization
* **In-Memory / Semantic Caching:** Bypasses the LLM for exact or semantically similar repeated queries, reducing response times from seconds to `0.01s`.

### 4. Post-Retrieval Guardrails
* Strict prompt engineering designed to enforce `[I DON'T KNOW]` behaviors for out-of-domain queries (e.g., non-legal or sci-fi questions).

---

## 🧪 Evaluation & Known Limitations (Benchmark)

We built a custom modular `RAGEvaluator` to test the pipeline against Fact Retrieval, Reasoning, and Hallucination limits. The current tests were run locally using a small-parameter uncensored model (`Llama-3.2-3B-Instruct-Q8_0.gguf`). 

While the **Retrieval Engine performed flawlessly** in finding the correct legal context, the benchmark exposed the hardware and architectural limits of using a 3B parameter model:

* **Hardware Constraints (VRAM Overload):** Heavy context windows in Q8 quantization caused CPU offloading, resulting in massive initial latency spikes (up to 15 minutes) before caching stabilized it to ~26 seconds.
* **Model Degeneration:** Complex reasoning tasks caused the 3B model to lose focus, occasionally generating foreign characters (Chinese/Japanese) at the end of long generations.
* **Guardrail Failure & Repetition Trap:** As an uncensored small model, it failed to follow negative constraints (e.g., "Do not answer if not in context"). When asked about "Chicken Rice Recipe" or "The penalty for theft on Mars", it either hallucinated a logical-sounding legal penalty or fell into an infinite text-repetition loop.

**Conclusion:** The RAG architecture is highly robust. To deploy to production, the local 3B model should be swapped with a larger model (e.g., Llama-3-8B) or a cloud API (OpenAI/Anthropic) capable of adhering to strict negative constraints.

---

## 🗺️ Future Roadmap
This repository serves as the foundation for exploring Next-Gen RAG architectures. Upcoming implementations include:
- [ ] **CRAG (Corrective RAG):** Implementing a routing agent to self-correct and fallback to web search if retrieved documents are irrelevant.
- [ ] **GraphRAG:** Extracting entities and relationships (Neo4j) for multi-hop legal reasoning.
- [ ] **RAPTOR:** Tree-organized retrieval using Milvus/pgvector for whole-document summarization.