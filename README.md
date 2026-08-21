# 🧠 Next-Gen Retrieval-Augmented Generation (RAG) Ecosystem

Welcome to the **RAG Ecosystem** repository. This project is a comprehensive, hands-on exploration of state-of-the-art (SOTA) RAG architectures. It serves as an active research and development workspace for building, evaluating, and comparing different retrieval and generation paradigms—from basic vector search to advanced agentic systems.

## 🎯 Project Vision

The primary goal of this repository is to move beyond standard "hello world" LangChain tutorials. Here, we build modular, production-ready AI pipelines to understand how different components (Chunkers, Retrievers, Re-rankers, and Caching mechanisms) interact and perform under stress. 

By testing these systems against domain-specific documents (e.g., the Turkish Penal Code), we can mathematically evaluate their context precision, hallucination resistance, and latency.

---

## 📂 Repository Structure & Architectures

This repository is organized into isolated folders, each representing a distinct RAG architecture. 

### 1. [Hybrid Vector RAG (Current)](RAG/Hybrid/)
A highly optimized, standard-bearer RAG pipeline that combines semantic and keyword search.
* **Features:** Parent-Child Chunking, Qdrant Vector Store, Dense + Sparse (BM25) Retrieval, Reciprocal Rank Fusion (RRF), Cross-Encoder Re-ranking, and In-Memory/Semantic Caching.
* **Status:** Completed. Includes a custom evaluation script demonstrating the architectural strengths and the limits of small-parameter models.

### 2. GraphRAG *(Coming Soon)*
Transitioning from flat vector spaces to interconnected Knowledge Graphs.
* **Focus:** Extracting entities and relationships to answer complex, multi-hop reasoning questions that traditional vector searches fail to connect.
* **Tech Stack:** Neo4j, LangGraph.

### 3. CRAG (Corrective RAG) (Current)
Introducing an "Agentic" editor layer.
* **Focus:** The system will evaluate its own retrieved documents. If the confidence score is low, it will autonomously rewrite the query, re-search, or fallback to external web searches.
* **Tech Stack:** LangGraph Agent Routing, Tavily Search API.

### 4. RAPTOR *(Coming Soon)*
Recursive Abstractive Processing for Tree-Organized Retrieval.
* **Focus:** Solving the "needle in a haystack" problem for massive documents by creating hierarchical, tree-based summaries during the ingestion phase.
* **Tech Stack:** Milvus / pgvector, Recursive Clustering.

---

## 🛠️ Core Technologies Used

Across the various pipelines in this ecosystem, we utilize:
* **Orchestration:** LangChain, LangGraph
* **Vector Stores & Databases:** Qdrant, SQLite (Upcoming: Neo4j, Milvus, pgvector)
* **Local LLM Execution:** Llama.cpp (via `llama-cpp-python`)
* **Embedding & Re-ranking:** HuggingFace `multilingual-e5-small`, BAAI `bge-reranker-base`

## 📊 Evaluation Methodology

Building the system is only half the battle. Every architecture in this repository is subjected to a custom `RAGEvaluator`. We test for:
1. **Fact Retrieval:** Can the system find the exact legal article?
2. **Reasoning:** Can it connect concepts across multiple documents?
3. **Hallucination & Guardrails:** Does it gracefully say "I don't know" to absurd queries (e.g., "What is the penalty for theft on Mars?"), or does it break?

---
*Created by [Mehmetcan Tozlu](https://github.com/MehmetcanTozlu) - Exploring the boundaries of Generative AI and Information Retrieval.*
