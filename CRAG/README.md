# 🕸️ CRAG (Corrective Retrieval-Augmented Generation) Pipeline

An advanced, Agentic RAG implementation using **LangGraph** to introduce self-reflection, grading, and automated web-search fallbacks. 

Standard RAG systems blindly pass retrieved documents to the LLM, often leading to hallucinations if the retrieval step fails. **CRAG (Corrective RAG)** solves this by introducing a "Grader Agent" that evaluates the retrieved context before generation. If the context is deemed irrelevant, the system autonomously queries the internet to fetch up-to-date, relevant information.

## 🧠 System Architecture (LangGraph Workflow)

This pipeline is modeled as a State Machine (Graph) rather than a linear chain.

1. **Retrieve Node:** Fetches the top-K documents from the local Qdrant Vector Store using a Hybrid Ensemble Retriever (Dense + Sparse BM25) and a Cross-Encoder Re-ranker.
2. **Grade Documents Node (The Judge):** The LLM evaluates each document against the user's query. It strict-matches `[UYGUN]` (Relevant) or `[YETERSİZ]` (Irrelevant).
3. **Conditional Router:** 
   * If at least one document is relevant ➡️ Route to **Generate Node**.
   * If ALL documents are irrelevant ➡️ Route to **Web Search Node**.
4. **Web Search Node (Fallback):** Uses the `DuckDuckGoSearchRun` tool to scrape the web for the missing information and appends it to the context state.
5. **Generate Node:** Produces the final answer using the validated local documents or the freshly injected web data, heavily guarded against hallucinations.

## 🛠️ Tech Stack
* **Orchestration:** LangGraph (StateGraph, Conditional Edges)
* **LLM Engine:** Local `Llama-3.2-3B-Instruct` via `llama-cpp-python`
* **Web Search API:** DuckDuckGo
* **Vector Store:** Qdrant (SQLite-backed)

## 🚀 How to Run

The system is fully modular and configurable via CLI arguments using `argparse`.

**Run with default paths:**
```bash
python run_crag.py
```

```bash
python run_crag.py \
  --model_path "/path/to/your/model.gguf" \
  --pdf_path "/path/to/document.pdf" \
  --embed_model_path "/path/to/embedding/model"
```