# Local RAG & LLM Engine (100% Offline & Private)

A modular, production-ready **Retrieval-Augmented Generation (RAG)** pipeline and **Local LLM Inference Engine** running entirely on local hardware (CPU/GPU) without relying on external paid APIs (like OpenAI or Anthropic).

This project uses the **Turkish Penal Code (Türk Ceza Kanunu)** as a legal document dataset to demonstrate accurate local question-answering.

---

## 🚀 Key Features

* **100% Local & Privacy-First:** No data leaves your machine. Runs offline.
* **Dual LLM Backend Engine (`LLMEngine`):** Supports both **HuggingFace Transformers** and **Llama.cpp (GGUF Quantized)** models with automatic GPU offloading.
* **Smart Prompt Routing:** Automatically detects model types (e.g., Llama-3, Qwen-2.5) and applies the correct system/user instruction templates (`ChatML` or `Llama-3 Instruct`).
* **Vector Database (Qdrant):** Fast and persistent vector search using local embeddings (`multilingual-e5-small`).
* **Interactive RAG CLI:** Real-time retrieval inspection combined with local text generation.

---

## 📂 Project Structure

```text
rag_tutorial/
│
├── data/
│   ├── turk_ceza_kanunu.pdf     # Dataset (Turkish Penal Code)
│   └── ...                      # Qdrant local storage
│
├── utils/
│   └── llm_models.py            # Modular LLM Engine (GGUF & Transformers)
│
├── 1.py                         # SimpleRAG & Vector Database Retriever
├── app.py                       # Main End-to-End RAG Chat Application
└── requirements.txt             # Python dependencies
```

## 🛠️ Installation & Setup
1. Clone the Repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
