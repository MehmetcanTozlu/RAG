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
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```
## 2. Create and Activate Virtual Environment
```bash
python3 -m venv rag_tutorial
source rag_tutorial/bin/activate
```
## 3. Install Dependencies
(Note: If you want GPU acceleration for Llama.cpp, make sure to install with CUDA support)
```bash
pip install -r requirements.txt
```
## 4. Download Required Models
- LLM: Download a GGUF model (e.g., Llama-3.2-3B-Instruct.gguf or Qwen-2.5-Coder) and place it in your local models directory.
- Embedding Model: Download intfloat/multilingual-e5-small locally.

## 💻 Usage
```bash
python app.py
```

#### The system will:
1. Connect to the local Qdrant vector database (ingesting the PDF if it's the first run).
2. Load your local GGUF model into VRAM/RAM.
3. Allow you to ask questions interactively with live context retrieval logs.
4. 
## 🛡️ Tech Stack
- Orchestration: LangChain
- Vector DB: Qdrant
- Embeddings: HuggingFace (multilingual-e5-small)
- LLM Runner: llama-cpp-python (C++ optimized for local GPUs)
