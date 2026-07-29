import os
import time
from utils.llm_models import LLMEngine

# Import the SimpleRAG class from the 1.py module
rag_module = __import__("1")
SimpleRAG = rag_module.SimpleRAG

def format_docs(docs):
    """
    This function converts LangChain Document objects into a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    """
    Main method to create a RAG pipeline and answer questions.
    """
    print("\n" + "=" * 50)
    print("\033[96m---RAG PIPELINE INITIALIZATION---\033[0m")
    print("\n" + "=" * 50)
    
    # 1. Start DB Engine (Retrieval)
    print(f"\033[94mStep 1: Starting DB Engine (Retrieval)...")
    rag_engine = SimpleRAG(
        path="", # document path
        embed_model_path="", # embedding model path
        db_path="", # database path for Qdrant
        collection_name="", # collection name
        chunk_size=500,
        chunk_overlap=50,
        n_retrieved=3,
    )
    print(f"\033[92m[SUCCESS] RAG Engine initialized.\033[0m")

    # 2. Start LLM Engine
    print(f"\033[94mStep 2: Starting LLM Engine...")
    llm_engine = LLMEngine(
        model_path="", # transformers or gguf model path
        temperature=0.1,
        max_new_tokens=512,
    )

    llm = llm_engine.get_llm()
    print(f"\033[92m[SUCCESS] LLM Engine initialized.\033[0m")

    # Prepare Prompt Template
    sys_message = (
        "Sen uzman bir Türk Hukuku asistanısın. Kullanıcıya "
        "SADECE sana verilen 'Bağlam (Context)' metinlerine dayanarak Türkçe cevap ver. "
        "Eğer cevap bağlamda yoksa, 'Bu bilgiye kanun metninden ulaşamıyorum' de. "
        "Kendi bilgini uydurma."
    )

    user_message = "Context:\n{context}\n\nQuestion: {question}"
    
    prompt = llm_engine.create_prompt(
        system_message=sys_message,
        user_message=user_message
    )
    rag_chain = prompt | llm

    # Chat Loop
    print("\n" + "="*50)
    print("SYSTEM IS READY! YOU CAN ASK QUESTIONS ABOUT THE TURKISH PENAL CODE.")
    print("If you want to exit you press 'q'.")
    print("="*50)

    while True:
        user_input = input(f"\033[94mYour Question: \033[0m ")

        if user_input.lower() in ["q", "quit", "exit"]:
            print("\033[90mGoodbye!")
            break
            
        if not user_input.strip():
            continue
        
        start_time = time.time()
        
        # Retrieval
        print("\033[90mScanning and retrieving context...\033[0m")
        docs = rag_engine.chunks_query_retriever.invoke(user_input)

        # Debug
        for i, doc in enumerate(docs):
            snippet = doc.page_content[:150].replace("\n", " ") + "..."
            print(f" {i+1}. {snippet}")
        print("\n" + "="*50)

        # Generation
        context_text = format_docs(docs)

        print("\033[93mGenerating Answer...\033[0m")
        response = rag_chain.invoke({
            "context": context_text,
            "question": user_input
        })

        print(f"\n[\033[92mAI Answer\033[0m]: {response.strip()}")
        print(f"\033[94mGeneration time: {time.time() - start_time:.2f} seconds\033[0m")

    if hasattr(rag_engine, "client"):
        rag_engine.client.close()


if __name__ == '__main__':
    main()
