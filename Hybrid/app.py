import time
from typing import List
from Hybrid.utils.llm_models import LLMEngine
from Hybrid.utils.cache_manager import SemanticCacheManager
from Hybrid.retriever_engine import RAG, clean_text

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

from Hybrid.pre_retrieval import (
    QueryRewriterAndRouter,
    HyDEGenerator,
    MultiQueryExpander,
)

from Hybrid.chunks import (
    RecursiveChunker,
    FixedSizeChunker,
    MarkdownChunker,
    SemanticTextChunker,
    ParentChildChunker,
    AgenticChunker,
    LateChunker,
)

from Hybrid.retrievers import (
    DenseVectorRetriever,
    SparseBM25Retriever,
    HybridEnsembleRetriever,
    CustomCrossEncoderReranker,
    MultiQueryCustomRetriever,
)


def setup_rag_pipeline():
    print("\n" + "=" * 50)
    print("\033[96m---RAG PIPELINE INITIALIZATION---\033[0m")
    print("\n" + "=" * 50)
    
    model_path = "./Llama-3.2-3B-Instruct-Q8_0.gguf"
    pdf_path = "./data/turk_ceza_kanunu.pdf"
    embed_model_path = "./multilingual-e5-small"
    db_path = "./data"
    embeddings = None
    cache_db_path = f"sqlite:///{db_path}/rag_cache.db"

    cache_manager = SemanticCacheManager(db_path=cache_db_path)

    # Initialize Cache
    print("\033[94mInitializing In-Memory Cache...\033[0m")
    set_llm_cache(InMemoryCache())
    
    # 1. Initialize LLM Engine
    print(f"\033[94mStarting LLM Engine...")
    llm_engine = LLMEngine(
        model_path=model_path,
        temperature=0.1,
        max_new_tokens=512,
        n_ctx=2048,
    )
    llm = llm_engine.get_llm()
    print(f"\033[92m[SUCCESS] LLM Engine is ready.\033[0m")
    
    # 2. Initialize Ingestion and Retrieval components
    print(f"\n\033[94mStarting Data Engine..\033[0m")

    # =========================================================================
    # 1. CHUNKING STRATEGIES
    # =========================================================================

    # 1. Recursive
    # chunker = RecursiveChunker(
    #     chunk_size=500,
    #     chunk_overlap=50
    # )

    # 2. Fixed Size
    # chunker = FixedSizeChunker(
    #     chunk_size=500,
    #     chunk_overlap=50
    # )

    # 3. Markdown
    # chunker = MarkdownChunker()

    # 4. Semantic
    # embeddings = HuggingFaceEmbeddings(
    #     model_name=embed_model_path,
    #     model_kwargs={'device': 'cuda'},
    #     encode_kwargs={'normalize_embeddings': True},
    # )
    # chunker = SemanticTextChunker(
    #     embeddings=embeddings,
    #     breakpoint_threshold_type="percentile",
    #     breakpoint_threshold=0.8,
    # )

    # 5. Parent-Child
    # chunker = ParentChildChunker(
    #     parent_chunk_size=1500,
    #     child_chunk_size=300,
    #     parent_chunk_overlap=150,
    #     child_chunk_overlap=50,
    # )

    # 6. Agentic
    chunker = AgenticChunker(
        max_chunk_size=600,
        overlap=50,
    )

    # 7. Late Chunking
    # chunker = LateChunker(
    #     chunk_size=1000,
    # )
    # =========================================================================

    rag_engine = RAG(
        pdf_path=pdf_path,
        embed_model_path=embed_model_path,
        chunker=chunker,
        db_path=db_path,
        collection_name="tck_parent_child_v1",
        n_retrieved=3,
        embedder=embeddings if embeddings is not None else None,
    )
    print(f"\033[92m[SUCCESS] Data Ingested loaded.\033[0m")

    # 3. Initialize Retrievers
    print(f"\033[94mConfigurring Advanced Retrievers...\033[0m")

    # =========================================================================
    # 2. RETRIEVAL STRATEGIES
    # =========================================================================

    loader = PyPDFLoader(pdf_path)
    raw_docs = clean_text(loader.load())

    if isinstance(chunker, ParentChildChunker):
        print(f"\033[93m[INFO] Parent-child chunker detected. Using auto-merging...\033[0m")
        parent_splitter, child_splitter = chunker.get_splitter()
        bm25_chunks = child_splitter.split_documents(raw_docs)
    else:
        bm25_chunks = chunker.split_documents(raw_docs)

    dense_base = DenseVectorRetriever(vector_store=rag_engine.vector_store, k=10).retriever()
    bm25_base = SparseBM25Retriever(documents=bm25_chunks, k=10).retriever() # Ingest edilen ham dökümanlar verilir

    # 1. Dense Vector
    # main_retriever = dense_base

    # 2. Sparse BM25
    # main_retriever = bm25_base

    # 3. Hybrid / Ensemble
    # main_retriever = HybridEnsembleRetriever(
    #     retrievers=[dense_base, bm25_base],
    #     weights=[0.4, 0.6],
    # ).retriever()

    # 4. Re-Ranker
    hybrid_retriever = HybridEnsembleRetriever(
        retrievers=[dense_base, bm25_base], 
        weights=[0.4, 0.6]
    ).retriever()
    main_retriever = CustomCrossEncoderReranker(
        base_retriever=hybrid_retriever, #dense_base
        model_name="./bge-reranker-base", # reranker model path
        top_n=3,
    ).retriever()

    # 5. Multi-Query
    # main_retriever = MultiQueryCustomRetriever(
    #     base_retriever=dense_base,
    #     llm=llm,
    # ).retriever()
    # =========================================================================
    print("\033[92m[SUCCESS] Retriever configured.")

    # 4. Prompt Engineering (Set system prompt for "I don't know")
    sys_message = (
        "Sen katı kurallara bağlı, profesyonel bir Türk Hukuku Asistanısın.\n"
        "AŞAĞIDAKİ KURALLARA KESİNLİKLE UYMAK ZORUNDASIN:\n"
        "1. Kullanıcının sorusunu SADECE sana <baglam> etiketleri arasında verilen metinleri kullanarak yanıtla.\n"
        "2. Eğer sorunun cevabı <baglam> içindeki metinlerde AÇIKÇA YER ALMIYORSA (örneğin uzay, gezegenler, alakasız isimler veya bağlam dışı konular içeriyorsa), "
        "kendi bilgilerini KESİNLİKLE KULLANMA ve sadece şu standart cümleyi söyle: 'Bu bilgiye kanun metninden ulaşamıyorum.'\n"
        "3. Asla tahmin etme, uydurma (halüsinasyon) veya dış bilgi kullanma."
    )
    
    user_message = "<baglam>\n{context}\n</baglam>\n\nKullanıcının Sorusu: {question}\nCevap:"
    
    prompt = llm_engine.create_prompt(
        system_message=sys_message,
        user_message=user_message,
    )

    # 5. Build RAG Chain
    rag_chain = prompt | llm

    # =========================================================================
    # EVALUATION
    # =========================================================================
    def ask_question(query: str) -> str:
        """
        Answers the given query using the RAG pipeline.
        """
        # =========================================================================
        # PRE-RETRIEVAL
        # =========================================================================
        # print("\033[94m[QUERY] Pre-Retrieval..\033[0m")

        # 1. QueryRewriterAndRouter
        # rewriter = QueryRewriterAndRouter(llm_engine=llm_engine)
        # optimized_query = rewriter.process_query(query)
        # if "[ALAKASIZ]" in optimized_query.upper():
        #     print(f"\n[\033[93mSystem Warning\033[0m] Question {query} is irrelevant.")
        #     return "Bu bilgiye kanun metninden ulaşamıyorum."
        # print(f"\n[\033[96mOptimized Query\033[0m] {optimized_query}")

        # 2. HyDE
        # hyde_generator = HyDEGenerator(llm_engine=llm_engine)
        # fake_document = hyde_generator.generate(query)
        # optimized_query = f"{query}\n{fake_document}"

        # 3. Multi-Query
        # expander = MultiQueryExpander(llm_engine=llm_engine)
        # search_queries = expander.generate(query)
        # print(f"\033[95m[Generated Queries]\033[0m")
        # for i, sq in enumerate(search_queries):
        #     print(f"  \033[90m> {sq}\033[0m")
        # optimized_query = "\n".join(search_queries)
        # =========================================================================

        search_query = query # If pre-retrievel is enabled, use `optimized_query` instead.

        print(f"\n\033[90mScanning document..\033[0m")
        retrieved_docs = main_retriever.invoke(search_query)

        context_texts = []
        for i, doc in enumerate(retrieved_docs):
            snippet = doc.page_content.replace('\n', ' ')
            context_texts.append(snippet)
            print(f"  \033[90m> Found Part {i+1}:\033[0m {snippet[:150]}...")
        
        combined_context = "\n\n---\n\n".join(context_texts)

        print(f"\033[93m[*] Generating Answer (Llama-3 is thinking)...\033[0m")
        response = rag_chain.invoke({
            "context": combined_context,
            "question": query,
        })

        return response.strip()

    return ask_question, rag_engine


def main():
    """
    Main method for interactive terminal chat.
    """
    ask_question_fn, rag_engine = setup_rag_pipeline()

    print("\n" + "=" * 50)
    print("\033[96m---RAG PIPELINE INITIALIZATION SUCCESSFUL---\033[0m")
    print("Type 'q' to exit.")
    print("=" * 50 + "\n")

    # 2. Interactive Loop
    while True:
        try:
            user_input = input(f"\033[94mYour Question: \033[0m ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["q", "quit", "exit"]:
                print("\033[93mGoodbye! Exiting RAG system...\033[0m")
                break
            
            start_time = time.time()

            answer = ask_question_fn(user_input)

            print(f"\n[\033[92mAI Answer\033[0m]: {answer}")
            print(f"\033[90m(Total Latency: {time.time() - start_time:.2f}s)\033[0m")
        
        except KeyboardInterrupt:
            print("\n\033[90mInterrupted by user. Exiting...\033[0m")
            break
        except Exception as e:
            print(f"\033[91mAn error occurred: {e}\033[0m")
            
    if hasattr(rag_engine, "client"):
        rag_engine.client.close()

if __name__ == '__main__':
    main()