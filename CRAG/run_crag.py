import time
import argparse
from llm_models import LLMEngine
from retriever_engine import RAG, clean_text
from langchain_community.document_loaders import PyPDFLoader

from chunks import (
    RecursiveChunker,
    FixedSizeChunker,
    MarkdownChunker,
    SemanticTextChunker,
    ParentChildChunker,
    AgenticChunker,
    LateChunker,
)

from retrievers import (
    DenseVectorRetriever,
    SparseBM25Retriever,
    HybridEnsembleRetriever,
    CustomCrossEncoderReranker,
    MultiQueryCustomRetriever,
)

from graph import create_crag_graph


def main():
    parser = argparse.ArgumentParser(description="CORRECTIVE RAG SYSTEM INITIALIZER")

    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/Llama-3.2-3B-Instruct-Q8_0.gguf",
        help="Path to the Language Model.",
    )
    parser.add_argument(
        "--pdf_path",
        type=str,
        default="./data/turk_ceza_kanunu.pdf",
        help="Path to the PDF Document.",
    )
    parser.add_argument(
        "--embed_model_path",
        type=str,
        default="./models/multilingual-e5-small",
        help="Path to the Embedding Model.",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="./data",
        help="Path to the Qdrant Data.",
    )
    parser.add_argument(
        "--reranker_model_path",
        type=str,
        default="./models/bge-reranker-base",
        help="Path to the Reranker Model.",
    )

    args = parser.parse_args()

    print("\n" + "\033[96m=" * 50 + "\033[0m")
    print(f"\033[96m--- CORRECTIVE RAG INITIALIZING ---\033[0m")
    print("\033[96m=" * 50 + "\033[0m\n")

    print(f"\033[94mInitializing LLM...\033[0m")
    llm_engine = LLMEngine(
        model_path=args.model_path,
        temperature=0.1,
        max_new_tokens=512,
    )
    print(f"\033[92m[SUCCESS] LLM Engine initialized!\033[0m")

    print(f"\033[94mRetrievers Initializing...\033[0m")
    chunker = LateChunker(chunk_size=1000)
    rag_engine = RAG(
        pdf_path=args.pdf_path,
        embed_model_path=args.embed_model_path,
        chunker=chunker,
        db_path=args.db_path,
        collection_name="tck_crag",
        n_retrieved=3,
    )

    loader = PyPDFLoader(args.pdf_path)
    raw_docs = clean_text(loader.load())
    bm25_chunks = chunker.split_documents(raw_docs)

    dense_base = DenseVectorRetriever(
        vector_store=rag_engine.vector_store,
        k=10,
    ).retriever()
    bm25_base = SparseBM25Retriever(
        documents=bm25_chunks,
        k=10,
    ).retriever()
    hybrid_retriever = HybridEnsembleRetriever(
        retrievers=[dense_base, bm25_base],
        weights=[0.4, 0.6]
    ).retriever()

    main_retriever = CustomCrossEncoderReranker(
        base_retriever=hybrid_retriever,
        model_name=args.reranker_model_path,
        top_n=3,
    ).retriever()
    print(f"\033[92m[SUCCESS] Retrievers Initialized!\033[0m")

    print(f"\033[94mCRAG Graph Initializing...\033[0m")
    crag_app = create_crag_graph(
        llm_engine=llm_engine,
        retriever=main_retriever,
    )
    print(f"\033[92m[SUCCESS] Agents Ready!\033[0m")

    print("\n" + "=" * 50)
    print("System passed interactive mode. Press 'q' to quit")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input(f"\033[94mUser: \033[0m").strip()
            if not user_input:
                continue
            if user_input.lower() in ["q", "quit", "exit"]:
                print("\n\033[93m[EXIT] Exiting CRAG system...\033[0m")
                break
            
            start_time = time.time()

            initial_state = {"question": user_input}

            print(f"\n\33[90m== AGENT ACTIVITY SCREEN ==\033[0m")
            final_state = None
            for output in crag_app.stream(initial_state):
                for node_name, state_value in output.items():
                    final_state = state_value
            
            final_answer = final_state.get("generation", "Answer not generated.")

            print(f"\n\033[92mCRAG System Answer: {final_answer.strip()}\033[0m")
            print(f"\033[90m(Total Time: {time.time() - start_time:.4f} seconds)\033[0m\n")

        except Exception as e:
            print(f"\n\033[91m[ERROR] An error occurred: {e}\033[0m")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
