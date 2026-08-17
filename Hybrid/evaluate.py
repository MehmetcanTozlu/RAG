import time
from typing import Callable, Dict, Any


class RAGEvaluator:
    """
    Evaluates the performance of the RAG(Hybrid, GraphRAG, CRAG, RAPTOR) pipeline.
    """
    def __init__(self):
        self.test_cases = [
            # 1. Fact Retrieval
            {"q": "Kasten adam öldürmenin cezası nedir?", "type": "Gerçek Soru (Bilgi)"},
            {"q": "Kullanma hırsızlığı (Madde 146) nedir?", "type": "Gerçek Soru (Madde)"},
            
            # 2. Reasoning
            {"q": "Hırsızlık yaparken silah kullanmak cezayı nasıl etkiler?", "type": "Gerçek Soru (Çıkarım)"},
            
            # 3. Guardrails / Hallucination
            {"q": "Mars'ta hırsızlık yapmak kaç yıl ceza alır?", "type": "Tuzak Soru (Uzay/Bilim Kurgu)"},
            {"q": "Tavuklu pilav nasıl yapılır?", "type": "Tuzak Soru (Alakasız Konu)"}
        ]
    
    def evaluate_pipeline(self, pipeline_name: str, query_function: Callable[[str], str]):
        """
        Evaluate the RAG pipeline.
        
        Args:
            pipeline_name: The name of the pipeline to evaluate (e.g. Hybrid, GraphRAG, CRAG, RAPTOR).
            query_function: The function to use for querying. Takes a single argument (the query) and returns the answer.
        """
        print(f"\n\033[96m{'='*50}\033[0m")
        print(f"\033[96mEVALUATING PIPELINE: {pipeline_name.upper()}\033[0m")
        print(f"\033[96m{'='*50}\033[0m")

        total_time = 0

        for i, test in enumerate(self.test_cases):
            print(f"\n\033[94m--- Test {i+1}/{len(self.test_cases)}: {test['type']} ---\033[0m")
            print(f"\033[93mQuestion: {test['q']}\033[0m")

            start_time = time.time()
            try:
                answer = query_function(test['q'])
            except Exception as e:
                answer = f"[CRITICAL ERROR] Pipeline crashed: {str(e)}"
            
            latency = time.time() - start_time
            total_time += latency

            print(f"\033[92mAnswer: {answer}\033[0m")
            print(f"\033[93mLatency: {latency:.2f} seconds\033[0m")
            print("-" * 50)
        
        print(f"\n\033[95m[SUMMARY] {pipeline_name.upper()} Evaluation Completed. Total time: {total_time:.2f} seconds. Average Latency: {total_time / len(self.test_cases):.2f} seconds\033[0m")


if __name__ == '__main__':
    from Hybrid.app import setup_rag_pipeline

    print(f"Initializing RAG Pipeline...")

    ask_question, rag_engine = setup_rag_pipeline()

    print("RAG Pipeline Initialized!")
    
    evaluator = RAGEvaluator()

    evaluator.evaluate_pipeline("Hybrid Vector RAG (Qdrant + BM25)", ask_question)

    if hasattr(rag_engine, "client"):
        rag_engine.client.close()
    