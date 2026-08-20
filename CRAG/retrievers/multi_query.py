import logging
from typing import Any
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from .base_retriever import BaseRetriever


# See to LLM's generated queries for the original query.
logging.basicConfig()
logger = logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


class MultiQueryCustomRetriever(BaseRetriever):
    """
    Take original query and generate multiple queries to be used for the RAG pipeline.
    Then fetch documents using the base retriever.
    """
    def __init__(self, base_retriever: Any, llm: Any):
        """
        :param base_retriever: Base retriever to be used for the RAG pipeline
        :param llm: Language model to generate multiple queries
        """
        self.base_retriever = base_retriever
        self.llm = llm
    
    def retriever(self):
        """
        Build and return retriever with multi query retriever
        """
        try:
            print("\033[94m Loading Multi Query Retriever\033[0m")
            
            retriever_from_llm = MultiQueryRetriever.from_llm(
                llm=self.llm,
                retriever=self.base_retriever
            )

            return retriever_from_llm
        
        except Exception as e:
            print(f"\033[91m[ERROR] Failed to build multi query retriever: {str(e)}\033[0m")
            print(f"\033[93m[WARN] Falling back to base retriever: {type(self.base_retriever).__name__}\033[0m")
            return self.base_retriever
