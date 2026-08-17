from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from .base_retriever import BaseRetriever


class SparseBM25Retriever(BaseRetriever):
    """
    A sparse retriever that uses BM25 for keyword-based retrieval.
    """
    def __init__(self, documents: List[Document], k: int = 3):
        self.documents = documents
        self.k = k
        self._bm25_retriever = self._build_bm25()
    
    def _build_bm25(self):
        try:
            if not self.documents:
                print("\033[93m[WARN] No documents provided for BM25 retriever.\033[0m")
                return None
            
            print(f"\033[90m(SparseBM25Retriever: Indexing for {len(self.documents)} documents)\033[0m")

            retriever = BM25Retriever.from_documents(self.documents)
            retriever.k = self.k

            return retriever
        
        except Exception as e:
            print(f"\033[91m[ERROR] Failed to build BM25 retriever: {str(e)}\033[0m")
            raise
    
    def retriever(self):
        """
        Return a retriever that can be used to retrieve documents.
        """
        return self._bm25_retriever
