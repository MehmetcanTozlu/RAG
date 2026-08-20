from typing import List, Any
from langchain_classic.retrievers import EnsembleRetriever
from .base_retriever import BaseRetriever


class HybridEnsembleRetriever(BaseRetriever):
    """
    A hybrid retriever that combines multiple retrieval strategies.
    Initializes with a list of retriever objects and combines their results
    using EnsembleRetriever from langchain.retrievers.
    """
    def __init__(self, retrievers: List[Any], weights: List[float] = None):
        """
        :param retrievers: Retrievers for combining (e.g. [dense_ret, bm25_ret])
        :param weights: Weights for each retriever.
        """
        self.retrievers = retrievers

        if weights is None:
            self.weights = [1.0 / len(retrievers)] * len(retrievers)
        else:
            self.weights = weights
        
    def retriever(self):
        """
        Build and return ensemble retriever with weights
        """
        try:
            if not self.retrievers:
                raise ValueError("Retriever list cannot be empty!")
            
            print(f"\033[90m(HybridEnsembleRetriever: Combining {len(self.retrievers)} retrievers)\033[0m")

            ensemble_retriever = EnsembleRetriever(
                retrievers=self.retrievers,
                weights=self.weights
            )

            return ensemble_retriever
            
        except Exception as e:
            print(f"\033[91m[ERROR] Failed to build ensemble retriever: {str(e)}\033[0m")
            
            if self.retrievers: # Return first retriever if available
                print(f"\033[93m[WARN] Falling back to first retriever: {type(self.retrievers[0]).__name__}\033[0m")
                return self.retrievers[0]
            
            raise
