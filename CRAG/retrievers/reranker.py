from typing import Any
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from .base_retriever import BaseRetriever

class CustomCrossEncoderReranker(BaseRetriever):
    """
    Re-ranker using CrossEncoder model.
    """
    def __init__(
        self,
        base_retriever: Any,
        model_name: str = "BAAI/bge-reranker-base",
        top_n: int = 3
    ):
        """
        :param base_retriever: Base retriever to use (e.g. HybridEnsembleRetriever)
        :param model_name: Name of the cross-encoder model for re-ranking
        :param top_n: Num of documents to return after re-ranking to the LLM
        """
        self.base_retriever = base_retriever
        self.model_name = model_name
        self.top_n = top_n

    def retriever(self):
        """
        Build and return re-ranker retriever with cross-encoder model
        """
        try:
            print(f"\033[94m Loading Cross-Encoder model for re-ranking {self.model_name}\033[0m")

            model = HuggingFaceCrossEncoder(model_name=self.model_name)

            # DÜZELTME 1: CrossEncoderReRanker değil, yukarıda import edilen CrossEncoderReranker
            compressor = CrossEncoderReranker(model=model, top_n=self.top_n)

            print(f"\033[90m(CrossEncoderReranker: Base Retriever results: {self.top_n})\033[0m")

            # DÜZELTME 2: Argüman 'compressor' değil, 'base_compressor' olmalıdır
            compression_retriever = ContextualCompressionRetriever(
                base_retriever=self.base_retriever,
                base_compressor=compressor 
            )
            
            return compression_retriever
        
        except ImportError as e:
            print(f"\033[91m[ERROR] Failed to build re-ranker retriever: {str(e)}\033[0m")
            return self.base_retriever
        
        except Exception as e:
            print(f"\033[91m[ERROR] Failed to build re-ranker retriever: {str(e)}\033[0m")
            print(f"\033[93m[WARN] Falling back to base retriever: {type(self.base_retriever).__name__}\033[0m")
            return self.base_retriever
