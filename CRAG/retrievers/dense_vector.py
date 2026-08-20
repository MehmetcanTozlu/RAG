from .base_retriever import BaseRetriever


class DenseVectorRetriever(BaseException):
    """
    A dense vector retriever that uses sentence transformers to embed queries
    and documents and performs cosine similarity search.
    """
    
    def __init__(self, vector_store, k: int = 3):
        """Initialize the retriever.

        Args:
            vector_store: The vector store to use for retrieval.
            k: The number of documents to retrieve.
        """
        self.vector_store = vector_store
        self.k = k
    
    def retriever(self):
        """
        Return a retriever that can be used to retrieve documents.
        """
        print(f"\033[90m(DenseVectorRetriever: Searching for {self.k} most relevant documents...)\033[0m")
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k},
        )
