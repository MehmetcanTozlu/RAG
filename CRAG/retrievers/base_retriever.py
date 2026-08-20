from abc import ABC, abstractmethod
from typing import Any


class BaseRetriever(ABC):
    """Absract base class for all retrievers.

    All retriever classes must inherit from this class and implement the
    retriever method, which takes a query as input and returns a list of
    retrieved documents.
    """
    
    @abstractmethod
    def retriever(self, query: str) -> Any:
        """Retrieve documents based on the query.

        Args:
            query: The query to retrieve documents for.

        Returns:
            A list of retrieved documents.
        """
        pass