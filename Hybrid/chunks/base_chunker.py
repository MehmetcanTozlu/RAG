from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class BaseChunker(ABC):
    """
    Base class for all chunker implementations
    All chunker must inherit from this class and implement the chunk method
    """

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Take list of documents and return list of chunks
        """
        pass