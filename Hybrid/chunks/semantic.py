from typing import List
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from .base_chunker import BaseChunker


class SemanticTextChunker(BaseChunker):
    """
    Chunks documents using semantic text splitting.
    This method is useful for documents that have clear semantic boundaries, 
    such as research papers, legal documents, or articles. 
    """
    def __init__(self, embeddings, breakpoint_threshold_type: str = "percentile", breakpoint_threshold: float = 0.8):
        # breakpoint_threshold_type: "percentile", "standard_deviation", "interquartile"
        # may be best choice : "percentile"
        self.embeddings = embeddings
        self.splitter = (
            SemanticChunker(
                embeddings,
                breakpoint_threshold_type=breakpoint_threshold_type
            )
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents into chunks using semantic text splitting
        """
        try:
            if not documents:
                print("\033[93m[WARN] No documents provided for chunking.\033[0m")
                return []
            
            print(f"\033[90mUsing semantic text splitting: {len(documents)} documents\033[0m")
            chunks = self.splitter.split_documents(documents)
            
            return chunks
        
        except Exception as e:
            print(f"\033[91m[ERROR] Failed to split documents: {str(e)}\033[0m")
            return documents # return original documents if chunking fails
