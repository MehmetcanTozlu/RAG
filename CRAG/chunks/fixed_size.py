from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from .base_chunker import BaseChunker
from .utils import get_token_length


class FixedSizeChunker(BaseChunker):
    """
    Fixed size chunker implementation
    Splits text into fixed-size chunks with overlap
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = CharacterTextSplitter(
            separator="",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=get_token_length,
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents using fixed size text splitter
        """
        try:
            if not documents:
                print(f"\033[93m[WARN] No documents provided for chunking.\033[0m")
                return []
            
            print(f"\033[90mUsing fixed size text splitting: {len(documents)} documents\033[0m")
            chunks = self.splitter.split_documents(documents)
            
            return chunks
        
        except Exception as e:
            print(f"\033[91m[ERROR] Failed to split documents: {str(e)}\033[0m")
            return documents # Return original documents if chunking fails
