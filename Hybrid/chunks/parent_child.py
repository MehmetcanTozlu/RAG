from langchain_text_splitters import RecursiveCharacterTextSplitter
from .utils import get_token_length


class ParentChildChunker:
    """
    This class implements parent-child chunking.
    Parent-Child includes 2 method:
        - Split parent and child chunks
        - Merge parent and child chunks
    """
    def __init__(self, parent_chunk_size: int = 2000, child_chunk_size: int = 400, parent_chunk_overlap: int = 200, child_chunk_overlap: int = 50):
        # Parent splitter
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            length_function=get_token_length,
        )

        # Child splitter
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            length_function=get_token_length,
        )
    
    def get_splitter(self):
        """
        Returns two splitter: parent splitter and child splitter
        """
        return self.parent_splitter, self.child_splitter
