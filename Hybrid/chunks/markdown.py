from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from .base_chunker import BaseChunker


class MarkdownChunker(BaseChunker):
    """
    Markdown chunker implementation
    
    Splits markdown documents by headers (#, ##, ###, etc.)
    """
    def __init__(self, headers_to_split_on: List[tuple] = None):
        if headers_to_split_on is None:
            self.headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),
                ("#####", "Header 5"),
                ("######", "Header 6"),
            ]
        else:
            self.headers_to_split_on = headers_to_split_on
        
        self.splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False # Keep headers in the text
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents using markdown header text splitter
        """
        try:
            if not documents:
                print(f"\033[93m[WARN] No documents provided for chunking.\033[0m")
                return []
            
            print(f"\033[90mUsing markdown header text splitting: {len(documents)} documents\033[0m")
            
            # Split documents
            chunks = []
            for doc in documents:
                splits = self.splitter.split_text(doc.page_content)

                for split in splits:
                    split.metadata.update(doc.metadata)
                    chunked_docs.append(split)
            
            return chunked_docs
        
        except Exception as e:
            print(f"\033[91m[ERROR] Failed to split documents: {str(e)}\033[0m")
            return documents # Return original documents if chunking fails
