from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .base_chunker import BaseChunker
from .utils import get_token_length


class AgenticChunker(BaseChunker):
    """
    Agentic chunking using LLM to decide where to split.
    """
    def __init__(self, max_chunk_size: int = 600, overlap: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            length_function=get_token_length,
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents using agentic chunking.
        """
        try:
            if not documents:
                print("\033[93m[WARN] No documents provided.\033[0m")
                return []
            
            print("\033[90mUsing Agentic chunking with Recursive Splitter fallback\033[0m")
            
            agentic_chunks = []
            for doc in documents:
                content = doc.page_content
                paragraphs = content.split('\n\n')
                
                current_chunk = ""
                for para in paragraphs:
                    if get_token_length(current_chunk + '\n\n' + para) > self.max_chunk_size:
                        if current_chunk.strip():
                            agentic_chunks.append(
                                Document(page_content=current_chunk.strip(), metadata=doc.metadata)
                            )
                    else:
                        current_chunk += ('\n\n' + para) if current_chunk else para
            
                if current_chunk.strip():
                    agentic_chunks.append(
                        Document(page_content=current_chunk.strip(), metadata=doc.metadata)
                    )
            
            print(f"\033[90mAgentic chunking: {len(agentic_chunks)} chunks created\033[0m")

            return agentic_chunks

        except Exception as e:
            print(f"\033[91m[ERROR - AgenticChunker] Failed to split documents: {str(e)}\033[0m")
            return self.fallback_splitter.split_documents(documents)
