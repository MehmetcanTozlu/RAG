"""
In the standart LangChain architecture, for this technique to work properly,
the embedding model(e.g. jina-embeddings-v3) must support. However, we'll write a
"Late Chunking Preparer", that splits the documents and embeds the exact start and end
coordinates(spans) of the original text into the metadata so that the model can use them.
"""
import uuid
from typing import List
from langchain_core.documents import Document
from .base_chunker import BaseChunker


class LateChunker(BaseChunker):
    """
    This class splits but that really power is adding metadata to the chunks.
    """
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents using late chunking.
        """
        try:
            if not documents:
                print(f"\033[93m[WARN] No documents provided.\033[0m")
                return []
            
            print(f"\033[90m Using Late Chunking Preparer\033[0m")

            late_chunks = []

            for doc in documents:
                text = doc.page_content
                doc_id = str(uuid.uuid4()) # we are giving an unique id to the document, this id will be used to retreive the original text
            
                start = 0
                while start < len(text):
                    end = min(start + self.chunk_size, len(text))

                    if end < len(text) and text[end] != ' ':
                        last_space = text.rfind(' ', start, end)
                        if last_space != -1:
                            end = last_space
                                
                    chunk_text = text[start:end].strip()

                    if chunk_text:
                        new_metadata = doc.metadata.copy()
                        new_metadata.update({
                            "parent_doc_id": doc_id,
                            "span_start": start,
                            "span_end": end,
                            "chunking_strategy": "late_chunking_sim"
                        })
                        late_chunks.append(Document(page_content=chunk_text, metadata=new_metadata))

                    start = end + 1
            
            print(f"\033[90mLate chunking: {len(late_chunks)} chunks created\033[0m")
            return late_chunks

        except Exception as e:
            print(f"\033[91m[ERROR - LateChunker] Failed to split documents: {str(e)}\033[0m")
            return documents
