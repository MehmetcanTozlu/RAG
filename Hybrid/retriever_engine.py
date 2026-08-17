import time
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore

from Hybrid.chunks import (
    RecursiveChunker,
    FixedSizeChunker,
    MarkdownChunker,
    SemanticTextChunker,
    ParentChildChunker
)


def clean_text(list_of_documents):
    """
    Clean text from documents.
    """
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace("\t", " ")
    
    return list_of_documents

def show_context(context_docs: List):
    """
    Print retrieved documents.
    """
    for i, doc in enumerate(context_docs):
        print(f"\n\033[96m[Found Documents {i + 1}]:\033[0m")
        print(doc.page_content)
        print("-" * 50)


class RAG:
    """
    Build RAG pipeline.
    """
    def __init__(
        self,
        pdf_path: str,
        embed_model_path: str,
        db_path: str,
        collection_name: str,
        chunker = None, # RecursiveChunker, FixedSizeChunker, etc.
        n_retrieved: int = 3,
        embedder = None,
    ):
        print("\033[92mInitializing RAG pipeline...\033[0m")

        self.n_retrieved = n_retrieved

        # If no chunker is provided, use default RecursiveChunker
        self.chunker = chunker if chunker is not None else RecursiveChunker(chunk_size=500, chunk_overlap=50)

        # Check if chunker is ParentChildChunker for parent-child chunking
        self.is_parent_child = isinstance(self.chunker, ParentChildChunker)

        # Initialize document store for parent-child chunking (Holds parent documents in RAM)
        self.docstore = InMemoryStore()
        
        if embedder is not None:
            self.embeddings = embedder
        else:
            # Load Embedding Model
            print("\033[94m[*]Loading embedding model...\033[0m")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embed_model_path,
                model_kwargs={'device': 'cuda'},
                encode_kwargs={'normalize_embeddings': True},
            )

        # Connect Qdrant DB
        self.client = QdrantClient(path=db_path)

        if self.is_parent_child or not self.client.collection_exists(collection_name=collection_name):
            print("\033[93m[INFO] Parent-child chunking or collection does not exist. Creating new collection.\033[0m")
            self.vector_store = self._ingest_documents(pdf_path, collection_name, chunker)
        else:
            print("\033[93m[WARN] Collection already exists. Using existing collection.\033[0m")
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embeddings,
            )
        
            # Set up Retriever
            self.retriever = self.vector_store.as_retriever(
                search_type='similarity',
                search_kwargs={'k': self.n_retrieved},
            )
    
    def _ingest_documents(self, pdf_path, collection_name, chunker):
        """
        Read a PDF file and ingest its content into Qdrant vector store
        """
        start_time = time.time()

        # Calculate Dynamic Embedding Dimension
        vector_dim = len(self.embeddings.embed_query("test_dim"))

        # Create Qdrant Collection
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )

        # Load PDF
        print(f"\033[94mLoading PDF: {pdf_path}\033[0m")
        loader = PyPDFLoader(pdf_path)
        documents = clean_text(loader.load())

        if self.is_parent_child:
            print("\033[95m[INFO] Using architecture: Parent-Child (Auto-Merging)\033[0m")
            parent_splitter, child_splitter = self.chunker.get_splitter()

            self.retriever = ParentDocumentRetriever(
                vectorstore=vector_store,
                docstore=self.docstore,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
                search_kwargs={'k': self.n_retrieved},
            )

            print("\033[95m[INFO] Ingesting parents to RAM and children to Qdrant...\033[0m")
            self.retriever.add_documents(documents)
        else:
            print(f"\033[94m[INFO] Using chunker: {self.chunker.__class__.__name__}\033[0m")

            raw_chunks = self.chunker.split_documents(documents)
            print(f"\033[94m[INFO] Chunked documents: {len(raw_chunks)}\033[0m")

            vector_store.add_documents(raw_chunks)

            self.retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={'k': self.n_retrieved},
            )
        
        print(f"\033[92m[+] Data Ingested Successfully! Time: {time.time() - start_time:.2f} seconds\033[0m")
        return vector_store
    
    def get_context(self, query: str) -> List[str]:
        """
        Retrieves context documents for a given query using vector similarity search.
        """
        docs = self.retriever.invoke(query)
        
        return [doc.page_content for doc in docs]
    
    def test_retrieval(self, query: str):
        """
        Tests the retrieval system with a given query.
        """
        print(f"\n\033[96m[Query]: {query}\033[0m")
        start_time = time.time()

        docs = self.retriever.invoke(query)

        print(f"\033[92m[+] Retrieved {len(docs)} documents in {time.time() - start_time:.2f} seconds\033[0m")
        show_context(docs)


# TEST
if __name__ == '__main__':
    embed_model_path = "./multilingual-e5-small"
    pdf_path = "...pdf"
    db_path = "./data/db"
    collection_name = "example"
    n_retrieved = 3
    chunk_size = 500
    chunk_overlap = 50

    # chunker = RecursiveChunker(chunk_size, chunk_overlap)
    # chunker = FixedSizeChunker(chunk_size, chunk_overlap)
    # chunker = MarkdownChunker()

    # embedder = HuggingFaceEmbeddings(
    #     model_name=embed_model_path,
    #     model_kwargs={'device': 'cuda'},
    #     encode_kwargs={'normalize_embeddings': True},
    # )
    # chunker = SemanticTextChunker(embeddings=embed_model_path, breakpoint_threshold_type="percentile", breakpoint_threshold=0.8)

    chunker = ParentChildChunker(
        parent_chunk_size=2000,
        child_chunk_size=200,
        parent_chunk_overlap=200,
        child_chunk_overlap=50,
    )

    rag = RAG(
        pdf_path=pdf_path,
        embed_model_path=embed_model_path,
        db_path=db_path,
        collection_name=collection_name,
        chunker=chunker,
        n_retrieved=n_retrieved,
        embedder=embedder if 'embedder' in locals() else None,
    )

    rag.test_retrieval(query="What is the capital of France?")

    print(f"\n\033[90mClosing Qdrant connection...\033[0m")
    if hasattr(rag, 'client'):
        rag.client.close()
