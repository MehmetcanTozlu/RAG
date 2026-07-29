import os
import time
import argparse
from typing import List
import tiktoken
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_huggingface import HuggingFaceEmbeddings


def replace_t_with_space(list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document.

    Args:
        list_of_documents: A list of document objects, each with a 'page_content' attribute.

    Returns:
        The modified list of documents with tab characters replaced by spaces.
    """
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')
    
    return list_of_documents

def retrieve_context_per_question(question: str, chunks_query_retriever) -> List[str]:
    """
    Retrieve context for a given question using the chunks query retriever.

    Args:
        question (str): The question to retrieve context for.
        chunks_query_retriever: The chunks query retriever.

    Returns:
        List[str]: A list of context for the given question.
    """
    docs = chunks_query_retriever.invoke(question)
    context = [doc.page_content for doc in docs]
    
    return context

def show_context(context: List[str]):
    """
    Display the retrieved context for a given question.
    """
    for i, con in enumerate(context):
        print(f"\n[Context {i + 1}]:")
        print(con)
        print("-" * 50)


def tiktoken_len(text):
    """
    Returns the number of tokens in a text using the tiktoken library.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))


class SimpleRAG:
    """
    Simple RAG class for document encoding and query retrieval.
    """
    def __init__(
        self,
        path: str,
        embed_model_path: str,
        db_path: str,
        collection_name: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        n_retrieved: int = 2
    ):
        print("\n--- Initializing Simple RAG Retriever ---")

        self.time_records = {}

        # 1. Loading Embedding Model
        print(f"\033[94mLoading BGE Embedding Model (Local)...\033[0m")
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True}

        self.embeddings = HuggingFaceEmbeddings(
            model_name = embed_model_path,
            model_kwargs = model_kwargs,
            encode_kwargs = encode_kwargs,
        )

        # 2. Client Connection
        self.client = QdrantClient(path=db_path)
        
        # 3. Check DB
        collection_exists = self.client.collection_exists(collection_name)
        if collection_exists:
            print(f"\033[92mCollection '{collection_name}' already exists. Loading existing collection...\033[0m")

            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embeddings,
            )
        else:
            print(f"\033[93mNo collection found. Creating new collection '{collection_name}' and ingesting documents...\033[0m")
            start_time = time.time()

            # Dinamically Dimension Calculation
            vector_dim = self.embeddings.embed_query("test")
            vector_dim = len(vector_dim)

            # Create Collection
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
            )

            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=collection_name,
                embedding=self.embeddings,
            )

            # Load Document
            loader = PyPDFLoader(path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=tiktoken_len,
            )

            cleaned_chunks = replace_t_with_space(text_splitter.split_documents(documents))

            print(f"Chunked {len(cleaned_chunks)} chunks from {path}. Chunks written to disk at {db_path}.")

            self.vector_store.add_documents(cleaned_chunks)
            
            print(f"\033[92mDocuments ingested successfully. Time: {time.time() - start_time:.4f} seconds\033[0m")
        
        # 4. Initialize the retriever
        self.chunks_query_retriever = self.vector_store.as_retriever(
            search_type='similarity',
            search_kwargs={'k': n_retrieved},
        )
    
    def run(self, query: str):
        """
        Execute the query. Bring and show context.

        Args:
            query (str): Query to execute.
        """
        print(f"\n--- Processing Query: {query} ---")
        start_time = time.time()
        
        context = retrieve_context_per_question(query, self.chunks_query_retriever)

        self.time_records['Retrieval'] = time.time() - start_time
        print(f"Retrieval Time: {self.time_records['Retrieval']:.4f} seconds")

        show_context(context)
    
    def inspect_db(self, collection_name :str, limit: int = 2):
        """
        Inspects the Qdrant database and prints information about the documents.
        """
        print("\n" + "=" * 50)
        print("QDRANT DB INSPECTION")
        print("=" * 50)
        
        # Count points in collection
        count = self.client.count(collection_name=collection_name)
        print(f"\033[94mTotal points in collection '{collection_name}': {count.count}\033[0m")

        # Get some records from the collection
        records, next_page = self.client.scroll(
            collection_name=collection_name,
            limit=limit,
            with_payload=True, # We want the text data
            with_vectors=False, # We don't want the vectors for now
        )

        print(f"\n\033[92mFirst {limit} retrieved documents:\033[0m")
        for i, record in enumerate(records):
            print(f"\n[Document {i + 1}]")
            print(f"ID: {record.id}")
            print(f"Payload: {record.payload}")
            print(f"Vector: {record.vector}")


if __name__ == '__main__':

    simple_rag = SimpleRAG(
        path = "",
        embed_model_path = "",
        db_path="",
        collection_name="",
        chunk_size=500,
        chunk_overlap=50,
    )

    simple_rag.run(
        query="Türk ceza kanununda ki 1. madde nedir?",
    )

    simple_rag.inspect_db(collection_name="turk_ceza_kanunu", limit=2)

    # Graceful Shutdown
    print("\nClosing Qdrant connection cleanly...")
    if hasattr(simple_rag, 'client'):
        simple_rag.client.close()


