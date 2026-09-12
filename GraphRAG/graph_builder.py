import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector


class GraphBuilder:

    def __init__(
        self,
        llm,
        graph_db,
        db_uri: str = None,
        db_user: str = None,
        db_password: str = None,
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        self.llm = llm
        self.graph_db = graph_db
        self.db_uri = db_uri
        self.db_user = db_user
        self.db_password = db_password

        self.transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=[
                "Suç", "Ceza", "Kişi", "Kavram", "KanunMaddesi", "Kurum"
            ], # Node types (Labels)
            allowed_relationships=[
                "CEZALANDIRILIR", "TANIMLAR", "İÇERİR", "İLGİLİDİR", "UYGULANIR", "KARAR_VERİR"
            ] # Relationship types between nodes
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cuda"},
        )
    
    def process_and_build(self, pdf_path: str):
        print(f"\033[94mLoading PDF from {pdf_path}...\033[0m")
        loader = PyPDFLoader(file_path=pdf_path)
        raw_documents = loader.load()

        chunks = self.text_splitter.split_documents(raw_documents)
        print(f"\033[92mSuccessfully loaded PDF and split into {len(chunks)} chunks.\033[0m")

        print(f"\033[94mExtracting entities and relationships from chunks...\033[0m")
        
        start_time = time.time()

        try:
            graph_documents = self.transformer.convert_to_graph_documents(chunks)
            self.graph_db.add_graph_documents(
                graph_documents,
                baseEntityLabel=True, # Creates the base entity node and adds it to the graph
                include_source=True, # Includes the source document as a node in the graph
            )

            print(f"\033[93mCreating Neo4j Vector store index (Hybrid Search)...\033[0m")
            Neo4jVector.from_existing_graph(
                embedding=self.embedding_model,
                url=self.db_uri,
                username=self.db_user,
                password=self.db_password,
                index_name="entity_vector_index",
                node_label="__Entity__",
                text_node_properties=["id"],
                embedding_node_property="embedding",
            )
            print(f"\033[92mNeo4j Vector store index created successfully.\033[0m")

            elapsed_time = time.time() - start_time
            print(f"\033[92mSaved graph documents to Neo4j database.\033[0m")
            print(f"\033[90mSuccessfully extracted entities and relationships in {elapsed_time:.2f} seconds.\033[0m")
        
        except Exception as e:
            print(f"\033[91mFailed to extract entities and relationships: {str(e)}\033[0m")
