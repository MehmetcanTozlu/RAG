import time
from typing import List
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# ====== Pydantic Templates ======
# Entity(Node) Template
class Entity(BaseModel):
    name: str = Field(description="Name of the entity or legal concept in Turkish (e.g., Kasten öldürme, Meşru savunma)")
    type: str = Field(description="MUST BE EXACTLY ONE OF: SUÇ, CEZA, NİTELİKLİ_HAL, İNDİRİM_NEDENİ, HUKUKA_UYGUNLUK, KİŞİ, KAVRAM")
    description: str = Field(description="Brief legal explanation of this entity in Turkish")
    source_sentence: str = Field(description="The EXACT ORIGINAL sentence from the provided Turkish source text. Do not translate.")


# Relationship (Edge) Template
class Relationship(BaseModel):
    source: str = Field(description="Name of the source entity (Must match exactly with an Entity name)")
    target: str = Field(description="Name of the target entity (Must match exactly with an Entity name)")
    type: str = Field(description="MUST BE EXACTLY ONE OF: CEZALANDIRILIR, CEZAYI_ARTIRIR, CEZAYI_İNDİRİR, CEZAYI_KALDIRIR, ŞARTIDIR, İLGİLİDİR")
    source_sentence: str = Field(description="The EXACT ORIGINAL sentence from the provided Turkish text that proves this relationship.")


# Final Extraction Template (Combines both)
class GraphExtraction(BaseModel):
    entities: List[Entity] = Field(description="List of extracted legal entities")
    relationships: List[Relationship] = Field(description="List of extracted relationships between entities")


class GraphBuilder:
    def __init__(
        self,
        llm,
        graph_db,
        db_uri: str = None,
        db_user: str = None,
        db_password: str = None,
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.llm = llm
        self.graph_db = graph_db
        self.db_uri = db_uri
        self.db_user = db_user
        self.db_password = db_password

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=[
                "\nMadde ",
                "\nBÖLÜM",
                "\nKISIM",
                "\n\n",
                "\n",
                " ", 
                ""
            ]
        )

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cuda"},
        )

        # We are giving Pydantic model to LangChain Json parser to make sure output is valid JSON
        self.parser = JsonOutputParser(pydantic_object=GraphExtraction)

        self.extraction_prompt = PromptTemplate(
            template="""You are an expert system analyzing the Turkish Penal Code (TCK). 
Your task is to extract legal entities and their relationships from the provided Turkish text.

INSTRUCTIONS:
1. Extract entities and relationships exactly as requested in the format instructions.
2. The values for 'name', 'description', and 'source_sentence' MUST be in Turkish, exactly as they appear in the source text.
3. NEVER invent or translate any 'source_sentence'. Extract it exactly from the text.
4. Output ONLY valid JSON. No explanations, no markdown outside of JSON.

FORMAT INSTRUCTIONS:
{format_instructions}

SOURCE TEXT (TURKISH):
{text}
""",
            input_variables=["text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
    
    def process_and_build(self, pdf_path: str):
        print(f"\033[94mLoading PDF from {pdf_path}...\033[0m")
        loader = PyPDFLoader(file_path=pdf_path)
        raw_documents = loader.load()

        chunks = self.text_splitter.split_documents(raw_documents)
        print(f"\033[92mSuccessfully loaded PDF and split into {len(chunks)} chunks.\033[0m")

        print(f"\033[94mExtracting entities and relationships from chunks...\033[0m")
        start_time = time.time()

        total_entities = 0
        total_relationships = 0

        chain = self.extraction_prompt | self.llm | self.parser

        for i, chunk in enumerate(chunks):
            print(f"\033[90mProcessing: {i+1}/{len(chunks)}\033[0m", end="\r")
            
            # Security Network: If llm fails to extract entities and relationships, we should still save the document to the graph.
            try:
                cypher_doc = """
                CREATE (d:Document {text: $text})
                """
                self.graph_db.query(cypher_doc, params={"text": chunk.page_content})
            
            except Exception as e:
                pass

            # LLM Processing
            try:
                parsed_data = chain.invoke({"text": chunk.page_content})

                # Entities save to Neo4js with Sentence
                for entity in parsed_data.get("entities", []):
                    cypher_node = """
                        MERGE (e:`__Entity__` {id: $name})
                        SET e.name = $name,
                            e.type = $type,
                            e.description = $desc,
                            e.source_sentence = $sentence
                    """
                    self.graph_db.query(
                        cypher_node,
                        params={
                            "name": entity.get("name"),
                            "type": entity.get("type", "KAVRAM"),
                            "desc": entity.get("description", ""),
                            "sentence": entity.get("source_sentence", "")
                        }
                    )
                    total_entities += 1
                
                # Relationships save to Neo4j with Sentence
                for rel in parsed_data.get("relationships", []):
                    cypher_rel = """
                    MATCH (s:`__Entity__` {id: $source})
                    MATCH (t:`__Entity__` {id: $target})
                    MERGE (s)-[r:İLGİLİDİR]->(t)
                    SET r.type = $type,
                        r.source_sentence = $sentence
                    """
                    self.graph_db.query(
                        cypher_rel,
                        params={
                            "source": rel.get("source"),
                            "target": rel.get("target"),
                            "type": rel.get("type", "İLGİLİDİR"),
                            "sentence": rel.get("source_sentence", "")
                        }
                    )
                    total_relationships += 1
            
            except Exception as e:
                pass
            
        print(f"\n\033[92mEntities completed. {total_entities} Entities, {total_relationships} Relationships saved.\033[0m")

        print(f"\033[93mCreating Neo4j Vector store index (Hybrid Search)...\033[0m")
        try:
            Neo4jVector.from_existing_graph(
                embedding=self.embedding_model,
                url=self.db_uri,
                username=self.db_user,
                password=self.db_password,
                index_name="entity_vector_index",
                keyword_index_name="entity_keyword_index",
                search_type="hybrid",
                node_label="__Entity__",
                text_node_properties=["name", "description"],
                embedding_node_property="embedding",
            )
            print(f"\033[92mNeo4j Vector store index created successfully.\033[0m")

            elapsed_time = time.time() - start_time
            print(f"\033[92mSaved graph documents to Neo4j database.\033[0m")
            print(f"\033[90mSuccessfully extracted entities and relationships in {elapsed_time:.2f} seconds.\033[0m")
        
        except Exception as e:
            raise(f"\033[91mFailed to extract entities and relationships: {str(e)}\033[0m")
            
