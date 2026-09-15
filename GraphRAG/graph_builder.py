import re
import time
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_core.prompts import PromptTemplate


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
        

        self.extraction_prompt = PromptTemplate(
            template="""Sen bir Türk Ceza Kanunu (TCK) uzmanısın. Aşağıdaki metni analiz et ve varlıkları (entities) ile ilişkileri (relationships) JSON formatında çıkar.

KULLANABİLECEĞİN VARLIK (ENTITY) TÜRLERİ SADECE ŞUNLARDIR:
- SUÇ (Örn: Kasten öldürme, İşkence, Hırsızlık)
- CEZA (Örn: Müebbet hapis, Adli para cezası)
- NİTELİKLİ_HAL (Örn: Tasarlayarak, Canavarca hisle, Gebe kadına karşı)
- İNDİRİM_NEDENİ (Örn: Haksız tahrik, Yaş küçüklüğü, İyi hal)
- HUKUKA_UYGUNLUK (Örn: Meşru savunma, Zorunluluk hali, Kanun hükmü)
- KİŞİ (Örn: Fail, Mağdur, Kamu görevlisi)

KULLANABİLECEĞİN İLİŞKİ (RELATIONSHIP) TÜRLERİ SADECE ŞUNLARDIR:
- CEZALANDIRILIR (Suç -> Ceza bağlantısı)
- CEZAYI_ARTIRIR (Nitelikli Hal -> Suç bağlantısı)
- CEZAYI_İNDİRİR (İndirim Nedeni -> Suç bağlantısı)
- CEZAYI_KALDIRIR (Hukuka Uygunluk -> Suç bağlantısı)
- İÇERİR (Bir madde veya bölüm -> Suç bağlantısı)

METİN:
{text}

Aşağıdaki KESİN JSON formatında çıktı ver. Başka hiçbir açıklama yazma.
ÖNEMLİ KURAL: 'source_sentence' kısmına, o varlığı/ilişkiyi çıkardığın metindeki ORİJİNAL TAM CÜMLEYİ harfi harfine yazmalısın. Bu hukuki ispat için zorunludur.

{{
  "entities": [
    {{
      "name": "Kasten öldürme",
      "type": "SUÇ",
      "description": "Bir insanı kasten öldürme fiili",
      "source_sentence": "Bir insanı kasten öldüren kişi, müebbet hapis cezası ile cezalandırılır."
    }}
  ],
  "relationships": [
    {{
      "source": "Kasten öldürme",
      "target": "Müebbet hapis cezası",
      "type": "CEZALANDIRILIR",
      "source_sentence": "Bir insanı kasten öldüren kişi, müebbet hapis cezası ile cezalandırılır."
    }}
  ]
}}
""",
            input_variables=["text"]
        )
    
    def _clean_json_output(self, text: str) -> dict:
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(text)
        except Exception as e:
            return {"entities": [], "relationships": []}
    
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

        chain = self.extraction_prompt | self.llm

        for i, chunk in enumerate(chunks):
            print(f"\033[90mProcessing: {i+1}/{len(chunks)}\033[0m", end="\r")
            try:
                cypher_doc = """
                CREATE (d:Document {text: $text})
                """
                self.graph_db.query(cypher_doc, params={"text": chunk.page_content})

                response = chain.invoke({"text": chunk.page_content})
                response_text = response.content if hasattr(response, 'content') else str(response)

                parsed_data = self._clean_json_output(response_text)

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
                    cypher_rel = f"""
                    MATCH (s:`__Entity__` {{id: $source}})
                    MATCH (t:`__Entity__` {{id: $target}})
                    MERGE (s)-[r:{rel.get('type', 'İLGİLİDİR')}]->(t)
                    SET r.source_sentence = $sentence
                    """
                    self.graph_db.query(
                        cypher_rel,
                        params={
                            "source": rel.get("source"),
                            "target": rel.get("target"),
                            "sentence": rel.get("source_manager", "")
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
            
