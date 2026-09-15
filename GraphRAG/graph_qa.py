from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from graph_state import GraphRAGState
from langchain_neo4j import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings


class GraphQA:
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

        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cuda"},
        )

        # 1. Connect to Neo4j Vector Index (for Hybrid Search)
        self.vector_index = Neo4jVector.from_existing_index(
            embedding=embedding_model,
            url=db_uri,
            username=db_user,
            password=db_password,
            index_name="entity_vector_index",
            node_label="__Entity__",
            text_node_property="name",
        )

        # Security Network Index (For Directly Original Texts)
        print("\033[93mSecurity Network (Document) is checking Vector Index..\033[0m")
        self.doc_index = Neo4jVector.from_existing_graph(
            embedding=embedding_model,
            url=db_uri,
            username=db_user,
            password=db_password,
            index_name="document_vector_index",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )

        # Generating Prompt Verifiable Attribution
        self.answer_prompt = PromptTemplate(
            template="""Sen katı ve analitik bir Türk Ceza Kanunu (TCK) asistanısın.
Aşağıdaki numaralandırılmış kesin kanun metinlerini (Kaynaklar) kullanarak soruyu cevapla.

KURALLAR:
1. Sadece verilen kaynaklardaki bilgileri kullan. Bilgi yoksa "Bu konu hakkında kanun metninde bilgi bulunmamaktadır." de.
2. Her hukuki iddianın sonuna, o bilgiyi aldığın kaynağın numarasını [1], [2] şeklinde EKLEMEK ZORUNDASIN.
3. Asla aynı cümleyi tekrar etme. 

Kaynaklar:
{context}

Soru: {question}
Cevap:""",
            input_variables=["context", "question"]
        )

    # Node 1: Hybrid Search (Vector + Graph Extension)
    def _hybrid_search(self, state: GraphRAGState):
        print("\n\033[94m(Step 1) Assets and original source texts relating to vector searches can be found...\033[0m")
        question = state["question"]

        try:
            context_str = "Kanun Maddeleri (Doğrulanabilir Kaynaklar):\n"
            unique_sources = set()
            relationships_text = "Graph Bağlantıları:\n"

            doc_results = self.doc_index.similarity_search(question, k=2)
            for doc in doc_results:
                unique_sources.add(doc.page_content)
            
            # We use vector search to find the three nodes that are semantically most relevant
            vector_results = self.vector_index.similarity_search(question, k=2)
            entity_ids = [res.metadata.get("id") for res in vector_results if res.metadata.get("id")]

            if entity_ids:
                cypher_query = """
                UNWIND $entity_ids AS e_id
                MATCH (start:`__Entity__` {id: e_id})
                OPTIONAL MATCH (start)-[r]-(neighbor:`__Entity__`)
                RETURN DISTINCT 
                    start.name AS Baslangic, type(r) AS Iliski, neighbor.name AS Komsuluk,
                    start.source_sentence AS Kaynak1, r.source_sentence AS Kaynak2, neighbor.source_sentence AS Kaynak3
                LIMIT 5
                """
                graph_results = self.graph_db.query(cypher_query, params={"entity_ids": entity_ids})

                for res in graph_results:
                    if res.get('Iliski') and res.get('Komsuluk'):
                        relationships_text += f"- {res['Baslangic']} -> {res['Iliski']} -> {res['Komsuluk']}\n"
                    
                    if res['Kaynak1']: unique_sources.add(res['Kaynak1'])
                    if res['Kaynak2']: unique_sources.add(res['Kaynak2'])
                    if res['Kaynak3']: unique_sources.add(res['Kaynak3'])

            for i, source in enumerate(list(unique_sources), 1):
                context_str += f"[{i}] {source}\n"
            
            if "->" in relationships_text:
                context_str += f"\n{relationships_text}"
            
            # print(f"\033[90m   {len(graph_results)} definite graph connections and legal texts were retrieved from Neo4j.\033[0m")
            return {"graph_context": context_str.strip()}
        
        except Exception as e:
            print(f"\033[91mSearch Error: {e}\033[0m")
            return {"graph_context": "[]"}
    
    # Node 2: Production and Hallucination Shield
    def _generate_answer(self, state: GraphRAGState):
        question = state["question"]
        context = state["graph_context"]

        # Hallucination Shield
        if not context or context.strip() == "" or context.strip() == "[]":
            print("\033[93m[!] No data returned from Neo4j; LLM is being skipped (hallucination prevented).\033[0m")
            return {"answer": "I’m sorry, I couldn’t find any information in the database that matches this query. Please try using different keywords."}
        
        print("\033[94m(Step 2)  The LLM generates the final answer using the extracted legal texts...\033[0m")
        chain = self.answer_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        # answer.replace("[BİTTİ]", "")

        return {"answer": answer}
    
    def build_workflow(self):
        workflow = StateGraph(GraphRAGState)

        workflow.add_node("hybrid_search", self._hybrid_search)
        workflow.add_node("generate_answer", self._generate_answer)

        workflow.add_edge(START, "hybrid_search")
        workflow.add_edge("hybrid_search", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()
    
    def ask(self, question: str):
        app = self.build_workflow()
        initial_state = {
            "question": question,
            "cypher_query": "",
            "graph_context": "",
            "answer": ""
        }
        result = app.invoke(initial_state)

        print("\n\033[92m" + "="*60 + "\033[0m")
        print(f"\033[96mQuestion:\033[0m {result['question']}")
        print(f"\033[96mCypher Query:\033[0m {result['cypher_query']}")
        print(f"\033[96mGraph Context:\033[0m {result['graph_context']}")
        print(f"\033[93mGraphRAG Bot:\033[0m {result['answer']}")
        print("\033[92m" + "="*60 + "\033[0m")
        
        return result


