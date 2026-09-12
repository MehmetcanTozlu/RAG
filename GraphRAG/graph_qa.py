import re
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
            text_node_property="id",
        )

        # Generating Prompt Verifiable Attribution
        self.answer_prompt = PromptTemplate(
            template = """Sen uzman bir hukuk asistanısın. Aşağıdaki veritabanından çekilmiş kesin kanun metinlerini (Context) kullanarak kullanıcının sorusunu cevapla.
Sadece verilen bağlamdaki bilgileri kullan, asla uydurma yapma.

Bağlam (Orijinal Kanun Metinleri ve İlişkiler):
{context}

Soru: {question}
Cevap:""",
            input_variables = ["context", "question"]
        )

    # Node 1: Hybrid Search (Vector + Graph Extension)
    def _hybrid_search(self, state: GraphRAGState):
        print("\n\033[94m(Step 1) Assets and original source texts relating to vector searches can be found...\033[0m")
        question = state["question"]

        try:
            # We use vector search to find the three nodes that are semantically most relevant
            vector_results = self.vector_index.similarity_search(question, k=3)
            entity_ids = [res.page_content for res in vector_results]

            if not entity_ids:
                return {"graph_context": "[]"}
            
            # STABLE AND SECURE CYPHER: We retrieve the original PDF text (Document.text) from the nodes found
            cypher_query = """
            MATCH (e:`__Entity__`) WHERE e.id IN $entity_ids
            MATCH (e)-[r]-(neighbor)
            MATCH (e)<-[:MENTIONS]-(d:Document)
            RETURN DISTINCT e.id AS Varlik, type(r) AS Iliski, neighbor.id AS BaglantiliVarlik, d.text AS KaynakMetin
            LIMIT 5
            """
            graph_results = self.graph_db.query(cypher_query, params={"entity_ids": entity_ids})

            # We are converting this into a clean text that the LLM can read easily
            context_str = ""
            for res in graph_results:
                context_str += f"- Grafik Bağlantısı: {res['Varlik']} {res['Iliski']} {res['BaglantiliVarlik']}\n"
                context_str += f"  Orijinal Kaynak: {res['KaynakMetin']}\n\n"
            
            print(f"\033[90m   {len(graph_results)} definite graph connections and legal texts were retrieved from Neo4j.\033[0m")
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


