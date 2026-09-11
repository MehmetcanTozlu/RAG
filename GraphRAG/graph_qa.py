import re
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from graph_state import GraphRAGState


class GraphQA:
    def __init__(self, llm, graph_db):
        self.llm = llm
        self.graph_db = graph_db
        self.schema = self.graph_db.get_schema # Pull the schema from the neo4j database

        # 1. Convert User Question to Cypher Query (Graph SQL)
        self.cypher_prompt = PromptTemplate(
            template="""Sen bir Neo4j uzmanısın. Görevin, kullanıcının sorusu için Cypher kodu yazmaktır.

KURALLAR:
1. Süslü parantez ile tam eşleşme YASAK (Örn: {{id: "..."}}).
2. HER ZAMAN WHERE toLower(n.id) CONTAINS "kelime" yapısını kullan.
3. Node'ların sadece 'id' özelliği vardır (n.id).

ÖRNEKLER:
Soru: Kasten yaralama suçunun cezası nedir?
Cypher: MATCH (s:Suç)-[r]->(c:Ceza) WHERE toLower(s.id) CONTAINS "yaralama" RETURN s.id, type(r), c.id LIMIT 10

Soru: Kasten öldürme suçu nedir?
Cypher: MATCH (n)-[r]->(m) WHERE toLower(n.id) CONTAINS "öldürme" RETURN n.id, type(r), m.id LIMIT 15

Soru: {question}
Cypher:""",
            input_variables=["schema", "question"],
        )

        # 2. Answer the User's Question based on the Cypher Query Results
        self.answer_prompt = PromptTemplate(
            template="""Sen bir hukuk asistanısın. Aşağıdaki grafik veritabanından çekilmiş kesin bilgileri (Context) kullanarak kullanıcının sorusunu cevapla.
Eğer bağlamda (Context) soruyla ilgili bir bilgi yoksa, "Veritabanımda bu konuyla ilgili bilgi bulamadım." de ve uydurma (halüsinasyon) yapma.

Bağlam (Context):
{context}

Soru: {question}
Cevap:""",
            input_variables=["context", "question"],
        )

    # NODE 1
    def _generate_cypher(self, state: GraphRAGState):
        """Generates a Cypher query based on the user's question and the graph schema."""
        print("\n\033[94m(Step 1) LLM is converting user's question into Cypher query...\033[0m")
        question = state["question"]


        chain = self.cypher_prompt | self.llm | StrOutputParser()
        raw_cypher = chain.invoke({"question": question})

        match = re.search(r'(MATCH\s+[\s\S]*)', raw_cypher, re.IGNORECASE)
        if match:
            cypher_query = match.group(1).strip()
        else:
            cypher_query = raw_cypher.strip()

        # # Clean up the raw Cypher query
        cypher_query = raw_cypher.replace("```cypher", "").replace("```", "")
        cypher_query = cypher_query.replace("))", ")").replace("}}", "}").strip()
        print(f"\033[90mGenerated Cypher Query: {cypher_query}\033[0m")

        return {"cypher_query": cypher_query}
    
    # NODE 2
    def _execute_cypher(self, state: GraphRAGState):
        """Executes the generated Cypher query on the Neo4j database."""
        print("\n\033[94m(Step 2) Executing Cypher query on Neo4j database...\033[0m")
        cypher_query = state["cypher_query"]

        # If the Cypher query is not valid, return an error
        if not cypher_query or not cypher_query.upper().startswith("MATCH"):
            print("\033[91mInvalid Cypher query. Please try again.\033[0m")
            return {"graph_context": "[]"}

        try:
            result = self.graph_db.query(cypher_query)
            graph_context = str(result) if result else "[]"
            print(f"\033[90mReturned graph context from Neo4j: {graph_context[:150]}\033[0m")
        
        except Exception as e:
            print(f"\033[91mFailed to execute Cypher query: {str(e)}\033[0m")
            graph_context = "[]"
        
        return {"graph_context": graph_context}

    # NODE 3
    def _generate_answer(self, state: GraphRAGState):
        """Generates the final answer based on the graph context and the user's question."""
        print("\033[94m(Step 3) LLM is generating the final answer based on the graph context and the user's question...\033[0m")
        question = state["question"]
        context = state["graph_context"]

        if not context or context == "[]" or context.strip() == "":
            print("\033[93mNo graph context available. Generating answer without context...\033[0m")
            return {"answer": "Veritabanımda bu konuyla ilgili bilgi bulamadım."}

        chain = self.answer_prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})

        return {"answer": answer}
    
    # LangGraph Workflow
    def build_workflow(self):
        # Initialize the state graph
        workflow = StateGraph(GraphRAGState)

        workflow.add_node("generate_cypher", self._generate_cypher)
        workflow.add_node("execute_cypher", self._execute_cypher)
        workflow.add_node("generate_answer", self._generate_answer)

        workflow.add_edge(START, "generate_cypher")
        workflow.add_edge("generate_cypher", "execute_cypher")
        workflow.add_edge("execute_cypher", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()
    
    def ask(self, question: str):
        """Asks a question to the knowledge graph."""
        app = self.build_workflow()
        initial_state = {
            "question": question,
            "cypher_query": "",
            "graph_context": "",
            "answer": ""
        }

        result = app.invoke(initial_state)

        print(f"\n\033[92m" + "="*60 + "\033[0m")
        print(f"\033[96mQuestion:\033[0m {result['question']}")
        print(f"\033[95mCypher Query:\033[0m {result['cypher_query']}")
        print(f"\033[94mGraph Context:\033[0m {result['graph_context']}")
        print(f"\033[93mFinal Answer:\033[0m {result['answer']}")
        print(f"\033[92m" + "="*60 + "\033[0m")

        return result
        

        