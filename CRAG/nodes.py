from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from state import GraphState
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document


class CRAGNodes:
    """
    CRAGNodes Class holds all the nodes used in the CRAG system
    """
    def __init__(self, llm_engine, retriever):
        self.llm_engine = llm_engine
        self.llm = llm_engine.get_llm()
        self.retriever = retriever
        self.web_search_tool = DuckDuckGoSearchRun()
    
    def retrieve(self, state: GraphState):
        """
        Retrieve documents based on the question in the state
        """
        print(f"\033[94m---AGENT: RETRIEVING DOCUMENTS USING EMBEDDINGS---\033[0m")
        question = state['question']
        documents = self.retriever.invoke(question)

        return {"documents": documents, "question": question}
    
    def grade_documents(self, state: GraphState):
        """
        Critically evaluate the retrieved documents using LLM
        """
        print(f"\033[94m---AGENT: CRITICALLY EVALUATING EMBEDDING RESULTS---\033[0m")
        question = state['question']
        documents = state['documents']

        sys_message = (
            "Sen katı kurallara bağlı bir Asistansın.\n"
            "Kullanıcının sorusunu SADECE sana verilen <baglam> metinlerini kullanarak yanıtla.\n"
            "DİKKAT: Eğer <baglam> metinlerinde bilim kurgu, gerçek dışı olaylar (Örn: Mars'ta yaşam/askerlik) "
            "veya saçma/alakasız bilgiler varsa, onlara güvenme ve '[BİLMİYORUM - Bağlam Mantıksız]' yaz.\n"
            "Eğer bilgi mantıklıysa, kısaca özetle."
        )

        user_message = "Soru: {question}\n\nBelge: {context}\n\nKarar:"

        prompt = self.llm_engine.create_prompt(
            system_message=sys_message,
            user_message=user_message,
        )
        
        grader_chain = prompt | self.llm | StrOutputParser()

        filtered_docs = []
        web_fallback = False

        for d in documents:
            score = grader_chain.invoke({
                "question": question,
                "context": d.page_content
            })
            grade = score.strip().upper()
            
            if "[UYGUN]" in grade:
                print(f"\033[92m [+] Document Accepted\033[0m")
                filtered_docs.append(d)
            else:
                print(f"\033[91m [-] Document Rejected\033[0m")

        # If any document rejected, use web search
        if not filtered_docs:
            print("\033[93mFallback: Searching web for relevant documents...\033[0m")
            web_fallback = True
        else:
            web_fallback = False
        
        return {"documents": filtered_docs, "question": question, "web_fallback": web_fallback}

    def web_search(self, state: GraphState):
        """
        Perform web search using DuckDuckGo
        """
        print("\033[93m---AGENT: SEARCHING ON THE WEB---\033[0m")
        question = state['question']
        documents = state['documents']
        
        try:
            docs = self.web_search_tool.invoke(question)
            
            web_results = Document(
                page_content=docs,
                metadata={"source": "duckduckgo_web"}
            )

            documents.append(web_results)
            print(f"\033[92m [+] Web search successful and added to context.\033[0m")
        
        except Exception as e:
            print(f"\033[91m [-] Error searching web: {e}\033[0m")
        
        return {"documents": documents, "question": question}
    
    def generate(self, state: GraphState):
        """
        Generate final response using the retrieved documents
        """
        print(f"\033[94m---AGENT: GENERATING RESPONSE---\033[0m")
        question = state['question']
        documents = state['documents']
        
        context_texts = [doc.page_content.replace('\n', ' ') for doc in documents]
        combined_context = "\n\n--\n\n".join(context_texts)

        sys_message = (
            "Sen katı kurallara bağlı bir Türk Hukuku ve Genel Bilgi asistanısın.\n"
            "Kullanıcının sorusunu SADECE sana verilen <baglam> metinlerini kullanarak yanıtla.\n"
            "Asla kendi bilgilerini uydurma."
        )

        user_message = "<baglam>\n{context}\n</baglam>\n\nSoru: {question}\nCevap:"

        prompt = self.llm_engine.create_prompt(
            system_message=sys_message, 
            user_message=user_message
        )

        rag_chain = prompt | self.llm | StrOutputParser()

        response = rag_chain.invoke({
            "context": combined_context,
            "question": question
        })

        return {"documents": documents, "question": question, "generation": response}
