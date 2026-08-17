from langchain_core.output_parsers import StrOutputParser


class QueryRewriterAndRouter:
    """
    Query Rewriter and Router.
    Rewrites a user query into multiple different forms to improve retrieval performance.
    """
    def __init__(self, llm_engine):
        self.llm_engine = llm_engine
        self.llm = llm_engine.get_llm()
        self.chain = self._build_chain()

    def _build_chain(self):
        sys_message = (
            "Sen bir arama motoru filtresisin.\n"
            "KURAL 1: Kullanıcının sorusu Türk Ceza Kanunu, hukuk veya adaletle ilgiliyse, soruyu arama motoru için düzelt ve YALNIZCA düzeltilmiş soruyu yaz.\n"
            "KURAL 2: Soru günlük sohbet (merhaba vb.) veya hukuk dışı saçma bir konuyken (Jüpiter, uzay, uzaylılar, yemek) KESİNLİKLE SADECE [ALAKASIZ] yaz.\n"
            "Asla kendi düşüncelerini, analizlerini veya 'Soru şununla ilgili...' gibi açıklamalar ekleme."
        )
        
        user_message = "Kullanıcı Sorusu: {question}"

        prompt = self.llm_engine.create_prompt(
            system_message=sys_message,
            user_message=user_message,
        )

        return prompt | self.llm | StrOutputParser()
    
    def process_query(self, query: str) -> str:
        """
        Process a user query and rewrite it into multiple different forms to improve retrieval performance.
        """
        try:
            print(f"\033[90mProcessing query: {query}\033[0m")
            result = self.chain.invoke({"question": query}).strip()
            return result

        except Exception as e:
            print(f"\033[91m[ERROR] Failed to process query: {e}\033[0m")
            return query
