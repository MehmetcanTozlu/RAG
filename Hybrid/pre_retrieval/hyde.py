from langchain_core.output_parsers import StrOutputParser


class HyDEGenerator:
    """
    Generates an abstract answer to the query using LLM
    and uses it to perform semantically enriched retrieval
    """
    def __init__(self, llm_engine):
        self.llm_engine = llm_engine
        self.llm = llm_engine.get_llm()
        self.chain = self._build_chain()
    
    def _build_chain(self):
        sys_message = (
            "Sen bir Türk Ceza Hukuku profesörüsün. Görevin, kullanıcının sorusuna "
            "sanki gerçek bir kanun maddesiymiş gibi HAYALİ (hypothetical) bir paragraf yazmaktır. "
            "Yazdığın bilginin %100 doğru olması önemli değildir, sadece hukuki bir dille, "
            "kanun metninde geçebilecek anahtar kelimeleri ve terimleri içeren bir taslak (context) oluştur. "
            "Asla 'Bu böyledir' veya 'Benim görüşüm' deme, doğrudan hayali kanun metnini yaz."
        )

        user_message = "Soru: {question}\nHayali Kanun Metni:"

        prompt = self.llm_engine.create_prompt(
            system_message=sys_message,
            user_message=user_message,
        )

        return prompt | self.llm | StrOutputParser()
    
    def generate(self, query: str) -> str:
        """
        Generates an abstract answer to the query using LLM
        """
        try:
            print(f"\033[90mHyDE: Generating abstract answer..\033[0m")
            hypothetical_doc = self.chain.invoke({"question": query}).strip()
            return hypothetical_doc

        except Exception as e:
            print(f"\033[91m[ERROR] Failed to generate abstract answer: {e}\033[0m")
            return query
