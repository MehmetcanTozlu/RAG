from typing import List
from langchain_core.output_parsers import CommaSeparatedListOutputParser


class MultiQueryExpander:
    """
    Generates multiple related queries based on the user's original query
    """
    def __init__(self, llm_engine):
        self.llm_engine = llm_engine
        self.llm = llm_engine.get_llm()
        self.chain = self._build_chain()
    
    def _build_chain(self):
        sys_message = (
            "Sen bir arama motoru optimizasyon uzmanısın.\n"
            "Görevin, kullanıcının girdiği hukuki veya genel soruyu almak ve vektör veritabanında (kanun metinlerinde) "
            "en iyi sonucu bulabilmek için bu sorunun AYNI ANLAMA GELEN 3 farklı versiyonunu yazmaktır.\n"
            "Soruların arasında virgül (,) koyarak listele. Asla madde imi, numara veya fazladan açıklama kullanma."
        )

        user_message = "Soru: {question}\nÇeşitlendirilmiş Sorgular:"

        prompt = self.llm_engine.create_prompt(
            system_message=sys_message,
            user_message=user_message,
        )

        return prompt | self.llm | CommaSeparatedListOutputParser()
    
    def generate(self, query: str) -> List[str]:
        """
        Generates multiple related queries based on the user's original query
        """
        try:
            print(f"\033[90mMultiQueryExpander: Generating diversified queries..\033[0m")
            variations = self.chain.invoke({"question": query})
            variations.append(query)
            clean_variations = [q.strip() for q in variations if q.strip()]
            return clean_variations

        except Exception as e:
            print(f"\033[91m[ERROR] Failed to generate diversified queries: {e}\033[0m")
            return [query]