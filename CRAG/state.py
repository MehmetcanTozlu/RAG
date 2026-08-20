from typing import TypedDict, List
from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Graph state for the CRAG system.
    Each node takes GraphState and returns it with updated values
    """
    question: str
    generation: str
    web_fallbak: bool
    documents: List[Document]
