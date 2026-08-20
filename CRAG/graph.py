from langgraph.graph import StateGraph, END
from state import GraphState
from nodes import CRAGNodes


def decide_to_generate(state: GraphState):
    """
    Decision node to determine if we should generate response
    """
    print(f"\033[94m---AGENT: DECIDING TO GENERATE RESPONSE---\033[0m")

    web_fallback = state.get("web_fallback", False)

    if web_fallback == False:
        print("\033[93mFallback activated: Proceeding with web search results\033[0m")
        return "web_search"
    else:
        print("\033[92mNo fallback needed: Proceeding with retrieved documents\033[0m")
        return "generate"

def create_crag_graph(llm_engine, retriever):
    """
    Create the CRAG graph with nodes and edges.
    """
    # 1. Initialize the workflow
    workflow = StateGraph(GraphState)
    
    # 2. Define the nodes
    nodes = CRAGNodes(llm_engine=llm_engine, retriever=retriever)

    # 3. Add nodes to the workflow
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("web_search", nodes.web_search)
    workflow.add_node("generate", nodes.generate)
    
    # 4. Define the edges
    workflow.set_entry_point("retrieve")
    
    # Go to grade documents after retrieving
    workflow.add_edge("retrieve", "grade_documents")

    # Decide if we should web search based on whether documents are graded
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "web_search": "web_search", # if web search is needed
            "generate": "generate", # if web search is not needed
        }
    )

    # After web search, go to generate
    workflow.add_edge("web_search", "generate")

    # After generating response, end the workflow
    workflow.add_edge("generate", END)

    # Compile the graph
    app = workflow.compile()

    return app
