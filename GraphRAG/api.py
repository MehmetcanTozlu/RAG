import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from app import get_db_credentials, connect_to_database
from graph_qa import GraphQA
from langchain_openai import ChatOpenAI


load_dotenv()

app = FastAPI(
    title="GraphRAG API",
    description="Microservice for Question Answering over Turkish Criminal Law Knowledge Graph",
    version="1.0"
)

# Global variable to store QA system instance
# Load the QA system once at startup (memoization)
qa_system = None


class QueryRequest(BaseModel):
    """Request model for querying the knowledge graph."""
    question: str


class QueryResponse(BaseModel):
    """Response model for querying the knowledge graph."""
    answer: str


# Load the model once at startup for memoization
@app.on_event("startup")
def startup_event():
    global qa_system
    print("\033[96mStarting GraphRAG API (Microservice Mode)...")

    db_config = {
        "uri": os.getenv("NEO4J_URI"),
        "user": os.getenv("NEO4J_USER"),
        "password": os.getenv("NEO4J_PASSWORD"),
    }
    graph_db = connect_to_database(db_config)

    llm = ChatOpenAI(
        base_url=os.getenv("LLM_SERVER_URL"),
        api_key="not-needed",
        max_tokens=2048,
        temperature=0.0,
    )

    qa_system = GraphQA(
        llm=llm, 
        graph_db=graph_db,
        db_uri=db_config["uri"],
        db_user=db_config["user"],
        db_password=db_config["password"],
        embedding_model_name=os.getenv("EMBEDDING_MODEL_PATH"),
    )
    print("\033[92mGraphRAG API is ready to use.\033[0m")

# REST Endpoint for Question Answering
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    try:
        result = await qa_system.ask(question=request.question)
        return QueryResponse(answer=result["answer"])
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
