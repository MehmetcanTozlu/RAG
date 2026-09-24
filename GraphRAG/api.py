import os
import uuid
import uvicorn
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from app import get_db_credentials, connect_to_database
from graph_qa import GraphQA
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache # for local test
from langchain_community.cache import RedisCache # for production

from prometheus_fastapi_instrumentator import Instrumentator # for tracking API requests
from celery import Celery


load_dotenv()

app = FastAPI(
    title="GraphRAG API",
    description="Microservice for Question Answering over Turkish Criminal Law Knowledge Graph",
    version="3.0"
)

# Prometheus Instrumenter - Saves Metrics into Prometheus.
Instrumentator().instrument(app).expose(app)

# Queue Manager - Redis (Broker: Message Queue, Backend: Result Storage)
celery_app = Celery(
    "api",
    broker=os.getenv("REDIS_URI"),
    backend=os.getenv("REDIS_URI"),
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


def initialize_system():
    """
    Initializes the QA system for both sync and async endpoints.
    """
    global qa_system
    if qa_system is not None:
        return qa_system
    print("\033[96mInitializing GraphRAG System Core...\033[0m")

    # Cache Initialization
    cache = SQLiteCache(database_path=os.getenv("CACHE_DB_PATH"))
    set_llm_cache(cache)

    # Redis - For Production
    # r = redis.Redis.from_url(os.getenv("REDIS_URI"))
    # set_llm_cache(RedisCache(redis_=r))

    # Database Connection
    db_config = {
        "uri": os.getenv("NEO4J_URI"),
        "user": os.getenv("NEO4J_USER"),
        "password": os.getenv("NEO4J_PASSWORD"),
    }
    graph_db = connect_to_database(db_config)

    # LLM Initialization
    llm = ChatOpenAI(
        base_url=os.getenv("LLM_SERVER_URL"),
        api_key="not-needed",
        max_tokens=2048,
        temperature=0.0,
    )

    # QA System Initialization
    qa_system = GraphQA(
        llm=llm,
        graph_db=graph_db,
        db_uri=db_config["uri"],
        db_user=db_config["user"],
        db_password=db_config["password"],
        embedding_model_name=os.getenv("EMBEDDING_MODEL_PATH"),
    )
    print("\033[92mGraphRAG System is ready to use.\033[0m")

    return qa_system

# Load model once at startup for memoization
@app.on_event("startup")
def startup_event():
    initialize_system()

# Process the question (background)
@celery_app.task(name="process_rag_query")
def process_rag_query(question: str):
    """
    This function is executed in a background worker.
    It initializes the QA system and processes the question.
    """
    system = initialize_system()
    result = asyncio.run(system.ask(question=question))
    return result["answer"]

# ======== ENDPOINTS ========
# REST Endpoint for Question Answering
@app.post("/ask", response_model=QueryResponse)
async def ask_question_direct(request: QueryRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")
    try:
        result = await qa_system.ask(question=request.question)
        return QueryResponse(answer=result["answer"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Takes question but doesn't wait for answer
@app.post("/ask-async")
async def ask_question_queue(request: QueryRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Question is required")
    task = process_rag_query.delay(request.question) # send task to queue
    # Gives just one Task ID every time so api saves locking system from being crashed or overloaded.
    return {"message": "Your question has been added to the queue. Check the status with the task ID: " + str(task.id)}

# Users ask this endpoint to get the answer.
@app.get("/status/{task_id}")
async def get_task_status(task_id: str):
    task_result = celery_app.AsyncResult(task_id)
    if task_result.ready():
        if task_result.successful():
            return {"status": "Done", "answer": task_result.result}
        else:
            return {"status": "Error", "answer": str(task_result.result)}

    return {"status": "Pending"}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
