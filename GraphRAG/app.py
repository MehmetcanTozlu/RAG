import argparse
import logging
import warnings
from langchain_neo4j import Neo4jGraph
from langchain_community.chat_models import ChatLlamaCpp
from graph_builder import GraphBuilder

logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
logging.getLogger("langchain_community").setLevel(logging.ERROR)
logging.getLogger("langchain_neo4j").setLevel(logging.ERROR)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_db_credentials(args: argparse.Namespace) -> dict:
    """
    Get database credentials from command-line arguments.
    
    Args:
        args: Command-line arguments containing database connection details.
    
    Returns:
        dict: A dictionary containing database credentials.
    """

    if args.db_mode == "local":
        print(f"\033[94mLocal Neo4j (Docker) is starting...\033[0m")
        return {
            "uri": "bolt://localhost:7687",
            "user": args.local_db_user,
            "password": args.local_db_password,
        }
    
    elif args.db_mode == "cloud":
        print(f"\033[94mCloud Neo4j (AuraDB) is starting...\033[0m")
        return {
            "uri": args.cloud_db_uri,
            "user": args.cloud_db_user,
            "password": args.cloud_db_password,
        }
    
    else:
        raise ValueError(f"\033[91mInvalid database mode: {args.db_mode}\033[0m")

def connect_to_database(db_config: dict) -> Neo4jGraph:
    """
    Connect to the Neo4j database.

    Args:
        db_config: A dictionary containing database connection details.

    Returns:
        Neo4jGraph: A Neo4jGraph object connected to the database.
    """
    if not all(db_config.values()):
        raise ValueError("\033[91mAll database connection details must be provided.\033[0m")

    try:
        # Initialize Neo4j connection
        graph = Neo4jGraph(
            url=db_config["uri"],
            username=db_config["user"],
            password=db_config["password"]
        )

        # Verify connection
        _ = graph.get_schema
        print(f"\033[92mSuccessfully connected to Neo4j!\033[0m")

        return graph
    
    except Exception as e:
        raise ValueError(f"\033[91mFailed to connect to Neo4j: {str(e)}\033[0m")

def main():
    parser = argparse.ArgumentParser(description="Start the GraphRAG application...")

    # DB arguments (local database)
    parser.add_argument("--db_mode", type=str, choices=["local", "cloud"], default="local", help="Database mode: 'local' (Docker) or 'cloud' (AuraDB)")
    parser.add_argument("--local_db_user", type=str, default=None, help="user for the local Neo4j database")
    parser.add_argument("--local_db_password", type=str, default=None, help="Password for the local Neo4j database")

    # DB arguments (cloud database)
    parser.add_argument("--cloud_db_uri", type=str, help="Neo4j AuraDB URI")
    parser.add_argument("--cloud_db_user", type=str, default="neo4j", help="Neo4j AuraDB username")
    parser.add_argument("--cloud_db_password", type=str, help="Neo4j AuraDB password")

    # Model, Document & Inference Tuning Arguments
    parser.add_argument("--model_path", type=str, default="", help="Path to the LLM model")
    parser.add_argument("--embedding_model_path", type=str, default="", help="Path to the embedding model")
    parser.add_argument("--pdf_path", type=str, default="", help="Path to the PDF file to be processed")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for the LLM model")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for the LLM model")
    parser.add_argument("--n_ctx", type=int, default=4096, help="Context window for the LLM model")
    parser.add_argument("--n_gpu_layers", type=int, default=-1, help="Number of GPU layers to use for the LLM model (if using CUDA)")
    parser.add_argument("--verbose", type=bool, default=False, help="Verbose mode for the LLM model")

    args = parser.parse_args()

    print("\n" + "=" * 55)
    print(f"\033[96mInitializing GraphRAG Architecture...\033[0m")
    print("\n" + "=" * 55)

    db_config = get_db_credentials(args)
    graph_db = connect_to_database(db_config=db_config)

    print(f"\033[94mLoading LLM model from {args.model_path}...\033[0m")

    try:
        llm = ChatLlamaCpp(
            model_path=args.model_path,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n_ctx=args.n_ctx,
            verbose=args.verbose,
            n_gpu_layers=args.n_gpu_layers,
        )
        print("\033[92mSuccessfully loaded the LLM model.\033[0m")

    except Exception as e:
        raise ValueError(f"\033[91mFailed to load the LLM model: {str(e)}\033[0m")

    print(f"\n\033[96mChecking the database status..\033[0m")
    try:
        result = graph_db.query("MATCH (n) RETURN count(n) AS node_count")
        node_count = result[0]["node_count"] if result else 0
    except Exception as e:
        node_count = 0

    # Build or Query
    if node_count == 0:
        print(f"\033[93m]The database is empty (0 node). The 'Build' process is starting automatically..\033[0m")

        if not args.pdf_path:
            raise ValueError(f"\033[91mPDF path must be provided for building the knowledge graph.\033[0m")
    
        # Process PDF files to build the knowledge graph
        print(f"\n\033[93mStarting Graph Builder (Entity Extraction)...\033[0m")
        builder = GraphBuilder(
            llm=llm,
            graph_db=graph_db,
            db_uri=db_config["uri"],
            db_user=db_config["user"],
            db_password=db_config["password"],
            embedding_model_name=args.embedding_model_path
        )
        builder.process_and_build(pdf_path=args.pdf_path)
        print("\033[92mThe database has been built. Please restart the application for the QA system.\033[0m")
    
    else:
        print(f"\033[92m{node_count} nodes were found in the database. The 'Build' stage is being skipped and the system is moving directly to 'Query' mode...\033[0m")
        from graph_qa import GraphQA

        qa_system = GraphQA(
            llm=llm,
            graph_db=graph_db,
            db_uri=db_config["uri"],
            db_user=db_config["user"],
            db_password=db_config["password"],
            embedding_model_name=args.embedding_model_path
        )

        while True:
            user_question = input("\n\033[95mYour Question (Type 'q' to quit): \033[0m")
            if user_question.lower().strip() == "q":
                print("\n\033[94mExiting the application...\033[0m")
                break
            
            if user_question.strip():
                print("\n\033[93mThinking...\033[0m")
                qa_system.ask(question=user_question)

if __name__ == '__main__':
    main()
