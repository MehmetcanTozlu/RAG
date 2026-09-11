import argparse
from langchain_neo4j import Neo4jGraph
from langchain_community.chat_models import ChatLlamaCpp
from graph_builder import GraphBuilder


def connect_to_database(args):
    """
    Connect to the Neo4j database.

    Args:
        args: Command-line arguments containing database connection details.

    Returns:
        Neo4jGraph: A Neo4jGraph object connected to the database.
    """
    if args.db_mode == "local":
        print(f"\033[94mLocal Neo4j (Docker) is starting...\033[0m")
        uri = "bolt://localhost:7687"
        user = args.local_db_user
        password = args.local_db_password
    
    elif args.db_mode == "cloud":
        print(f"\033[94mCloud Neo4j (AuraDB) is starting...\033[0m")
        uri = args.cloud_db_uri
        user = args.cloud_db_user
        password = args.cloud_db_password
    
        if not all([uri, user, password]):
            raise ValueError(f"\033[91mAll database connection details must be provided.\033[0m")
    
    else:
        raise ValueError(f"\033[91mInvalid database mode: {args.db_mode}\033[0m")
    
    try:
        # Initialize Neo4j connection
        graph = Neo4jGraph(url=uri, username=user, password=password)

        # Verify connection
        schema = graph.get_schema
        print(f"\033[92mSuccessfully connected to Neo4j!\033[0m")

        return graph
    
    except Exception as e:
        raise ValueError(f"\033[91mFailed to connect to Neo4j: {str(e)}\033[0m")

def main():
    parser = argparse.ArgumentParser(description="Start the GraphRAG application...")

    # DB connection arguments (local database)
    parser.add_argument(
        "--db_mode",
        type=str,
        choices=["local", "cloud"],
        default="local",
        help="Database mode: 'local' (Docker) or 'cloud' (AuraDB)"
    )
    parser.add_argument(
        "--local_db_user",
        type=str,
        default=None,
        help="user for the local Neo4j database"
    )
    parser.add_argument(
        "--local_db_password",
        type=str,
        default=None,
        help="Password for the local Neo4j database"
    )

    # DB connection arguments (cloud database)
    parser.add_argument(
        "--cloud_db_uri",
        type=str,
        help="Neo4j AuraDB URI"
    )
    parser.add_argument(
        "--cloud_db_user",
        type=str,
        default="neo4j",
        help="Neo4j AuraDB username"
    )
    parser.add_argument(
        "--cloud_db_password",
        type=str,
        help="Neo4j AuraDB password"
    )

    # LLM model and document arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to the LLM model"
    )
    parser.add_argument(
        "--pdf_path",
        type=str,
        default="",
        help="Path to the PDF file to be processed"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Temperature for the LLM model"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Max tokens for the LLM model"
    )
    parser.add_argument(
        "--n_ctx",
        type=int,
        default=4096,
        help="Context window for the LLM model"
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Verbose mode for the LLM model"
    )

    args = parser.parse_args()

    print("\n" + "=" * 55)
    print(f"\033[96mInitializing GraphRAG Architecture...\033[0m")
    print("\n" + "=" * 55)

    graph_db = connect_to_database(args=args)

    print(f"\033[94mLoading LLM model from {args.model_path}...\033[0m")

    try:
        llm = ChatLlamaCpp(
            model_path=args.model_path,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n_ctx=args.n_ctx,
            verbose=args.verbose,
            n_gpu_layers=-1,
        )
        print("\033[92mSuccessfully loaded the LLM model.\033[0m")

    except Exception as e:
        raise ValueError(f"\033[91mFailed to load the LLM model: {str(e)}\033[0m")
    
    print("\n\033[96mChecking the database status..\033[0m")
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
        builder = GraphBuilder(llm=llm, graph_db=graph_db)
        builder.process_and_build(pdf_path=args.pdf_path)
        print("\033[92mThe database has been built. Please restart the application for the QA system.\033[0m")
    
    else:
        print(f"\033[92m{node_count} nodes were found in the database. The 'Build' stage is being skipped and the system is moving directly to 'Query' mode...\033[0m")
        from graph_qa import GraphQA

        qa_system = GraphQA(llm=llm, graph_db=graph_db)

        while True:
            user_question = input("\n\033[95mYour Question (Type 'q' to quit): \033[0m")
            if user_question.lower().strip() == "q":
                break
            
            qa_system.ask(question=user_question)

if __name__ == '__main__':
    main()


"""
python app.py --db_mode="local" --local_db_user="neo4j" --local_db_password="12345678" \
    --model_path="/home/mehmet02828/genai_workspace/models/Qwen2.5-7B-Instruct.Q4_K_M.gguf" \
    --pdf_path="/home/mehmet02828/genai_workspace/rag_tutorial/workspace/data/turk_ceza_kanunu.pdf"
"""
