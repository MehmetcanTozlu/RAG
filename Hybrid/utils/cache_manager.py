import os
from langchain_community.cache import SQLAlchemyCache
from langchain_core.globals import set_llm_cache
from sqlalchemy import create_engine


class SemanticCacheManager:
    """
    This class is used to manage the semantic cache for the RAG pipeline.
    If user asks the same question again, the cache will be used to return the answer without rerunning the pipeline.
    """
    def __init__(self, db_path: str = "sqlite:///semantic_cache.db"):
        if db_path.startswith("sqlite:///"):
            file_path = db_path.replace("sqlite:///", "")
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                    print(f"\033[94mCache direcory created at: {directory}\033[0m")
                except OSError as e:
                    print(f"\033[91m[ERROR] Failed to create cache directory: {e}\033[0m")
                    
        self.db_path = db_path
        self._init_cache()
    
    def _init_cache(self):
        """
        Initialize the semantic cache. Using SQLALchemyCache for persistence.
        """
        try:
            print(f"\033[94mInitializing semantic cache at {self.db_path}\033[0m")
            engine = create_engine(self.db_path)
            
            set_llm_cache(SQLAlchemyCache(engine))

            print(f"\033[92m[SUCCESS] Semantic cache enabled.\033[0m")
        
        except Exception as e:
            print(f"\033[91m[ERROR] Failed to initialize semantic cache: {e}\033[0m")
            set_llm_cache(None)
