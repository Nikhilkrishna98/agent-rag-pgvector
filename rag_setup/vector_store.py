import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

class VectorStoreFactory:
    # Use the psycopg3 driver as required by the new langchain-postgres package
    # Format: postgresql+psycopg://user:password@host:port/dbname
    CONNECTION_STRING = os.getenv("POSTGRES_URL")

    @staticmethod
    def get_embeddings():
        return AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
            api_version=os.getenv("AZURE_EMBEDDING_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )

    @classmethod
    def _get_collection_name(cls, agent_name: str):
        """Helper to format the collection name consistently."""
        return f"{agent_name.lower()}_collections"

    @classmethod
    def create_store(cls, agent_name: str, docs, embeddings):
        """
        Creates a new store for a specific agent.
        pre_delete_collection=True ensures that if an agent's store already exists, 
        it is wiped and replaced with the new documents (similar to your old shutil.rmtree logic).
        """
        collection_name = cls._get_collection_name(agent_name)
        
        return PGVector.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=collection_name,
            connection=cls.CONNECTION_STRING,
            use_jsonb=True,
            pre_delete_collection=False
        )

    @classmethod
    def load_active_store(cls, agent_name: str):
        """Loads the existing PGVector store for the specific agent."""
        embeddings = cls.get_embeddings()
        collection_name = cls._get_collection_name(agent_name)
        
        return PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=cls.CONNECTION_STRING,
            use_jsonb=True
        )

#=========================================================
# FOR TESTING WITH CHROMADB
#=========================================================

# import os
# import shutil
# import tempfile
# from langchain_openai import AzureOpenAIEmbeddings
# from langchain_chroma import Chroma

# class VectorStoreFactory:
#     # Everything lives here in the system temp folder
#     BASE_STORAGE_PATH = os.path.join(tempfile.gettempdir(), "mcp_rag_storage")

#     @staticmethod
#     def get_embeddings():
#         return AzureOpenAIEmbeddings(
#             azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#             azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
#             api_version=os.getenv("AZURE_EMBEDDING_API_VERSION"),
#             api_key=os.getenv("AZURE_OPENAI_API_KEY")
#         )

#     @classmethod
#     def _get_agent_paths(cls, agent_name: str):
#         """Returns the specific directory and collection name for an agent."""
#         agent_dir = os.path.join(cls.BASE_STORAGE_PATH, agent_name.lower())
#         collection_name = f"{agent_name.lower()}_collections"
#         return agent_dir, collection_name

#     @classmethod
#     def create_store(cls, agent_name: str, docs, embeddings):
#         """
#         Creates a new Chroma store for a specific agent.
#         Deletes the agent's specific folder before creation to ensure a fresh start.
#         """
#         agent_dir, collection_name = cls._get_agent_paths(agent_name)
#         if not docs:
#             raise ValueError(f"No documents provided to create store for agent '{agent_name}'. Aborting.")
#         # Clear old data for THIS specific agent only
#         if os.path.exists(agent_dir):
#             shutil.rmtree(agent_dir)
        
#         # Create and persist the store
#         return Chroma.from_documents(
#             documents=docs,
#             embedding=embeddings,
#             persist_directory=agent_dir,
#             collection_name=collection_name
#         )

#     @classmethod
#     def load_active_store(cls, agent_name: str):
#         """Loads the existing Chroma store for the specific agent."""
#         agent_dir, collection_name = cls._get_agent_paths(agent_name)
        
#         if not os.path.exists(agent_dir):
#             return None

#         embeddings = cls.get_embeddings()
        
#         return Chroma(
#             persist_directory=agent_dir,
#             embedding_function=embeddings,
#             collection_name=collection_name
#         )