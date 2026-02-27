import os
# from mcp.server.fastmcp import FastMCP
from langchain_core.tools import tool
from rag_setup.vector_store import VectorStoreFactory
from rag_setup.reranker import RerankService
from dotenv import load_dotenv, dotenv_values 
# from rag_setup.vector_store import PATH
# loading variables from .env file
load_dotenv()
# Initialize FastMCP Server
# mcp = FastMCP("MCP-RAG-Server")

@tool()
def retrieval_tool(agent_name: str = None, query: str = None, execution_context: dict = None):
    """
    Retrieves and processes relevant documents based on the input query.
    This function performs the following steps:
    1. Loads the active vector store using the specified store type.
    2. Conducts a similarity search to fetch initial candidate documents.
    3. Reranks the retrieved documents to refine the results.
    4. Formats the final documents for use by an agent.
    Args:
        query (str): The input query string for which relevant documents are to be retrieved.
    Returns:
        dict: A dictionary containing:
            - "Serialized" (str): A formatted string representation of the final documents.
            - "Documents" (list): A list of the final reranked document objects.
    Raises:
        ValueError: If the vector store could not be initialized.
    """
    # 3. RETRIEVAL
    # We load the store we just created
    # print(f"PATH: {PATH}")
    if execution_context:
        agent_name = execution_context.get("agent_name", agent_name)
        query = execution_context.get("query", query)

    agent_name = agent_name or "default_agent"
    query = query or ""

    if not query:
        raise ValueError("Missing 'query' parameter for retrieval_tool")

    print(f"MCP: Retrieving for agent '{agent_name}' | Query: '{query}'")
    
    # 1. Load the PGVector store for the specific agent
    vector_store = VectorStoreFactory.load_active_store(agent_name)
    if not vector_store:
        # return "Error: Vector store could not be initialized."
        try:
            agent_dir, _ = VectorStoreFactory._get_agent_paths(agent_name)
        except Exception:
            agent_dir = "<unknown>"
        raise FileNotFoundError(
            f"Vector store not found for agent '{agent_name}'. Expected directory: {agent_dir}. "
            "Ensure configure_rag returned non-empty splits and VectorStoreFactory.create_store(...) was called successfully."
        )

    # 4. Similarity Search (Fetch 5 candidates for reranking)
    print("4. Performing similarity search...")
    initial_docs = vector_store.similarity_search(query, k=5)
    print(f"Initial Docs Retrieved: {(initial_docs)}, type: {type(initial_docs)}")
    
    
    # 5. Rerank (Refine to best matches)
    print("5. Reranking documents...")
    final_docs = RerankService.rerank(query, initial_docs)
    print(f"Final Docs after Reranking: {(final_docs)}, type: {type(final_docs)}")
    # 6. Format for Agent
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in final_docs
    )
    # print(f"Type: {type(serialized)} Content: {serialized}")
    # print(f"Type: {type(initial_docs)} Content: {initial_docs}" )
    return {"Serialized": serialized, "Documents": final_docs}



# if __name__ == "__main__":
#     # Start the MCP server
#     mcp.run()