import os
from rag_setup.ingestor import configure_rag  # Your loading/splitting logic
from rag_setup.vector_store import VectorStoreFactory
from rag_setup.reranker import RerankService
from tools import retrieval_tool
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent
from urllib.parse import urlparse


def detect_source_type(path: str) -> str:
    """
    Returns:
        - 's3'  → S3 URL
        - 'web' → Normal web URL
        - 'pdf' / 'docx' → Local file
    """

    parsed = urlparse(path)

    # Check S3 protocol style (s3://)
    if parsed.scheme == "s3":
        return "s3"

    # Check HTTP(S) S3 URL
    if parsed.scheme in ("http", "https") and parsed.netloc:
        if "s3.amazonaws.com" in parsed.netloc or ".s3." in parsed.netloc:
            return "s3"
        return "web"

    # Check file extension
    lower_path = path.lower()

    if lower_path.endswith(".pdf"):
        return "pdf"

    if lower_path.endswith(".docx"):
        return "docx"

    raise ValueError("Unsupported source type")

def execute_rag_flow(query: str,  path: str, agent_name: str, chunk_size: int, chunk_overlap: int):
    """
    Complete RAG Pipeline: Ingest -> Split -> Embed -> Store -> Retrieve -> Rerank
    """

    source_type = detect_source_type(path)
    print(f"--- Starting RAG Flow for {source_type} ---")

    # 1. INGESTION & SPLITTING
    # This calls your WebBaseLoader/PyPDFLoader logic and returns splits
    print(f"1. Loading and splitting documents from: {path}")
    splits = configure_rag(agent_name=agent_name, source_type=source_type, path=path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 2. CREATE VECTOR STORE
    # This clears the temp directory and saves the new splits
    print(f"2. Creating  vector store in temporary storage...")
    embeddings = VectorStoreFactory.get_embeddings()
    VectorStoreFactory.create_store(agent_name, splits, embeddings)

    execution_context = {
        "agent_name": agent_name,
        "query": query
    }
    result = retrieval_tool.invoke({"execution_context": execution_context})

    content = result.get("Serialized") or result.get("content") or result.get("serialized")
    artifacts = result.get("Documents") or result.get("documents") or result.get("Artifacts") or []
    
    print("--- CONTENTS ---")
    print(f"Type of content: {type(content)}")
    # print("Content Preview:")
    # print(content)

    print("--- ARTIFACTS ---")
    print(f"Number of reranked documents returned: {len(artifacts)}")
    # for doc in artifacts:
    #     print(f"Artifact Source: {doc.metadata}")
    #     print(f"Artifact Content: {doc.page_content}")
    
    print("--- END OF RETRIEVAL TOOL OUTPUT ---")
    print("=========================================================")
    print("=========================================================")

    print("--- LLM CONFIG ---")
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0,
        # max_tokens=200,
    )
    print("Generating answer using RAG Agent")
    tools = [retrieval_tool]
    prompt = (
        f"You have access to a tool named 'retrieval_tool' which accepts a single argument "
        "structured as: {\"execution_context\": {\"agent_name\": \"<agent_name>\", \"query\": \"<user query>\"}}.\n"
        f"When you call retrieval_tool for this session, include the agent_name \"{agent_name}\" in the execution_context "
        "and set the query to the user's query. Only call the tool with that structure."
    )
    agent = create_agent(llm, tools, system_prompt=prompt)
    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


    # # 6. EXECUTE AGENT
    # print(f"--- Agent executing query: {query} ---")
    # result = agent_executor.invoke({"input": query})
    # print("--- AGENT RESULT ---")
    # print(result)
    print(f"--- Streaming Agent Response for query: {query} ---")
    messages = [
        {"role": "system", "content": f"agent_name: {agent_name}"},  # makes agent aware of the active agent
        {"role": "user", "content": query},
    ]
    for event in agent.stream({"messages": messages}, stream_mode="values"):
        event["messages"][-1].pretty_print()
    return result


    

# --- Example of how to call this ---
if __name__ == "__main__":
    # Ensure your Environment Variables are set for Azure OpenAI!
    
    result = execute_rag_flow(
        # query="How does the politeness level of prompts influence the accuracy of large language models according to “Mind Your Tone,” and what experimental evidence supports this?",
        query="What is Memory?",
        path="https://lilianweng.github.io/posts/2023-06-23-agent/",
        # path="docs/5_pager.pdf",
        # store_type="faiss",
        agent_name="my_agent",
        chunk_size=500,
        chunk_overlap=100
    )
    
    if result:
        print("RAG Pipeline executed successfully")
    else:
        print("Error")
    # print("FINAL RESULTS:")
    # print(result)