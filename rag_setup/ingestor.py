import bs4
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.helper import extract_bucket_and_key, fetch_document_by_type

def configure_rag(agent_name, source_type, path, chunk_size=1000, chunk_overlap=200):
    """
    Configures RAG for PDF, DOCX, or Web sources.
    """
    docs = []

    # 1. Handle Web
    if source_type == "web":
        loader = WebBaseLoader(
            web_paths=(path,),
            bs_kwargs={"parse_only": bs4.SoupStrainer(True)}
        )
        docs = loader.load()

    # 2. Handle S3 (PDF & DOCX Only)
    else:
        bucket, key, file_type = extract_bucket_and_key(path)
        
        if file_type not in ["pdf", "docx"]:
            raise ValueError(f"Unsupported file type: {file_type}. Only PDF and DOCX are allowed.")

        # Fetch using your helper
        data = fetch_document_by_type(bucket, key, file_type)
        raw_content = data.get("content", "")

        # Normalize content: DOCX helper returns a list, PDF returns a string
        text = "\n".join(raw_content) if isinstance(raw_content, list) else raw_content

        # Create the standard LangChain Document
        docs = [Document(page_content=text, metadata={"source": agent_name, "file_type": file_type})]

    # 3. Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    splits = splitter.split_documents(docs)

    # 4. Store in PGVector
    print(f"Creating PGVector store for '{agent_name}' with {len(splits)} chunks...")
    # embeddings = VectorStoreFactory.get_embeddings()
    # VectorStoreFactory.create_store(agent_name, splits, embeddings)

    return splits