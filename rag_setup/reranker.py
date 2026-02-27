import os
from typing import List
from langchain_openai import AzureChatOpenAI
from langchain_classic.retrievers.document_compressors.listwise_rerank import LLMListwiseRerank
from langchain_core.documents import Document
from dotenv import load_dotenv, dotenv_values 
# loading variables from .env file
load_dotenv()
class RerankService:
    
    @staticmethod
    def rerank(query: str, documents: List[Document]) -> List[Document]:
        """
        Takes a query and a list of LangChain Documents and returns them reordered.
        """
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0  # Keep temperature 0 for ranking consistency
        )
        
        # 2. Setup the Reranker
        # top_n determines how many documents to return after re-ordering
        reranker = LLMListwiseRerank.from_llm(llm=llm, top_n=2)
        return reranker.compress_documents(documents, query)