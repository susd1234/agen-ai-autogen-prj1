from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from app.config import CHUNK_SIZE, CHUNK_OVERLAP, VECTOR_DB_DIR, AGENT_CONFIG


class RAGEngine:
    def __init__(self):
        # Defer embedding initialization to avoid slow startup
        self.embeddings = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        self.vector_store = None
        self.qa_chain = None
        self.current_model = AGENT_CONFIG["model"]

    def _get_embeddings(self):
        """Lazy initialization of embeddings."""
        if self.embeddings is None:
            # Use Ollama embeddings if using llama model, otherwise OpenAI
            if AGENT_CONFIG["model"] == "llama3.2":
                self.embeddings = OllamaEmbeddings(
                    model="bge-m3", base_url="http://localhost:11434"
                )
            else:
                self.embeddings = OpenAIEmbeddings()
        return self.embeddings

    def _get_llm(self, model_name: str = None):
        """Get the appropriate LLM based on the model name."""
        model_name = model_name or self.current_model
        model_config = AGENT_CONFIG["model_configs"][model_name]

        if model_name == "llama3.2":
            return Ollama(
                model="llama3.2:latest",
                base_url=model_config["base_url"],
                temperature=AGENT_CONFIG["temperature"],
                timeout=120,
            )
        else:
            return ChatOpenAI(
                temperature=AGENT_CONFIG["temperature"],
                max_tokens=AGENT_CONFIG["max_tokens"],
                model=model_name,
                api_key=model_config["api_key"],
            )

    def set_model(self, model_name: str):
        """Set the current model to use."""
        if model_name not in AGENT_CONFIG["available_models"]:
            raise ValueError(f"Model {model_name} not supported")
        self.current_model = model_name
        if self.qa_chain:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self._get_llm(model_name),
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(),
            )

    def process_documents(self, texts: List[str]) -> None:
        """Process documents and create vector store."""
        # Split texts into chunks
        chunks = self.text_splitter.create_documents(texts)

        # Create or update vector store with lazy embedding initialization
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self._get_embeddings(),
            persist_directory=VECTOR_DB_DIR,
        )

        # Defer QA chain initialization until first query to speed up document processing
        # self.qa_chain = RetrievalQA.from_chain_type(
        #     llm=self._get_llm(),
        #     chain_type="stuff",
        #     retriever=self.vector_store.as_retriever(),
        # )
        self.qa_chain = None  # Will be initialized on first query

    def query(self, question: str, model_name: str = None) -> Dict:
        """Query the RAG system with a question."""
        if not self.vector_store:
            raise ValueError(
                "RAG system not initialized. Please process documents first."
            )

        # Initialize QA chain if not already done or if model changed
        if not self.qa_chain or (model_name and model_name != self.current_model):
            if model_name and model_name != self.current_model:
                self.set_model(model_name)
            else:
                # Initialize QA chain with current model
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self._get_llm(),
                    chain_type="stuff",
                    retriever=self.vector_store.as_retriever(),
                )

        response = self.qa_chain.invoke({"query": question})
        return {
            "answer": response["result"],
            "source_documents": response.get("source_documents", []),
        }
