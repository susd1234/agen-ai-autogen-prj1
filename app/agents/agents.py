import autogen
from typing import Dict, List
from app.config import AGENT_CONFIG
from app.utils.document_processor import DocumentProcessor


class AgentSystem:
    def __init__(self):
        # Configure for Ollama or OpenAI based on the model
        current_model = AGENT_CONFIG["model"]
        model_config = AGENT_CONFIG["model_configs"][current_model]

        if current_model == "llama3.2":
            self.config_list = [
                {
                    "model": "llama3.2:latest",
                    "base_url": model_config["base_url"],
                    "api_key": "ollama",  # Ollama doesn't need a real API key
                    "api_type": "openai",
                }
            ]
        else:
            self.config_list = [
                {"model": current_model, "api_key": model_config["api_key"]}
            ]
        self.document_processor_util = DocumentProcessor()

        # Create agents
        self.user_proxy = autogen.UserProxyAgent(
            name="User_Proxy",
            system_message="A proxy for the user to interact with the system.",
            code_execution_config={
                "last_n_messages": 3,
                "work_dir": "workspace",
                "use_docker": False,
            },
            max_consecutive_auto_reply=1,  # Limit auto replies to prevent loops
        )

        self.document_processor = autogen.AssistantAgent(
            name="Document_Processor",
            system_message="""You are a document processing expert. Your role is to:
            1. Process uploaded documents using the DocumentProcessor utility
            2. Extract text content from various file formats (PDF, DOCX, TXT)
            3. Prepare the content for the RAG system
            4. Ensure proper chunking and processing of documents
            
            You have access to the DocumentProcessor utility which can:
            - Extract text from PDF files using PdfReader
            - Extract text from DOCX files using python-docx
            - Extract text from TXT files
            - Save uploaded documents to the storage directory""",
            llm_config={
                "config_list": self.config_list,
                "timeout": 30,  # 30 second timeout
            },
        )

        self.qa_specialist = autogen.AssistantAgent(
            name="QA_Specialist",
            system_message="""You are a question-answering specialist. Your role is to:
            1. Understand user questions
            2. Formulate effective queries for the RAG system
            3. Analyze and refine answers
            4. Provide clear and accurate responses""",
            llm_config={
                "config_list": self.config_list,
                "timeout": 30,  # 30 second timeout
            },
        )

        self.rag_coordinator = autogen.AssistantAgent(
            name="RAG_Coordinator",
            system_message="""You are a RAG system coordinator. Your role is to:
            1. Manage the RAG system operations
            2. Coordinate between document processing and QA
            3. Ensure proper retrieval and generation
            4. Maintain context and coherence in responses""",
            llm_config={
                "config_list": self.config_list,
                "timeout": 30,  # 30 second timeout
            },
        )

    def process_documents(self, documents: List[str]) -> Dict:
        """Process documents using the agent system."""
        # Extract text from documents using the DocumentProcessor utility
        processed_texts = []
        for doc_path in documents:
            try:
                texts = self.document_processor_util.extract_text(doc_path)
                processed_texts.extend(texts)
            except Exception as e:
                print(f"Error processing document {doc_path}: {str(e)}")
                continue

        # Simplified processing - avoid multiple chat initiations that cause delays
        try:
            # Single agent interaction with timeout
            response = self.user_proxy.initiate_chat(
                self.document_processor,
                message=f"Documents processed: {len(processed_texts)} text chunks extracted.",
                max_turns=1,  # Limit to single turn to prevent loops
            )
        except Exception as e:
            print(f"Agent processing warning: {str(e)}")
            # Continue processing even if agent interaction fails

        return {
            "status": "success",
            "message": "Documents processed successfully",
            "processed_texts": processed_texts,
        }

    def answer_question(self, question: str) -> Dict:
        """Answer a question using the agent system."""
        # Start with the QA specialist
        chat_initiator = self.user_proxy.initiate_chat(
            self.qa_specialist, message=f"Please answer this question: {question}"
        )

        # The QA specialist will coordinate with the RAG coordinator
        self.qa_specialist.initiate_chat(
            self.rag_coordinator, message=f"Need information for question: {question}"
        )

        return {"status": "success", "message": "Question answered successfully"}
