from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Dict, Any
import os
from pydantic import BaseModel
import logging
from prometheus_client import make_asgi_app

from app.utils.document_processor import DocumentProcessor
from app.rag.rag_engine import RAGEngine
from app.agents.agents import AgentSystem
from app.config import APP_NAME, DEBUG
from app.utils.logging_config import setup_logging
from app.utils.metrics import (
    track_request_metrics,
    track_document_processing,
    track_rag_query,
    track_agent_processing,
)
from app.utils.health import HealthCheck

# Set up logging
setup_logging(log_level="INFO" if DEBUG else "WARNING")


# Request/Response Models
class QuestionRequest(BaseModel):
    question: str
    model: str = "llama3.2"


class QuestionResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    agent_status: Dict[str, Any]


app = FastAPI(
    title=APP_NAME,
    description="An Agentic RAG Application that combines RAG with autonomous agents",
    version="1.0.0",
    debug=DEBUG,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Mount Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Initialize components
document_processor = DocumentProcessor()
rag_engine = RAGEngine()
agent_system = AgentSystem()
health_check = HealthCheck()


@app.post("/upload", response_model=Dict[str, str])
@track_request_metrics
async def upload_document(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Upload and process a document.

    Args:
        file: The document file to upload

    Returns:
        Dict containing success message and filename

    Raises:
        HTTPException: If document processing fails
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        # Save the uploaded file
        file_content = await file.read()
        file_path = document_processor.save_document(file_content, file.filename)

        # Extract text from the document
        texts = document_processor.extract_text(file_path)

        # Skip RAG processing during upload for faster response
        # RAG processing will happen during first question
        # rag_engine.process_documents(texts)

        # Skip agent processing during upload to improve performance
        # Agent processing can be done during question answering if needed
        # agent_system.process_documents([file_path])  # This was causing delays

        return {
            "message": "Document uploaded successfully - processing deferred",
            "filename": file.filename,
        }
    except Exception as e:
        logging.error(f"Document processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Document processing failed: {str(e)}"
        )


@app.post("/ask", response_model=QuestionResponse)
@track_request_metrics
async def ask_question(request: QuestionRequest) -> QuestionResponse:
    """
    Ask a question about the uploaded documents.

    Args:
        request: QuestionRequest containing the question and model

    Returns:
        QuestionResponse containing the answer, sources, and agent status

    Raises:
        HTTPException: If question processing fails
    """
    try:
        # Get answer from RAG engine
        rag_response = await track_rag_query(request.model)(rag_engine.query)(
            request.question, request.model
        )

        # Process with agent system
        agent_response = await track_agent_processing("question_answering")(
            agent_system.answer_question
        )(request.question)

        return QuestionResponse(
            answer=rag_response["answer"],
            sources=rag_response["source_documents"],
            agent_status=agent_response,
        )
    except Exception as e:
        logging.error(f"Question processing failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Question processing failed: {str(e)}"
        )


@app.get("/health")
@track_request_metrics
async def health_check_endpoint() -> Dict[str, Any]:
    """
    Health check endpoint that checks all system dependencies.

    Returns:
        Dict containing health status of all components
    """
    try:
        health_status = await health_check.check_all()
        return health_status
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/health/simple")
@track_request_metrics
async def simple_health_check() -> Dict[str, str]:
    """
    Simple health check endpoint.

    Returns:
        Dict containing basic health status
    """
    return {"status": "healthy"}
