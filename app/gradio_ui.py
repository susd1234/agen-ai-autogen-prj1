"""
Gradio UI for the Agentic RAG Application.

This module provides a web interface for document upload and question answering.
Implementation: Immediate RAG processing during upload for ready-to-query vector database.

Processing Strategy:
- Upload: Complete RAG pipeline (save, extract, chunk, embed, store) (~10-15 seconds)
- First Question: Initialize agents + query pre-built vector store (~5-10 seconds)
- Subsequent Questions: Fast queries against existing vector database (~2-5 seconds)
"""

import gradio as gr
from app.utils.document_processor import DocumentProcessor
from app.rag.rag_engine import RAGEngine
from app.agents.agents import AgentSystem
from app.config import AGENT_CONFIG
import os
import time

# Initialize lightweight components immediately
# Heavy components (AgentSystem) are initialized on-demand for better performance
document_processor = DocumentProcessor()
rag_engine = RAGEngine()  # Uses lazy embedding initialization
# Defer agent system initialization to avoid slow startup (~20-30 seconds)
agent_system = None

# Global list to track uploaded files for processing
uploaded_files = []


def _is_generic_response(response: str) -> bool:
    """
    Check if agent response is generic/irrelevant and should be filtered out.

    Args:
        response (str): Agent response text

    Returns:
        bool: True if response should be filtered out
    """
    generic_phrases = [
        "you've entered a single space",
        "please rephrase or provide more context",
        "I'm happy to continue",
        "Could you please rephrase",
        "Let me know, and I'll do my best",
        "Is there a specific topic",
        "Are you thinking of",
        "Do you have something else in mind",
        "I can suggest some questions",
        "feeling stuck",
        "better understand what you're looking for",
    ]

    response_lower = response.lower()
    return any(phrase.lower() in response_lower for phrase in generic_phrases)


def upload_document_gradio(file):
    """
    Handle document upload with immediate RAG processing.

    This function processes documents completely during upload:
    1. Save file to disk
    2. Extract text content
    3. Process with RAG engine (chunking, embedding, vector storage)
    4. Store in vector database for immediate querying

    Args:
        file: Gradio file object containing the uploaded document

    Returns:
        str: Status message with upload time and processing info

    Performance Impact:
        Complete processing during upload including vector embeddings and database storage.
        Upload time: ~10-15 seconds (includes full RAG pipeline)
    """
    start_time = time.time()

    if file is None:
        return "No file uploaded."

    try:
        # Debug logging to understand file object structure
        print(f"File object type: {type(file)}")
        print(f"File object: {file}")

        # Extract filename from Gradio file object
        original_filename = file.name if hasattr(file, "name") else "uploaded_file"

        # Ensure file has proper extension for processing
        # This helps the document processor identify the correct extraction method
        if not any(
            original_filename.lower().endswith(ext) for ext in [".pdf", ".docx", ".txt"]
        ):
            # Try to infer extension from MIME type if available
            if hasattr(file, "type"):
                if "pdf" in file.type.lower():
                    original_filename += ".pdf"
                elif "word" in file.type.lower():
                    original_filename += ".docx"
                else:
                    original_filename += ".txt"
            else:
                original_filename += ".txt"  # Safe default

        # PHASE 1: File Operations (Fast - typically <1 second)
        print(f"⏱️ Starting document processing: {original_filename}")
        file_path = document_processor.save_document(file, original_filename)
        save_time = time.time()
        print(f"✅ File saved in {save_time - start_time:.2f}s")

        # PHASE 2: Text Extraction (Fast - typically <1 second)
        texts = document_processor.extract_text(file_path)
        extract_time = time.time()
        print(f"✅ Text extracted in {extract_time - save_time:.2f}s")

        # PHASE 3: RAG Processing (Full pipeline - 5-10 seconds)
        # Process documents immediately with complete RAG pipeline:
        # - Document chunking and text splitting
        # - Vector embedding generation
        # - Chroma vector database creation/update
        # - Document indexing for retrieval
        print("⏱️ Processing with RAG engine (chunking, embedding, vector storage)...")
        rag_engine.process_documents(texts)
        rag_time = time.time()
        print(f"✅ RAG processing completed in {rag_time - extract_time:.2f}s")

        # Track uploaded files for agent processing
        uploaded_files.append(file_path)

        total_time = time.time() - start_time
        return f"✅ Successfully processed: {original_filename} (took {total_time:.2f}s) - Ready for questions!"

    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ Error after {error_time:.2f}s: {str(e)}")
        return f"❌ Error processing file: {str(e)}"


def ask_question_gradio(question, model_name):
    """
    Handle question answering using pre-processed documents.

    Since documents are now fully processed during upload (including RAG pipeline),
    this function focuses on question answering and agent initialization.

    Args:
        question (str): User's question about the uploaded documents
        model_name (str): Name of the LLM model to use for answering

    Returns:
        tuple: (answer_text, sources_text) for display in Gradio interface

    Performance Impact:
        - First question: ~5-10 seconds (agent initialization + answer generation)
        - Subsequent questions: Normal speed (~2-5 seconds)
    """
    global agent_system

    if not uploaded_files:
        return "⚠️ Please upload a document first.", ""

    # Check if RAG engine is ready (should be ready since processing happens during upload)
    if rag_engine.vector_store is None:
        return "⚠️ Documents not yet processed. Please upload a document first.", ""

    try:
        # Generate answer using RAG engine (documents already processed)
        # This queries the pre-built vector database and generates response using LLM
        print("🔍 Querying RAG engine...")
        rag_response = rag_engine.query(question, model_name)

        # DEFERRED: Agent System Setup (First Question Only)
        # Initialize the AutoGen agent system for enhanced reasoning
        agent_response = None
        try:
            if agent_system is None:
                print("⏱️ Initializing agent system...")
                print("🤖 Setting up AutoGen agents with LLM configurations...")

                # This involves:
                # - Creating multiple AutoGen agents (Document Processor, QA Specialist, RAG Coordinator)
                # - Configuring LLM connections (Ollama/OpenAI)
                # - Setting up agent conversation workflows
                # - Validating agent configurations
                agent_system = AgentSystem()

                # Process documents with agent system
                agent_system.process_documents(uploaded_files)
                print("✅ Agent system initialized")

            # Generate additional insights using agent system
            # Agents provide enhanced reasoning and multi-step analysis
            print("🤖 Getting agent insights...")
            agent_response = agent_system.answer_question(question)
        except Exception as e:
            print(f"⚠️ Agent system failed, continuing with RAG only: {str(e)}")
            agent_response = None

        # Format response for Gradio interface
        answer = rag_response["answer"]
        sources = rag_response["source_documents"]
        sources_str = "\n".join([f"📄 {str(s)}" for s in sources])

        # Format agent response for readability
        if agent_response:
            print(f"🤖 AGENT RESPONSE:")
            print(f"Status: {agent_response.get('status', 'unknown')}")
            print(f"Message: {agent_response.get('message', '')}")
            if "conversation" in agent_response:
                print("Conversation History:")
                for msg in agent_response["conversation"]:
                    print(f"- {msg}")
        else:
            print("🤖 No agent response available")
        return answer, sources_str

    except Exception as e:
        print(f"❌ Error in question processing: {str(e)}")
        return f"❌ Error processing question: {str(e)}", ""


# Create the Gradio interface
with gr.Blocks(title="Agentic RAG Application", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
    # Agentic RAG Application
    
    This application allows you to upload documents and ask questions about their content. 
    The system uses advanced RAG (Retrieval-Augmented Generation) with immediate processing:
    - **Upload**: Documents are chunked, embedded, and stored in vector database (~10-15 seconds)
    - **Questions**: Fast retrieval from pre-processed vector store with AI agent analysis
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Document Upload")
            gr.Markdown(
                "Upload your documents in PDF, DOCX, or TXT format. Processing includes chunking, embedding, and vector storage."
            )
            file_input = gr.File(
                label="Upload Document",
                file_types=[".pdf", ".docx", ".txt"],
                type="binary",
                file_count="single",
            )
            upload_btn = gr.Button("Upload & Process with RAG", variant="primary")
            upload_output = gr.Textbox(
                label="Upload Status", interactive=False, show_copy_button=True
            )

        with gr.Column(scale=1):
            gr.Markdown("### ❓ Ask Questions")
            gr.Markdown("Type your question about the uploaded document(s).")
            model_dropdown = gr.Dropdown(
                choices=AGENT_CONFIG["available_models"],
                value=AGENT_CONFIG["model"],
                label="Select Model",
                info="Choose the AI model to use for answering questions",
            )
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="What would you like to know about the document?",
                lines=3,
                show_copy_button=True,
            )
            ask_btn = gr.Button("Get Answer", variant="primary")
            answer_output = gr.Textbox(
                label="Answer", lines=5, interactive=False, show_copy_button=True
            )
            sources_output = gr.Textbox(
                label="Sources", lines=3, interactive=False, show_copy_button=True
            )

    # Set up event handlers
    upload_btn.click(
        fn=upload_document_gradio, inputs=file_input, outputs=upload_output
    )

    ask_btn.click(
        fn=ask_question_gradio,
        inputs=[question_input, model_dropdown],
        outputs=[answer_output, sources_output],
    )

    # Add some helpful examples
    gr.Examples(
        examples=[
            ["What are the main points in the document?"],
            ["Can you summarize the key findings?"],
            ["What are the recommendations?"],
        ],
        inputs=question_input,
    )

if __name__ == "__main__":
    demo.launch(share=True, show_error=True)
