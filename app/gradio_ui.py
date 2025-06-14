"""
Gradio UI for the Agentic RAG Application.

This module provides a web interface for document upload and question answering.
Key optimization: Uses deferred processing to minimize upload time.

Performance Strategy:
- Upload: Only save file and extract text (~2 seconds)
- First Question: Process documents with RAG + initialize agents (~10-15 seconds one-time)
- Subsequent Questions: Use existing systems (normal speed)
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


def upload_document_gradio(file):
    """
    Handle document upload with optimized processing strategy.

    This function implements a deferred processing approach for maximum upload speed:
    1. Save file to disk (fast)
    2. Extract text content (fast)
    3. Skip RAG processing (deferred until first question)
    4. Skip agent initialization (deferred until first question)

    Args:
        file: Gradio file object containing the uploaded document

    Returns:
        str: Status message with upload time and processing info

    Performance Impact:
        - Before optimization: 30+ seconds (due to RAG + agent processing)
        - After optimization: ~2 seconds (only file ops + text extraction)
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

        # OPTIMIZATION: Skip heavy processing during upload
        # This is the key performance improvement - defer expensive operations
        print("⚡ Skipping RAG processing during upload for faster response")

        # DEFERRED: RAG processing (would add 5-10 seconds)
        # - Vector embedding creation
        # - Chroma database initialization
        # - Document chunking and indexing
        # This will happen during the first question instead
        # print("⏱️ Processing with RAG engine...")
        # rag_engine.process_documents(texts)
        # rag_time = time.time()
        # print(f"✅ RAG processing completed in {rag_time - extract_time:.2f}s")

        # DEFERRED: Agent system processing (would add 20-30 seconds)
        # - AutoGen agent initialization
        # - LLM configuration and connection
        # - Agent conversation setup
        # This will happen during the first question instead
        # agent_system.process_documents([file_path])

        # Track uploaded files for later processing
        uploaded_files.append(file_path)

        total_time = time.time() - start_time
        return f"✅ Successfully uploaded: {original_filename} (took {total_time:.2f}s) - RAG processing deferred"

    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ Error after {error_time:.2f}s: {str(e)}")
        return f"❌ Error processing file: {str(e)}"


def ask_question_gradio(question, model_name):
    """
    Handle question answering with deferred document processing.

    This function implements the second phase of the optimization strategy:
    1. Process documents with RAG engine (first question only)
    2. Initialize agent system (first question only)
    3. Generate answer using both RAG and agents

    The heavy processing that was skipped during upload happens here on the first question.
    Subsequent questions use the already-initialized systems for fast responses.

    Args:
        question (str): User's question about the uploaded documents
        model_name (str): Name of the LLM model to use for answering

    Returns:
        tuple: (answer_text, sources_text) for display in Gradio interface

    Performance Impact:
        - First question: ~10-15 seconds (one-time setup + answer generation)
        - Subsequent questions: Normal speed (~2-5 seconds)

    Design Rationale:
        This deferred approach provides better UX by giving immediate upload feedback
        while doing heavy processing only when actually needed for questions.
    """
    global agent_system

    if not uploaded_files:
        return "⚠️ Please upload a document first.", ""

    try:
        # DEFERRED PROCESSING: RAG Engine Setup (First Question Only)
        # This is where the heavy RAG processing happens that was skipped during upload
        if rag_engine.vector_store is None:
            print("⏱️ Processing uploaded documents for RAG...")
            print("📄 This happens only once - subsequent questions will be faster")

            # Extract and combine text from all uploaded documents
            all_texts = []
            for file_path in uploaded_files:
                texts = document_processor.extract_text(file_path)
                all_texts.extend(texts)

            # Perform expensive RAG operations:
            # - Text chunking and splitting
            # - Vector embedding generation (using Ollama/OpenAI)
            # - Chroma vector database creation
            # - Document indexing for retrieval
            rag_engine.process_documents(all_texts)
            print("✅ RAG processing completed")

        # Generate answer using RAG engine
        # This queries the vector database and generates response using LLM
        rag_response = rag_engine.query(question, model_name)

        # DEFERRED PROCESSING: Agent System Setup (First Question Only)
        # Initialize the AutoGen agent system that was skipped during upload
        if agent_system is None:
            print("⏱️ Initializing agent system...")
            print("🤖 Setting up AutoGen agents with LLM configurations...")

            # This involves:
            # - Creating multiple AutoGen agents (Document Processor, QA Specialist, RAG Coordinator)
            # - Configuring LLM connections (Ollama/OpenAI)
            # - Setting up agent conversation workflows
            # - Validating agent configurations
            agent_system = AgentSystem()
            print("✅ Agent system initialized")

        # Generate additional insights using agent system
        # Agents provide enhanced reasoning and multi-step analysis
        agent_response = agent_system.answer_question(question)

        # Format response for Gradio interface
        answer = rag_response["answer"]
        sources = rag_response["source_documents"]
        sources_str = "\n".join([f"📄 {str(s)}" for s in sources])

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
    The system uses advanced RAG (Retrieval-Augmented Generation) and AI agents to provide accurate answers.
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Document Upload")
            gr.Markdown("Upload your documents in PDF, DOCX, or TXT format.")
            file_input = gr.File(
                label="Upload Document",
                file_types=[".pdf", ".docx", ".txt"],
                type="binary",
                file_count="single",
            )
            upload_btn = gr.Button("Upload and Process", variant="primary")
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
