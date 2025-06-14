import gradio as gr
from app.utils.document_processor import DocumentProcessor
from app.rag.rag_engine import RAGEngine
from app.agents.agents import AgentSystem
from app.config import AGENT_CONFIG
import os
import time

# Initialize components
document_processor = DocumentProcessor()
rag_engine = RAGEngine()
# Defer agent system initialization to avoid slow startup
agent_system = None

uploaded_files = []


def upload_document_gradio(file):
    start_time = time.time()

    if file is None:
        return "No file uploaded."
    try:
        # Debug logging to see the structure
        print(f"File object type: {type(file)}")
        print(f"File object: {file}")

        # Get the original filename from the file object
        original_filename = file.name if hasattr(file, "name") else "uploaded_file"

        # Ensure the file has an extension
        if not any(
            original_filename.lower().endswith(ext) for ext in [".pdf", ".docx", ".txt"]
        ):
            # If no extension, try to determine from file type
            if hasattr(file, "type"):
                if "pdf" in file.type.lower():
                    original_filename += ".pdf"
                elif "word" in file.type.lower():
                    original_filename += ".docx"
                else:
                    original_filename += ".txt"
            else:
                original_filename += ".txt"  # Default to txt if we can't determine

        # Save and extract text from document
        print(f"⏱️ Starting document processing: {original_filename}")
        file_path = document_processor.save_document(file, original_filename)
        save_time = time.time()
        print(f"✅ File saved in {save_time - start_time:.2f}s")

        texts = document_processor.extract_text(file_path)
        extract_time = time.time()
        print(f"✅ Text extracted in {extract_time - save_time:.2f}s")

        # Skip RAG processing during upload for maximum speed
        # RAG processing will happen during first question
        print("⚡ Skipping RAG processing during upload for faster response")

        # Process with RAG engine (this might be the bottleneck)
        # print("⏱️ Processing with RAG engine...")
        # rag_engine.process_documents(texts)
        # rag_time = time.time()
        # print(f"✅ RAG processing completed in {rag_time - extract_time:.2f}s")

        # Skip agent processing during upload - it's slow and unnecessary
        # agent_system.process_documents([file_path])  # This was causing the delay

        uploaded_files.append(file_path)
        total_time = time.time() - start_time
        return f"✅ Successfully uploaded: {original_filename} (took {total_time:.2f}s) - RAG processing deferred"
    except Exception as e:
        error_time = time.time() - start_time
        print(f"❌ Error after {error_time:.2f}s: {str(e)}")
        return f"❌ Error processing file: {str(e)}"


def ask_question_gradio(question, model_name):
    global agent_system
    if not uploaded_files:
        return "⚠️ Please upload a document first.", ""
    try:
        # Process documents if not already done
        if rag_engine.vector_store is None:
            print("⏱️ Processing uploaded documents for RAG...")
            all_texts = []
            for file_path in uploaded_files:
                texts = document_processor.extract_text(file_path)
                all_texts.extend(texts)
            rag_engine.process_documents(all_texts)
            print("✅ RAG processing completed")

        rag_response = rag_engine.query(question, model_name)

        # Initialize agent system only when needed for questions
        if agent_system is None:
            print("⏱️ Initializing agent system...")
            agent_system = AgentSystem()
            print("✅ Agent system initialized")

        agent_response = agent_system.answer_question(question)
        answer = rag_response["answer"]
        sources = rag_response["source_documents"]
        sources_str = "\n".join([f"📄 {str(s)}" for s in sources])
        return answer, sources_str
    except Exception as e:
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
