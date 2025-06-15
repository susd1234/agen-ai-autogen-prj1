# Agentic RAG Application Using AutoGen Framework and OLLAMA

An intelligent application that combines Retrieval-Augmented Generation (RAG) with autonomous agents (Agentic AI) to provide enhanced document processing and question answering capabilities.

## Features

- Document upload and processing
- Intelligent question answering using RAG
- Autonomous agent system for enhanced reasoning
- Support for multiple LLM models (Llama and GPT)
- FastAPI backend with Gradio UI
- Document storage and vector database integration

## APP UI
![UI Screenshot](app/static/APP%20UI.jpg)

## Prerequisites

- Python 3.9+
- Local Llama model server

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agen-ai-autogen-prj1
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate 
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here (In case use GPT Model)
```

## Project Structure

```
agen-ai-autogen-prj1/
├── app/
│   ├── agents/         # Autonomous agent implementations
│   ├── api/           # API endpoints and routes
│   ├── models/        # Data models and schemas
│   ├── rag/           # RAG engine implementation
│   ├── static/        # Static files for UI
│   ├── storage/       # Document and vector storage
│   ├── utils/         # Utility functions
│   ├── config.py      # Application configuration
│   ├── main.py        # FastAPI application
│   └── gradio_ui.py   # Gradio interface
├── storage/           # Persistent storage
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Usage

1. Start the application:
```bash
uvicorn app.main:app --reload
```

2. Access the API documentation at `http://localhost:8000/docs`

3. Use the Gradio interface at `http://localhost:8000/static/gradio`

### API Endpoints

- `POST /upload`: Upload and process documents
- `POST /ask`: Ask questions about processed documents
- `GET /health`: Health check endpoint

### Example API Usage

```python
import requests

# Upload a document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload",
        files={"file": f}
    )

# Ask a question
response = requests.post(
    "http://localhost:8000/ask",
    json={
        "question": "What is the main topic of the document?",
        "model": "llama3.2"
    }
)
```


## License

This project is licensed under the MIT License.