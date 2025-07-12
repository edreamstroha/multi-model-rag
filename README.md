# Multimodal PDF RAG System

This project implements a Retrieval-Augmented Generation (RAG) system capable of processing multimodal PDF documents (combining OCR text and visual descriptions) and answering user queries. It consists of a FastAPI backend for data ingestion and querying, and a React frontend for user interaction.

## Features

*   **Multimodal PDF Processing:** Extracts text via OCR and generates visual descriptions of images/layouts using a Vision Language Model (VLM).
*   **Vector Store:** Stores processed document chunks and visual descriptions using `ChromaDB` for efficient retrieval.
*   **RAG Chain:** Leverages Ollama models (`llama3` and `all-minilm`) to retrieve relevant information and generate coherent answers.
*   **FastAPI Backend:** Provides RESTful API endpoints for uploading PDFs and querying the RAG system.
*   **React Frontend:** A user-friendly web interface to upload documents and interact with the RAG chatbot.

## Architecture

*   **Backend:** Python (FastAPI, LangChain, PyMuPDF, pytesseract, Pillow, ChromaDB, Ollama client)
*   **Frontend:** JavaScript (React)
*   **Language Models:** Hosted via Ollama (`moondream:latest`, `llama3:8b-instruct-q4_K_M`, `all-minilm:l6-v2`)

## Getting Started

### Prerequisites

Before running the application, ensure you have the following installed:

1.  **Python 3.8+**
2.  **Node.js & npm** (for the React frontend)
3.  **Docker (recommended) or Ollama installed locally:**
    *   Download and install [Ollama](https://ollama.com/download).
    *   Pull the required models:
        ```bash
        ollama pull moondream:latest
        ollama pull llama3:8b-instruct-q4_K_M
        ollama pull all-minilm:l6-v2
        ```
4.  **Tesseract OCR:** Install `tesseract` on your system.
    *   **macOS (Homebrew):** `brew install tesseract`
    *   **Ubuntu/Debian:** `sudo apt install tesseract-ocr`
    *   **Windows:** Download installer from [Tesseract Wiki](https://tesseract-ocr.github.io/tessdoc/Installation.html).

### Installation

#### 1. Backend Setup

Navigate to your backend project root (where `app.py` is located).

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate # On Windows: .\venv\Scripts\activate

# Install Python dependencies
pip install fastapi uvicorn python-multipart pydantic \
            langchain langchain-core langchain-community langchain-text-splitters \
            langchain-ollama pymupdf pytesseract Pillow "chromadb>=0.4.14"
```

#### 2. Frontend Setup

Navigate to your React frontend project root (e.g., `rag-frontend` directory, where `package.json` is located).

```bash
npm install
```

### Running the Application

You need to run both the backend and frontend simultaneously.

#### 1. Start the FastAPI Backend

In your backend project root:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can test endpoints via `http://127.0.0.1:8000/docs`.

#### 2. Start the React Frontend

In your React frontend project root:

```bash
npm start
```

The React app will open in your browser, typically at `http://localhost:3000`.

## Usage

1.  **Upload Documents:**
    *   Go to `http://localhost:3000` in your browser.
    *   Use the "Upload PDF Documents" section to select one or more PDF files.
    *   Click "Upload & Process PDFs". The backend will process them and add them to the vector store.
2.  **Ask Questions:**
    *   Once documents are processed, type your question into the "Ask a Question" textarea.
    *   Click "Get Answer". The RAG system will retrieve relevant information and provide a synthesized answer.

## Project Structure
