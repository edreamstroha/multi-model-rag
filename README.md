# Multimodal Local RAG System

## Project Overview

This project implements a sophisticated Retrieval-Augmented Generation (RAG) system designed to interact with PDF documents that contain both text and images, including scanned PDFs where text is embedded within images. Unlike traditional RAG systems that only process text, this solution leverages a multimodal approach to understand and index both textual content (via OCR for images and direct text extraction for native PDFs) and visual content (by generating descriptions of images using a Vision-Language Model).

The entire system is designed to run locally, making use of [Ollama](https://ollama.com/) for serving the Large Language Models (LLMs) and Vector Databases. This ensures data privacy and allows for offline operation, making it ideal for sensitive documents or environments with limited internet access.

## Features

*   **PDF Processing:** Extracts content from PDF documents, handling both native text and images.
*   **Optical Character Recognition (OCR):** Utilizes Tesseract OCR to extract text from scanned PDF pages (pages treated as images).
*   **Image Description:** Employs a local Vision-Language Model (Moondream2) to generate textual descriptors for images found within the PDF pages.
*   **Multimodal Indexing:** Combines extracted text (from OCR or native PDFs) and image descriptions into a unified knowledge base.
*   **Local Vector Database:** Uses ChromaDB to store and manage vector embeddings of all content.
*   **Local LLM (Llama 3):** Leverages a quantized version of Llama 3 via Ollama for generating natural language responses.
*   **Retrieval-Augmented Generation (RAG):** Grounds LLM responses in the indexed content, reducing hallucinations and providing verifiable answers.
*   **Contextual Sourcing:** The LLM is prompted to cite the source document and page number for its answers, including whether the information came from text or an image description.

## Architecture Diagram

```mermaid
graph TD
    subgraph "Indexing Phase (Offline)"
        A[PDF Page (as an Image)] --render page image--> A1(PyMuPDF);
        A1 --> B(OCR Model<br><b>Tesseract</b>);
        A1 --> C(Vision Model<br><b>Moondream2</b>);

        B --> D[Extracted Text (from OCR)];
        C --> E[Visual Description (from Moondream2)];

        subgraph "Combined Document Creation"
            D & E --> F{LangChain Document Objects<br>with Metadata};
            F --> G(Embedding Model<br><b>all-MiniLM-L6-v2</b>);
            G --> H[Store in Vector DB<br><b>ChromaDB</b>];
        end
    end

    subgraph "Querying Phase (Live)"
        I[User Query] --> J(Embedding Model<br><b>all-MiniLM-L6-v2</b>);
        J --> K{Retrieve from Vector DB};
        H --> K;
        K --> L[Relevant Chunks<br>(Text & Image Descriptors)];
        I & L --> M(Generative LLM<br><b>Llama 3 8B Instruct</b>);
        M --> N[Final Answer];
    end
```

## Prerequisites

Before running this project, ensure you have the following installed and set up:

1.  **Ollama:**
    *   Download and install Ollama from [ollama.com](https://ollama.com/).
    *   Ensure the Ollama application is running in the background.

2.  **Ollama Models:**
    *   Open your terminal and pull the necessary models using Ollama:
        ```bash
        ollama pull moondream
        ollama pull all-minilm
        ollama pull llama3:8b-instruct-q4_K_M
        ```
    *   *Note: `llama3:8b-instruct-q4_K_M` is a 4-bit quantized version, which should fit within devices with 6GB of VRAM.*

3.  **Python 3.10+**

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/edreamstroha/multimodal-local-rag.git
    cd multimodal-local-rag
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: .\venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    ```
    langchain
    langchain-community
    langchain-core
    ollama
    chromadb
    PyMuPDF
    fpdf
    pytesseract
    Pillow
    ```
## Usage

To run the Multimodal Local RAG system:

1.  **Ensure all [Prerequisites](#prerequisites) are met.** Specifically, Ollama and its models must be running.
2.  **Execute the Python script:**
    ```bash
    python multimodal_rag.py
    ```

The script will perform the following actions:
*   Create a `demo_scanned_document.pdf` (if it doesn't exist) which contains an image with text elements.
*   Process this PDF: running OCR on the page image to extract text and using Moondream2 via Ollama to describe the visual elements of the image.
*   Store the extracted text chunks and image descriptions (along with their metadata) into a local ChromaDB instance.
*   Initialize the RAG chain using Llama 3.
*   Run two example queries and print the generated answers, demonstrating its ability to synthesize information from both textual and visual sources.

## Code Structure

The main script `multimodal_rag.py` is divided into the following logical sections:

*   **Configuration & Constants:** Defines file paths and names.
*   **`create_demo_scanned_pdf()`:** Helper function to generate a test PDF.
*   **`process_multimodal_pdf_with_ocr()`:** The core indexing logic. It handles PDF rendering, OCR, VLM description, and creates LangChain `Document` objects with appropriate metadata.
*   **`create_rag_chain()`:** Sets up the LangChain Expression Language (LCEL) chain for RAG, including the retriever, LLM, and a sophisticated prompt for context integration and citation.
*   **Main Execution Block (`if __name__ == "__main__":`)**: Orchestrates the entire process from PDF creation, indexing, to querying.

## Customization and Future Improvements

*   **Advanced PDF Parsing:** For complex PDFs with varying layouts, consider using libraries like `unstructured` or building more sophisticated layout analysis to accurately link text blocks to nearby images.
*   **Hybrid Retrieval:** Implement a hybrid search that combines vector similarity with keyword search (e.g., BM25) for more robust retrieval.
*   **Re-ranking:** Integrate a re-ranking model after initial retrieval to ensure the most relevant chunks are always presented to the LLM.
*   **Chat History:** Extend the RAG chain to support multi-turn conversations by managing chat history.
*   **User Interface:** Build a simple web interface (e.g., with Streamlit or Gradio) to interact with the RAG system.
*   **Persistent Storage:** For larger projects, consider persisting your ChromaDB outside of the default location or using a dedicated server.
*   **More Sophisticated Metadata:** Add more granular metadata (e.g., bounding boxes for text/images) if your use case requires spatial awareness beyond just page number.