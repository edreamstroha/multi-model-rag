import base64
import os
import io
import fitz
import pytesseract
from PIL import Image
from typing import List, Dict, Union
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

# Ensure the 'data' directory exists
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# --- Your original RAG processing functions (slightly modified for file handling) ---

def process_multimodal_pdf(pdf_bytes: bytes, pdf_filename: str) -> List[Document]:
    """
    Processes a PDF where each page is an image. It uses OCR for text
    and a VLM for visual descriptions.
    """
    # Create a temporary file to allow fitz to open it directly
    temp_pdf_path = os.path.join(DATA_DIR, f"temp_{pdf_filename}")
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_bytes)

    doc = fitz.open(temp_pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_docs = []
    image_caption_model = ChatOllama(model="moondream:latest", temperature=0)

    print(f"Processing {pdf_filename} with OCR and Vision model...")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        current_page = page_num + 1

        # Render the entire page as a high-quality image
        pix = page.get_pixmap(dpi=300)
        page_image = Image.open(io.BytesIO(pix.tobytes("png")))

        # Save the page image for potential future reference (e.g., in UI)
        page_image_save_path = os.path.join(
            DATA_DIR, f"{pdf_filename.replace('.pdf', '')}_p{current_page}.png"
        )
        page_image.save(page_image_save_path)

        # a) OCR Path: Extract text from the page image
        print(f"  - Running OCR on page {current_page} of {pdf_filename}...")
        try:
            ocr_text = pytesseract.image_to_string(page_image)
            if ocr_text:
                text_chunks = text_splitter.split_text(ocr_text)
                for i, chunk in enumerate(text_chunks):
                    all_docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source_type": "ocr_text",
                                "source_document": pdf_filename,
                                "page_number": current_page,
                            },
                        )
                    )
        except Exception as e:
            print(f"  - Error running OCR on page {current_page} of {pdf_filename}: {e}")
            # Continue processing even if OCR fails for one page

        # b) Vision Path: Get a visual description of the page image
        print(f"  - Running Vision model on page {current_page} of {pdf_filename}...")
        try:
            image_bytes = pix.tobytes("png")
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            human_message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Describe this image in detail for a search index. (Scan each and every part of the page)",
                    },
                    {"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"},
                ]
            )
            response = image_caption_model.invoke([human_message])
            visual_descriptor = response.content

            all_docs.append(
                Document(
                    page_content=visual_descriptor,
                    metadata={
                        "source_type": "visual_description",
                        "source_document": pdf_filename,
                        "page_number": current_page,
                        "image_path": page_image_save_path,
                    },
                )
            )
        except Exception as e:
            print(f"  - Error running Vision model on page {current_page} of {pdf_filename}: {e}")
            # Continue processing even if vision model fails for one page

    doc.close()
    os.remove(temp_pdf_path) # Clean up the temporary file
    return all_docs


def create_rag_chain(vector_store):
    retriever = vector_store.as_retriever()
    llm = ChatOllama(model="llama3:8b-instruct-q4_K_M", temperature=0)
    template = """
You are a meticulous AI assistant. Your task is to answer the user's question by synthesizing information from the context provided below.

The context consists of two types of information from a document:
1.  `ocr_text`: Text directly extracted from the page via Optical Character Recognition.
2.  `visual_description`: A summary of a visual element like a chart, image, or diagram on the same page.

CRITICAL RULES:
- You MUST combine information from both `ocr_text` and `visual_description` if they are relevant to the user's question. A complete answer often requires understanding how the text refers to the visuals.
- Base your answer STRICTLY and ONLY on the provided context.
- Do not invent any information, figures, or details.
- If the answer cannot be found in the combination of the provided sources, you MUST state that you do not have enough information to answer.
- For each piece of information you use, cite the source document and page number.

Context:
{context}

Question:
{question}

Answer:
"""
    prompt = ChatPromptTemplate.from_template(template=template)

    def format_docs(docs):
        return "\n\n".join(
            f"Source: {d.metadata.get('source_document', 'N/A')}, "
            f"Page: {d.metadata.get('page_number', 'N/A')}\n"
            f"Content: {d.page_content}"
            for d in docs
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# --- FastAPI Application Setup ---

app = FastAPI(
    title="Multimodal PDF RAG API",
    description="API for ingesting multimodal PDFs (OCR + Vision) and querying with RAG.",
    version="1.0.0",
)

# Global variables for the RAG system
# Initialize vector_store as None; it will be initialized on startup or first upload
vector_store: Union[Chroma, None] = None
rag_chain = None
embedding_model = None
CHROMA_PERSIST_DIR = "./chroma_db_api" # Use a different directory for the API's vector store

origins = [
    "http://localhost",
    "http://localhost:3000", # Your React app's default address
    "http://127.0.0.1:3000", # Sometimes it resolves to this
    # Add any other origins where your frontend might be hosted, e.g.,
    # "http://your-production-domain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.on_event("startup")
async def startup_event():
    """
    Initializes the embedding model and attempts to load the vector store
    from disk on application startup.
    """
    global embedding_model, vector_store, rag_chain
    print("Initializing embedding model...")
    embedding_model = OllamaEmbeddings(model="all-minilm:l6-v2")

    print(f"Attempting to load vector store from {CHROMA_PERSIST_DIR}...")
    try:
        # Check if the persist directory exists and has data
        if os.path.exists(CHROMA_PERSIST_DIR) and any(os.scandir(CHROMA_PERSIST_DIR)):
            vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_model
            )
            print(f"Loaded vector store with {vector_store._collection.count()} entries.")
            if vector_store._collection.count() > 0:
                rag_chain = create_rag_chain(vector_store=vector_store)
                print("RAG chain initialized from persistent store.")
            else:
                print("Vector store loaded but is empty. Will initialize RAG chain upon first upload.")
        else:
            print("No existing vector store found or it's empty. It will be created on first upload.")
            # Chroma will create the directory if it doesn't exist when we add documents
            # We don't initialize an empty Chroma DB here directly, it's done when `from_documents` or `add_documents` is called
            # and the first set of documents is added.

    except Exception as e:
        print(f"Error loading existing vector store: {e}")
        vector_store = None # Ensure it's None if loading fails


class QueryRequest(BaseModel):
    question: str


@app.post("/upload", summary="Upload PDF files for RAG ingestion", response_model=Dict[str, str])
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Uploads one or more PDF files. Each file will be processed using OCR and a Vision Model,
    and their contents (text and visual descriptions) will be added to the RAG system's
    vector store.
    """
    global vector_store, rag_chain, embedding_model

    if embedding_model is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Embedding model not initialized. Please restart the application.",
        )

    processed_files_info = {}
    all_new_docs: List[Document] = []

    for file in files:
        if file.content_type != "application/pdf":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{file.filename}' is not a PDF.",
            )

        try:
            pdf_bytes = await file.read()
            docs_from_pdf = process_multimodal_pdf(pdf_bytes, file.filename)
            all_new_docs.extend(docs_from_pdf)
            processed_files_info[file.filename] = (
                f"Processed {len(docs_from_pdf)} documents."
            )
            print(f"Successfully processed {file.filename}.")
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            processed_files_info[file.filename] = f"Failed to process: {e}"
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process {file.filename}: {e}",
            )

    if not all_new_docs:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents were successfully extracted from the uploaded PDFs.",
        )

    print(f"Adding {len(all_new_docs)} new documents to the vector store...")
    try:
        if vector_store is None:
            # First time adding documents, create the Chroma instance
            vector_store = Chroma.from_documents(
                documents=all_new_docs,
                embedding=embedding_model,
                collection_name="rag", # Use a consistent collection name
                persist_directory=CHROMA_PERSIST_DIR,
            )
            print(f"Initialized new vector store with {vector_store._collection.count()} entries.")
        else:
            # If vector_store already exists, add documents to it
            vector_store.add_documents(documents=all_new_docs)
            vector_store.persist() # Ensure changes are saved
            print(f"Added documents to existing vector store. Total entries: {vector_store._collection.count()}")

        # Always re-create the RAG chain if documents are added or the store is new
        rag_chain = create_rag_chain(vector_store=vector_store)
        print("RAG chain updated with new documents.")

    except Exception as e:
        print(f"Error adding documents to vector store: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add documents to vector store: {e}",
        )

    return JSONResponse(
        content={
            "message": "Files processed and added to vector store.",
            "details": processed_files_info,
            "total_documents_in_store": vector_store._collection.count() if vector_store else 0,
        },
        status_code=status.HTTP_200_OK,
    )


@app.post("/query", summary="Query the RAG system", response_model=Dict[str, str])
async def query_rag_model(request: QueryRequest):
    """
    Queries the RAG system with a given question and returns the answer.
    """
    global rag_chain, vector_store

    if vector_store is None or vector_store._collection.count() == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No documents have been uploaded yet. Please upload PDFs first.",
        )

    if rag_chain is None:
         # This case should ideally not happen if startup and upload work correctly,
         # but as a safeguard, recreate the chain if it's somehow missing.
        rag_chain = create_rag_chain(vector_store=vector_store)
        print("RAG chain re-initialized during query request.")


    print(f"Received query: '{request.question}'")
    try:
        answer = await rag_chain.ainvoke(request.question)
        print(f"Generated answer: {answer[:100]}...") # Print first 100 chars
        return JSONResponse(
            content={"question": request.question, "answer": answer},
            status_code=status.HTTP_200_OK,
        )
    except Exception as e:
        print(f"Error during RAG chain invocation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your query: {e}",
        )
