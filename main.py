import base64
import os
import io
import fitz
import pytesseract
from PIL import Image
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

def process_multimodal_pdf(pdf_path: str) -> List[Document]:
    """
    Processes a PDF where each page is an image. It uses OCR for text
    and a VLM for visual descriptions.
    """ 

    pdf_filename = os.path.basename(pdf_path)
    doc = fitz.open(pdf_path)
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

        page_image_path = os.path.join("data/", f"{pdf_filename}_p{current_page}.png")
        page_image.save(page_image_path)

        # a) OCR Path: Extract text from the page image
        print(f"  - Running OCR on page {current_page}...")
        ocr_text = pytesseract.image_to_string(page_image)
        if ocr_text:
            text_chunks = text_splitter.split_text(ocr_text)
            for i, chunk in enumerate(text_chunks):
                all_docs.append(
                    Document(
                        page_content=chunk,
                        metadata = {
                            "source_type": "ocr_text",
                            "source_document": pdf_filename,
                            "page_number": current_page,
                        }
                    )
                )
        
        # b) Vision Path: Get a visual description of the page image
        print(f"  - Running Vision model on page {current_page}...")
        image_bytes = pix.tobytes("png")
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        human_message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image in detail for a search index. (Scan each and every part of the page)"},
                {"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"}
            ]
        )
        response = image_caption_model.invoke([human_message])
        visual_descriptor = response.content

        all_docs.append(
            Document(
                page_content=visual_descriptor,
                metadata = {
                    "source_type": "visual_description",
                    "source_document": pdf_filename,
                    "page_number": current_page,
                    "image_path": page_image_path,
                }
            )
        )
    doc.close()
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
        { "context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
            

if __name__ == "__main__":
    
    all_docs = process_multimodal_pdf("data/aluminum_hinge.pdf")
    print(f"\nGenerated a total of {len(all_docs)} documents from OCR and Vision models.")

    embedding_model = OllamaEmbeddings(model="all-minilm:l6-v2")
    print("\nCreating or loading vector store...")
    vector_store = Chroma.from_documents(
        documents=all_docs,
        embedding=embedding_model,
        collection_name="rag",
        persist_directory="./chroma.db",
    )
    print(f"Vector store contains {vector_store._collection.count()} entries.")

    rag_chain = create_rag_chain(vector_store=vector_store)
    print("\n--- Querying RAG System ---")
    query = "describe the hinge base."
    print(f"\n❓ Query: {query}")
    answer = rag_chain.invoke(query)
    print(f"\n✅ Answer:\n{answer}")