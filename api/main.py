from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import uuid
from pathlib import Path
import asyncio
import logging

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF Chat API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
UPLOAD_DIR = Path("uploads")
VECTOR_STORE_DIR = Path("vector_stores")
UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    session_id: str

class ProcessResponse(BaseModel):
    session_id: str
    status: str
    message: str

class QuestionResponse(BaseModel):
    answer: str
    session_id: str

class HealthResponse(BaseModel):
    status: str
    ollama_available: bool

# Global storage for processing status
processing_status = {}

class PDFProcessor:
    def __init__(self):
        self.embeddings = None
        self.llm = None
        self._init_models()
    
    def _init_models(self):
        try:
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url="http://localhost:11434"
            )
            self.llm = Ollama(
                model="llama2",
                base_url="http://localhost:11434",
                temperature=0.3
            )
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            self.embeddings = None
            self.llm = None
    
    def extract_text_from_pdfs(self, pdf_paths: List[str]) -> str:
        text = ""
        for pdf_path in pdf_paths:
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
            except Exception as e:
                logger.error(f"Error reading PDF {pdf_path}: {e}")
        return text
    
    def create_chunks(self, text: str) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks
    
    def create_vector_store(self, chunks: List[str], session_id: str) -> bool:
        if not self.embeddings:
            return False
        
        try:
            vector_store = FAISS.from_texts(chunks, embedding=self.embeddings)
            store_path = VECTOR_STORE_DIR / session_id
            vector_store.save_local(str(store_path))
            return True
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return False
    
    def get_answer(self, question: str, session_id: str) -> Optional[str]:
        if not self.embeddings or not self.llm:
            return None
        
        try:
            store_path = VECTOR_STORE_DIR / session_id
            if not store_path.exists():
                return "Please upload and process PDF files first."
            
            vector_store = FAISS.load_local(
                str(store_path), 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            docs = vector_store.similarity_search(question, k=3)
            
            prompt_template = """
            Use the following pieces of context to answer the question at the end. 
            If you don't know the answer based on the context provided, just say that you don't know.
            
            Context: {context}
            Question: {question}
            Answer:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
            response = chain(
                {"input_documents": docs, "question": question},
                return_only_outputs=True
            )
            
            return response["output_text"]
            
        except Exception as e:
            logger.error(f"Error getting answer: {e}")
            return f"Error processing question: {str(e)}"

# Initialize processor
pdf_processor = PDFProcessor()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF Chat API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PDF Chat API</h1>
            <p>Welcome to the PDF Chat API. Use the following endpoints:</p>
            
            <div class="endpoint">
                <h3>GET /health</h3>
                <p>Check API and Ollama status</p>
            </div>
            
            <div class="endpoint">
                <h3>POST /upload</h3>
                <p>Upload PDF files for processing</p>
            </div>
            
            <div class="endpoint">
                <h3>POST /ask</h3>
                <p>Ask questions about uploaded PDFs</p>
            </div>
            
            <div class="endpoint">
                <h3>GET /docs</h3>
                <p>View API documentation</p>
            </div>
            
            <p><a href="/docs">📚 View Full API Documentation</a></p>
        </div>
    </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    ollama_available = pdf_processor.embeddings is not None and pdf_processor.llm is not None
    return HealthResponse(
        status="healthy" if ollama_available else "degraded",
        ollama_available=ollama_available
    )

async def process_pdfs_background(pdf_paths: List[str], session_id: str):
    processing_status[session_id] = "processing"
    
    try:
        # Extract text
        text = pdf_processor.extract_text_from_pdfs(pdf_paths)
        if not text.strip():
            processing_status[session_id] = "failed - no text found"
            return
        
        # Create chunks
        chunks = pdf_processor.create_chunks(text)
        
        # Create vector store
        if pdf_processor.create_vector_store(chunks, session_id):
            processing_status[session_id] = "completed"
        else:
            processing_status[session_id] = "failed - vector store creation"
            
    except Exception as e:
        logger.error(f"Background processing error: {e}")
        processing_status[session_id] = f"failed - {str(e)}"
    finally:
        # Clean up uploaded files
        for pdf_path in pdf_paths:
            try:
                os.remove(pdf_path)
            except:
                pass

@app.post("/upload", response_model=ProcessResponse)
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    session_id = str(uuid.uuid4())
    pdf_paths = []
    
    # Save uploaded files
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        file_path = UPLOAD_DIR / f"{session_id}_{file.filename}"
        
        try:
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            pdf_paths.append(str(file_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    # Start background processing
    background_tasks.add_task(process_pdfs_background, pdf_paths, session_id)
    processing_status[session_id] = "started"
    
    return ProcessResponse(
        session_id=session_id,
        status="started",
        message=f"Processing {len(files)} PDF files. Use session_id to check status and ask questions."
    )

@app.get("/status/{session_id}")
async def get_status(session_id: str):
    status = processing_status.get(session_id, "not_found")
    return {"session_id": session_id, "status": status}

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    if request.session_id not in processing_status:
        raise HTTPException(status_code=404, detail="Session not found")
    
    status = processing_status[request.session_id]
    if status != "completed":
        raise HTTPException(status_code=400, detail=f"PDFs not ready. Current status: {status}")
    
    answer = pdf_processor.get_answer(request.question, request.session_id)
    if answer is None:
        raise HTTPException(status_code=500, detail="Error generating answer. Check Ollama setup.")
    
    return QuestionResponse(answer=answer, session_id=request.session_id)

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    # Clean up vector store
    store_path = VECTOR_STORE_DIR / session_id
    if store_path.exists():
        shutil.rmtree(store_path)
    
    # Remove from processing status
    processing_status.pop(session_id, None)
    
    return {"message": "Session deleted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)