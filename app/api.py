import io
from typing import Optional

from fastapi import FastAPI, UploadFile, HTTPException, File
from pydantic import BaseModel

from app.logger import get_logger
from app.config import TOP_K
from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_text
from app.vectorstore.faiss_store import FaissVectorStore
from app.ingestion.embedder import embed_chunks
from app.retrieval.retriever import retrieve_context
from app.retrieval.prompt import build_prompt
from app.llm.model import generate_answer, tokenizer

logger = get_logger()

app = FastAPI(
    title="Ask the Docs API",
    version="1.0.0",
    description="RAG backend for document Q&A"
)

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = TOP_K

class QueryResponse(BaseModel):
    answer: str

# maintain one vector store per app instance
vector_store: Optional[FaissVectorStore] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    global vector_store

    try:
        contents = await file.read()
        file_size = len(contents)

        if file_size == 0:
            raise ValueError("Uploaded file is empty")
        
        # document ingestion
        text = load_document(
            file_name = file.filename,
            file = io.BytesIO(contents),
            file_size_bytes=file_size
        )

        # chunking
        chunks = chunk_text(text)

        # embedding
        embeddings, metadata = embed_chunks(chunks)

        # vector store
        if vector_store is None:
            vector_store = FaissVectorStore(embeddings.shape[1])

        vector_store.add(embeddings, metadata)
        vector_store.save()

        logger.info(
            f"Document ingested successfully | chunks={len(chunks)}"
        )

        return {
            "message" : "Document ingested successfully",
            "chunks_indexed": len(chunks)
        }

    except Exception as e:
        logger.exception("Document ingestion failed")
        raise HTTPException(status_code=400, detail=str(e))

    
@app.post("/query", response_model=QueryResponse)
def query_document(request: QueryRequest):
    global vector_store

    if vector_store is None:
        try:
            vector_store = FaissVectorStore.load()
            logger.info("FAISS index loaded from disk")
        except Exception:
            raise HTTPException(
                status_code = 400,
                detail = "No document indexed yet. Please ingest as document first."
            )
        
    try:
        # retrieval
        retrieved_chunks = retrieve_context(
            query=request.question,
            store=vector_store,
            top_k=request.top_k
        )

        # prompt construction
        prompt = build_prompt(
            question=request.question,
            retrieved_chunks=retrieved_chunks,
            tokenizer=tokenizer
        )

        answer = generate_answer(prompt)

        return QueryResponse(answer=answer)
    except Exception as e:
        logger.exception("Query processing failed")
        raise HTTPException(status_code=400, detail=str(e))

