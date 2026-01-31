# Ask the Docs 📄🤖

**Ask the Docs** is a production-ready Retrieval-Augmented Generation (RAG) application that allows users to upload documents and ask natural language questions grounded strictly in the document’s content.

The system supports both **digital and scanned PDFs**, performs **OCR when required**, retrieves the most relevant document sections using **vector search**, and generates accurate answers using a **language model**.

---

## 🚀 Problem Statement

Build a small RAG app titled **“Ask the Docs”** that:

- Allows users to upload `.pdf` or `.txt` files
- Performs OCR for scanned PDFs
- Accepts natural language questions
- Retrieves relevant document sections
- Generates accurate, context-grounded answers
- Is Dockerized and deployed on AWS (ECS preferred)
- Is publicly accessible via a URL

---

## 🧠 High-Level Architecture

User (Browser)
↓
Streamlit UI (Frontend)
↓ HTTP (localhost)
FastAPI Backend
↓
Document Ingestion
├─ PDF/Text Loader
├─ OCR (Tesseract fallback)
├─ Text Chunking
├─ Embeddings (all-MiniLM-L6-v2)
├─ FAISS Vector Store
└─ Retrieval
↓
Intent-Aware Prompt Construction
↓
LLM Inference
↓
Answer


Both **FastAPI** (backend) and **Streamlit** (frontend) run inside a **single Docker container**.

---

## ✨ Key Features

### 📄 Document Upload
- Supports `.pdf` and `.txt` files
- File size validation and safe handling

### 🔍 OCR Support
- Uses native PDF text extraction when available
- Automatically falls back to **Tesseract OCR** for scanned PDFs

### 📚 Retrieval-Augmented Generation (RAG)
- Document text is chunked and embedded
- FAISS is used for fast vector similarity search
- Only relevant chunks are passed to the LLM

### 🧠 Intent-Aware Prompting
The system adapts prompts based on question type:
- **Factual Q&A** (strict grounding)
- **Definitions / explanations** (controlled rephrasing)
- **Summarization** (context-only synthesis)

This prevents hallucinations while allowing meaningful answers.

### 🔐 Grounded Responses
- Answers are generated **only from retrieved context**
- If the answer is not supported, the system replies:  
  **“I don’t know.”**

---

## 🧰 Tech Stack

| Component | Technology |
|--------|-----------|
| UI | Streamlit |
| Backend API | FastAPI |
| OCR | Tesseract |
| PDF Processing | pdfplumber, pdf2image |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS (CPU) |
| LLM | google/flan-t5-large / Qwen2.5 |
| Containerization | Docker |
| Cloud Deployment | AWS ECS (Fargate) |

---


---

## 🐳 Running Locally with Docker

```bash
docker build -t ask-the-docs .

docker run \
  -e HF_TOKEN=hf_xxxxxxxxx \
  -p 8501:8501 \
  -p 8000:8000 \
  ask-the-docs

Streamlit UI: http://localhost:8501

FastAPI Docs: http://localhost:8000/docs