import os
from typing import BinaryIO
import pdfplumber

from app.config import STORAGE_DIR
from app.logger import get_logger

logger = get_logger()

SUPPORTED_FILES = {".pdf", ".txt"}
MAX_FILE_SIZE_MB = 10


def validate_file(filename: str, file_size_bytes: int) -> None:
    extension = os.path.splitext(filename)[1].lower()

    if extension not in SUPPORTED_FILES:
        raise ValueError(f"Unsupported file type: {extension}. Please upload .pdf or .txt file!")
    
    max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_size_bytes:
        raise ValueError("File size exceeds allowed limit")
    
    if file_size_bytes == 0:
        raise ValueError("Empty file uploaded")
    

def load_text_file(file: BinaryIO) -> str:
    try:
        raw_text = file.read().decode("utf-8")
    except Exception as e:
        logger.error("text file decode failed")
        raise ValueError("Unable to decode text file as UTF-8") from e
    
    text = raw_text.replace("\x00", "").strip()

    if not text:
        raise ValueError("Text file contains no readable text")
    
    return text

def load_pdf_file(file_path: str) -> str:
    extracted_pages = []

    with pdfplumber.open(file_path) as pdf:
        logger.info(f"PDF opened with {len(pdf.pages)} pages")

        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                extracted_pages.append(page_text)
    
    if not extracted_pages:
        raise ValueError("PDF contains no extractable text (OCR not supported)")
    
    text = "\n".join(extracted_pages).strip()

    if not text:
        raise ValueError("PDF text extraction failed")
    
    return text

def load_document(
        file_name: str,
        file: BinaryIO,
        file_size_bytes: int
) -> str:
    """
    Validates and loads a document, returning extracted clean text.
    """
    logger.info(
        f"Starting ingestion for file={file_name},"
        f"size={file_size_bytes} bytes"
    )

    validate_file(file_name, file_size_bytes)

    extension = os.path.splitext(file_name)[1].lower()

    if extension == ".txt":
        text = load_text_file(file)
    
    elif extension == ".pdf":
        os.makedirs(STORAGE_DIR, exist_ok=True)
        temp_path = os.path.join(STORAGE_DIR, file_name)
        
        with open(temp_path, "wb") as f:
            f.write(file.read())

        text = load_pdf_file(temp_path)

        os.remove(temp_path)

    else:
        raise ValueError("Unsupported file format")
    
    logger.info(
        f"Ingestion successful for file={file_name},"
        f"text_length={len(text)} charcters"
    )

    return text