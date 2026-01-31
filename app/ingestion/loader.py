import os
from typing import BinaryIO
import pdfplumber

from app.config import STORAGE_DIR
from app.logger import get_logger

from pdf2image import convert_from_bytes
import pytesseract
from app.config import MIN_TEXT_LENGTH

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

def extract_text_from_pdf(file_path: str) -> str:
    extracted_pages = []

    with pdfplumber.open(file_path) as pdf:
        logger.info(f"PDF opened with {len(pdf.pages)} pages")

        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                extracted_pages.append(page_text)

    text = "\n".join(extracted_pages).strip()
    return text


def ocr_pdf(file_bytes: bytes) -> str:
    images = convert_from_bytes(file_bytes, dpi=300)
    ocr_text = []

    for img in images:
        img = img.convert("L")  # grayscale
        text = pytesseract.image_to_string(
            img,
            lang="eng",
            config="--oem 3 --psm 6"
        )
        if text:
            ocr_text.append(text)

    return "\n".join(ocr_text).strip()


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

        file_bytes = file.read()

        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        # PDF text extraction
        text = extract_text_from_pdf(temp_path)

        # Decide whether OCR is needed
        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
            logger.warning(
                "PDF text extraction insufficient, falling back to OCR"
            )

            ocr_text = ocr_pdf(file_bytes)

            if not ocr_text:
                os.remove(temp_path)
                raise ValueError("OCR failed to extract text from PDF")

            text = ocr_text
            logger.info(
                f"OCR successful | extracted_length={len(text)}"
            )
        else:
            logger.info(
                f"PDF text extracted without OCR | length={len(text)}"
            )

        os.remove(temp_path)

    else:
        raise ValueError("Unsupported file format")
    
    logger.info(
        f"Ingestion successful for file={file_name},"
        f"text_length={len(text)} charcters"
    )

    return text