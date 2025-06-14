import os
from typing import List, Optional
from pypdf import PdfReader
from docx import Document
from app.config import DOCUMENTS_DIR


class DocumentProcessor:
    @staticmethod
    def save_document(file_content: bytes, filename: str) -> str:
        """Save uploaded document to storage."""
        file_path = os.path.join(DOCUMENTS_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(file_content)
        return file_path

    @staticmethod
    def extract_text(file_path: str) -> List[str]:
        """Extract text from different file types."""
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == ".pdf":
            return DocumentProcessor._extract_from_pdf(file_path)
        elif file_extension == ".docx":
            return DocumentProcessor._extract_from_docx(file_path)
        elif file_extension == ".txt":
            return DocumentProcessor._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    @staticmethod
    def _extract_from_pdf(file_path: str) -> List[str]:
        """Extract text from PDF file."""
        reader = PdfReader(file_path)
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
        return texts

    @staticmethod
    def _extract_from_docx(file_path: str) -> List[str]:
        """Extract text from DOCX file."""
        doc = Document(file_path)
        texts = []
        for paragraph in doc.paragraphs:
            if paragraph.text:
                texts.append(paragraph.text)
        return texts

    @staticmethod
    def _extract_from_txt(file_path: str) -> List[str]:
        """Extract text from TXT file."""
        try:
            # Try UTF-8 first
            with open(file_path, "r", encoding="utf-8") as f:
                return [f.read()]
        except UnicodeDecodeError:
            # If UTF-8 fails, try with latin-1 which can handle any byte sequence
            with open(file_path, "r", encoding="latin-1") as f:
                return [f.read()]
