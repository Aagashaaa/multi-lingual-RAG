from pathlib import Path

from .document import Document
from .docx_loader import load_docx
from .pdf_loader import load_pdf
from .txt_loader import load_txt


def load_document(file_path: str) -> list[Document]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    if extension == ".txt":
        return load_txt(file_path)

    if extension == ".docx":
        return load_docx(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}. "
        "Supported types: .pdf, .txt, .docx"
    )