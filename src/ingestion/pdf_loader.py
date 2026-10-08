from pathlib import Path
from pypdf import PdfReader
from .document import Document

def load_pdf(file_path: str) -> list[Document]:
    path = Path(file_path)

    reader = PdfReader(path)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            documents.append(
                Document(
                    text=text.strip(),
                    source=path.name,
                    file_type="pdf",
                    page_number=page_number,
                )
            )

    return documents