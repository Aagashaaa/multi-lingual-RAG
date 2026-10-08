from pathlib import Path

from .document import Document


def load_txt(file_path: str) -> list[Document]:
    path = Path(file_path)

    text = path.read_text(encoding="utf-8")

    if not text.strip():
        return []

    return [
        Document(
            text=text.strip(),
            source=path.name,
            file_type="txt",
        )
    ]