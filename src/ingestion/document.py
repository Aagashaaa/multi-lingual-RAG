from dataclasses import dataclass
from typing import Optional

@dataclass
class Document:
    text: str
    source: str
    file_type: str
    page_number: Optional[int] = None
    language: Optional[str] = None