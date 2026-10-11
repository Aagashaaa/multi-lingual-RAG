
import re
import unicodedata


def process_query(query: str) -> str:
    """
    Normalize and validate a user query without
    changing its language or intended meaning.
    """
    if not isinstance(query, str):
        raise TypeError("Query must be a string")

    # Normalize Unicode representations.
    query = unicodedata.normalize("NFKC", query)

    # Normalize line endings.
    query = query.replace("\r\n", "\n")
    query = query.replace("\r", "\n")

    # Collapse repeated spaces, tabs, and newlines.
    query = re.sub(r"\s+", " ", query)

    # Remove whitespace at the beginning and end.
    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty")

    return query
