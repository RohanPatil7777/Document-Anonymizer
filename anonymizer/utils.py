import re
from typing import List, Tuple

def clean_pdf_text(text: str) -> str:
    """
    Cleans extracted PDF text to improve NER performance.
    - Removes excessive newlines
    - Fixes hyphenated line breaks
    - Removes duplicate spaces
    - Joins split words like 'John Sm\nith' -> 'John Smith'
    """
    # Remove hyphen line breaks first
    text = re.sub(r'-\s*\n\s*', '', text)

    # Join lines that are broken mid-word or mid-sentence
    text = re.sub(r'\n(?=[a-z])', '', text)       # lowercase continuation
    text = re.sub(r'\n(?=[A-Z])', ' ', text)      # new sentence/word
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing spaces
    text = text.strip()
    return text

def chunk_text(text: str, max_words: int = 200) -> List[Tuple[str, int]]:
    """
    Splits text into chunks for NER processing.
    Uses sentence-based splitting for cleaner entity detection.
    Returns list of (chunk_text, offset_in_original_text).
    """
    # Split into sentences using punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_offset = 0
    start_idx = 0

    for sentence in sentences:
        words = sentence.split()
        if len(current_chunk) + len(words) > max_words:
            chunk_text_val = " ".join(current_chunk)
            chunks.append((chunk_text_val, start_idx))
            start_idx += len(chunk_text_val) + 1
            current_chunk = []
        current_chunk.extend(words)
    
    if current_chunk:
        chunk_text_val = " ".join(current_chunk)
        chunks.append((chunk_text_val, start_idx))
    
    return chunks

def generate_placeholder(label: str, counters: dict) -> str:
    """Generates sequential placeholders like [PER_1], [EMAIL_2]."""
    counters[label] = counters.get(label, 0) + 1
    return f"[{label}_{counters[label]}]"
