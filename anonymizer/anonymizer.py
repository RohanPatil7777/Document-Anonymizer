import re
from typing import Dict, Any, List, Tuple
from transformers import pipeline
from .utils import chunk_text, generate_placeholder, clean_pdf_text

class DocumentAnonymizer:
    def __init__(self, model_name: str = "dslim/bert-base-NER", threshold: float = 0.5):
        self.ner_pipeline = pipeline("ner", model=model_name, aggregation_strategy="simple")
        self.threshold = threshold
        self.regex_patterns = {
            "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            "PHONE": r'\b(?:\+?\d[\d -]{8,}\d)\b',
            "URL": r'https?://\S+|www\.\S+'
        }
        self.global_map: Dict[str, str] = {}      # placeholder -> original text
        self.reverse_map: Dict[str, str] = {}     # original -> placeholder
        self.counters: Dict[str, int] = {}

    # -----------------------------
    # 1. Text Preprocessing
    # -----------------------------
    def preprocess_text(self, text: str) -> str:
        text = clean_pdf_text(text)

        # Merge fragmented emails
        text = re.sub(r'([A-Za-z0-9._%+-])\s*@\s*([A-Za-z0-9.-]+)', r'\1@\2', text)
        # Merge fragmented URLs
        text = re.sub(r'www\.\s*([A-Za-z0-9.-]+)', r'www.\1', text)
        # Merge accidental split characters: "R obert" -> "Robert"
        text = re.sub(r'\b([A-Z])\s+([a-z])', r'\1\2', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    # -----------------------------
    # 2. Detect Entities (NER)
    # -----------------------------
    def detect_entities(self, text: str) -> List[Tuple[int, int, str, str]]:
        """
        Returns entity spans as (start, end, label, original_text)
        """
        chunks = chunk_text(text)
        entity_spans = []

        for chunk, offset in chunks:
            entities = self.ner_pipeline(chunk)
            for ent in entities:
                label = ent['entity_group'].upper()
                if label in ["EMAIL", "PHONE", "URL"]:
                    continue  # handled by regex
                if ent['score'] < self.threshold:
                    continue

                start = ent['start'] + offset
                end = ent['end'] + offset
                original = text[start:end].strip()

                if len(original) < 2:
                    continue

                # Merge contiguous entities of the same type
                if entity_spans and start <= entity_spans[-1][1] + 1 and entity_spans[-1][2] == label:
                    prev_start, prev_end, prev_label, prev_text = entity_spans[-1]
                    entity_spans[-1] = (prev_start, end, label, f"{prev_text} {original}".strip())
                else:
                    entity_spans.append((start, end, label, original))

        return entity_spans

    # -----------------------------
    # 3. Main Anonymization
    # -----------------------------
    def anonymize(self, text: str) -> Dict[str, Any]:
        cleaned_text = self.preprocess_text(text)
        entity_spans: List[Tuple[int, int, str, str]] = []

        # Step 1: Regex-based detection for EMAIL, PHONE, URL
        for label, pattern in self.regex_patterns.items():
            for match in re.finditer(pattern, cleaned_text):
                start, end = match.span()
                original = match.group(0).strip()

                if len(original) < 3:
                    continue

                placeholder = self.reverse_map.get(original)
                if not placeholder:
                    placeholder = generate_placeholder(label, self.counters)
                    self.global_map[placeholder] = original
                    self.reverse_map[original] = placeholder

                entity_spans.append((start, end, placeholder, original))

        # Step 2: NER-based detection for PER, ORG, LOC
        for start, end, label, original in self.detect_entities(cleaned_text):
            if len(original) < 3:
                continue

            # Optional heuristic: detect wrong classification
            if label == "ORG" and re.match(r'^[A-Z][a-z]+\s[A-Z][a-z]+$', original):
                label = "PER"
            if label == "PER" and re.search(r'(Inc|Corporation|Ltd|LLC|Company)$', original):
                label = "ORG"

            placeholder = self.reverse_map.get(original)
            if not placeholder:
                placeholder = generate_placeholder(label, self.counters)
                self.global_map[placeholder] = original
                self.reverse_map[original] = placeholder

            entity_spans.append((start, end, placeholder, original))

        # Step 3: Sort by start position (descending) to safely replace
        entity_spans.sort(key=lambda x: x[0], reverse=True)

        anonymized_text = cleaned_text
        for start, end, placeholder, _ in entity_spans:
            anonymized_text = anonymized_text[:start] + placeholder + anonymized_text[end:]

        # Step 4: Prepare statistics
        statistics = {
            "total_entities": sum(self.counters.values()),
            "by_category": self.counters
        }

        return {
            "anonymized_text": anonymized_text,
            "statistics": statistics,
            "entity_mapping": self.global_map
        }
