
import argparse
import json
import pdfplumber
from anonymizer.anonymizer import DocumentAnonymizer
from anonymizer.utils import clean_pdf_text

def read_pdf(file_path: str) -> str:
    """Extracts text from a PDF file and cleans it for better NER performance."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    # Clean text immediately after extraction
    return clean_pdf_text(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Anonymizer CLI (Improved PDF handling)")
    parser.add_argument("--input", required=True, help="Path to PDF or TXT file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    # Step 1: Read and clean the input file
    if args.input.lower().endswith(".pdf"):
        text = read_pdf(args.input)
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            text = clean_pdf_text(f.read())

    # Step 2: Initialize and run the anonymizer
    anonymizer = DocumentAnonymizer()
    result = anonymizer.anonymize(text)

    # Step 3: Save the output as JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"✅ Anonymization complete! Output saved to {args.output}")
    print(f"Entities found: {result['statistics']['total_entities']}")
