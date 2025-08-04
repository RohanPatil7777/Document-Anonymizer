# Document Anonymizer

This project anonymizes sensitive information (PII) from long documents using Hugging Face transformers and regex.

## Features
- Transformer-based NER (dslim/bert-base-NER)
- Regex fallback for EMAIL, PHONE, URL
- Chunking for long documents
- Generates anonymized text, statistics, and entity mapping
- CLI tool and unit tests

## Usage
```bash
pip install -r requirements.txt
python main.py --input (NAME OF THE PDF).pdf --output anonymized_output.json
```

