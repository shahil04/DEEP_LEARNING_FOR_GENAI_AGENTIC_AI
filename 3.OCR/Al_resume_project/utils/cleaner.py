import re

def clean_resume_text(text: str) -> str:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)         # Separate camelCase words
    text = re.sub(r"[^\x00-\x7F]+", " ", text)               # Remove non-ASCII
    text = re.sub(r"\.(\w)", r". \1", text)                  # Add space after periods
    text = re.sub(r"[^a-zA-Z0-9.,:;@()\-\s]", " ", text)     # Remove unwanted characters
    text = re.sub(r"\s+", " ", text)                         # Collapse multiple spaces
    return text.strip()
