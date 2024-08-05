import fitz
import re
from pathlib import Path

def extract_text_from_pdf(pdf_path : Path) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def clean(text):
    # Compile patterns for URLs and emails to speed up cleaning process
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

    # Remove URLs
    clean_text = url_pattern.sub('', text)

    # Remove emails
    clean_text = email_pattern.sub('', clean_text)

    # Remove special characters (keeping only words and whitespace)
    clean_text = re.sub(r'[^\w\s]', '', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', '', clean_text)  # https://stackoverflow.com/questions/20889996/how-do-i-remove-all-non-ascii-characters-with-regex-and-notepad no ascii
    clean_text = re.sub(r'\b\d{3,}\b', '', clean_text)
    return clean_text

def prepare(pdf_path : Path) -> str:
    text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean(text)
    return cleaned_text
