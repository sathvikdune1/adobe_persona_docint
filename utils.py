import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import os
import datetime

def load_model(model_path=None):
    # Local model, if provided
    return SentenceTransformer(model_path or "sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    text_by_page = []
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text_by_page.append(text)
    return text_by_page

def simple_section_split(page_text):
    # Simple heuristic: Split by lines with ALL CAPS or numbered sections (can be improved for domain)
    lines = page_text.split('\n')
    sections = []
    sec_buffer = []
    sec_title = None
    for line in lines:
        if re.match(r"^(?P<num>\d+\.*\d*)?\s*[A-Z][A-Z\s\-,]+$", line.strip()):
            if sec_buffer:
                sections.append((sec_title, "\n".join(sec_buffer)))
                sec_buffer = []
            sec_title = line.strip()
        elif line.strip():
            sec_buffer.append(line.strip())
    if sec_buffer:
        sections.append((sec_title if sec_title else "Section", "\n".join(sec_buffer)))
    return sections if sections else [("Section", page_text)]

def get_doc_sections(pdf_path):
    pages = extract_text_from_pdf(pdf_path)
    doc_sec = []
    for pg_num, pg_text in enumerate(pages):
        page_sections = simple_section_split(pg_text)
        for title, sec_text in page_sections:
            doc_sec.append({
                "page": pg_num + 1,
                "title": title,
                "text": sec_text
            })
    return doc_sec

def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk: chunks.append(chunk)
    return chunks

def cosine_sim(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)+1e-8)/(np.linalg.norm(b)+1e-8)

def rank_sections(sections, query_embedding, model):
    sec_embeddings = model.encode([sec["text"] for sec in sections], show_progress_bar=False)
    ranked = []
    for i, sec in enumerate(sections):
        score = cosine_sim(sec_embeddings[i], query_embedding)
        ranked.append((score, sec))
    ranked.sort(reverse=True, key=lambda x: x[0])
    return ranked

def analyze_subsections(section, query_embedding, model):
    chunks = chunk_text(section["text"])
    if not chunks: return []
    chunk_embeds = model.encode(chunks, show_progress_bar=False)
    results = []
    for i, chunk in enumerate(chunks):
        score = cosine_sim(chunk_embeds[i], query_embedding)
        results.append({
            "Refined Text": chunk,
            "Page Number": section["page"],
            "Score": score
        })
    # Rank descending and keep the topmost N (say, 3)
    results = sorted(results, key=lambda x: x['Score'], reverse=True)[:3]
    return results
