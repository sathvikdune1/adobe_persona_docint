import os
import json
from datetime import datetime
from utils import get_doc_sections, load_model, rank_sections, analyze_subsections

DOCUMENTS_FOLDER = "./sample_input/"
DOCUMENTS = [os.path.join(DOCUMENTS_FOLDER, f) for f in sorted(os.listdir(DOCUMENTS_FOLDER)) if f.endswith(".pdf")]

# --- For submission, set these three per test case ---
PERSONA = "Investment Analyst"
JOB_TO_BE_DONE = "Analyze revenue trends, R&D investments, and market positioning strategies"

# -----------------------------------------------------

def main():
    model = load_model()
    results_sec = []
    results_subsec = []

    # Combine persona/job for embedding prompt
    query_prompt = f"Persona: {PERSONA}\nJob-to-be-done: {JOB_TO_BE_DONE}"
    query_embedding = model.encode([query_prompt], show_progress_bar=False)[0]

    input_metadata = {
        "input_documents": [os.path.basename(d) for d in DOCUMENTS],
        "persona": PERSONA,
        "job_to_be_done": JOB_TO_BE_DONE,
        "processing_timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }

    # Step 1: Parse & extract sections to rank
    all_sections = []
    section_locator = []
    for doc_path in DOCUMENTS:
        doc_sections = get_doc_sections(doc_path)
        for sec in doc_sections:
            sec_entry = {
                "document": os.path.basename(doc_path),
                "page_number": sec["page"],
                "section_title": sec["title"],
                "text": sec["text"]
            }
            all_sections.append(sec_entry)
    # Step 2: Rank sections w.r.t. query
    ranked_sec = rank_sections(all_sections, query_embedding, model)
    # Add top K (e.g., top 8) for output, with importance rank
    for rank, (score, sec) in enumerate(ranked_sec[:8], 1):
        results_sec.append({
            "document": sec["document"],
            "page_number": sec["page_number"],
            "section_title": sec["section_title"],
            "importance_rank": rank
        })
        # Analyze subsections (granular chunks)
        subsec_results = analyze_subsections(sec, query_embedding, model)
        for sub in subsec_results:
            results_subsec.append({
                "document": sec["document"],
                "refined_text": sub["Refined Text"],
                "page_number": sub["Page Number"],
            })

    output_obj = {
        "metadata": input_metadata,
        "extracted_section": results_sec,
        "sub_section_analysis": results_subsec
    }
    # Write to output file
    with open("challenge1b_output.json", "w", encoding="utf-8") as fout:
        json.dump(output_obj, fout, indent=2, ensure_ascii=False)
    print("[INFO] Output written to challenge1b_output.json")

if __name__ == "__main__":
    main()
