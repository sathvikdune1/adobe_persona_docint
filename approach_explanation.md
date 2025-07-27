Approach Explanation (Persona-Driven Document Intelligence)

Our solution takes a generic approach to prioritize and extract relevant document sections and subsections based on any specified persona and job-to-be-done.

Document Parsing: We use PyPDF2 to extract text from PDFs page-wise. Each page is heuristically segmented into sections by identifying headings in ALL CAPS or numbered section lines.

Section Embedding: Sections are embedded using the lightweight MiniLM sentence-transformers model, which fits comfortably under 1GB RAM usage. The combined persona/job-to-be-done prompt is also embedded for relevance alignment.

Section Ranking: All sections are compared to the persona-job query using cosine similarity. The top-ranked sections are selected and assigned importance_rank.

Sub-Section Analysis: Each top section is broken into smaller textual chunks. The same embedding and similarity ranking is performed to find and output the most aligned sub-portions of text.

Output: The system outputs all data in the prescribed JSON format, with full metadata. The entire workflow is designed to generalize to arbitrary document, persona, and job-to-be-done inputs, and runs easily within the required time on CPU.