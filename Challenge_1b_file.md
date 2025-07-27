
# Challenge 1B: Persona-Driven PDF Section Extractor

## What This Project Does

This project is an intelligent and flexible system that processes collections of PDF documents to extract, rank, and analyze relevant document sections. Unlike simple outline extractors, it is designed to tailor its extraction based on a specific **persona** (for example, a Travel Planner, HR Professional, or Food Contractor) and the **job-to-be-done**. This ensures you get sections most meaningful and actionable for the user’s specific role and needs.

By combining document layout insight, semantic language understanding, and persona-specific rules, it identifies which parts of PDFs truly matter for a given context, then organizes and surfaces those key sections along with detailed content excerpts.

## How It Works

1. **Persona-Aware Section Detection:**  
   The system is equipped with predefined regex patterns and keywords relevant to different personas. It uses these patterns, alongside general heading heuristics (like font size, bold text, and position on the page), to precisely identify candidate section headings.

2. **Robust PDF Parsing Using PyMuPDF:**  
   The tool digs deeply into the PDF, accessing layout details such as font sizes, boldness, and vertical position, enabling it to ignore boilerplate headers, footers, and non-heading lines.

3. **Semantic Embedding & Relevance Scoring:**  
   Each discovered section title is encoded by a tiny pre-trained BERT model (`prajjwal1/bert-tiny`) into an embedding vector. It then measures semantic similarity of that heading to both the persona and the job description text, factoring in keyword matches and pattern hits. This composite score reflects how relevant a section is to the user's context.

4. **Deduplication & Ranking:**  
   To avoid noise and redundancy, detected sections are deduplicated by document, normalized section title, and page number. Only the top scoring, unique sections (up to 5) are retained and ranked by importance.

5. **Content Extraction for Subsection Analysis:**  
   For each top-ranked section, the system extracts detailed text around the heading from the respective page, providing rich context and content for deeper analysis or summarization.

6. **Output Packaging:**  
   The final output is a JSON file per collection containing:
   - Metadata (processed docs, persona, job description, timestamp)
   - A ranked list of extracted and deduplicated sections with page numbers
   - Detailed text snippets for each section to support further tasks.

## Key Technologies and Why They Matter

- **PyMuPDF (fitz):** Essential for low-level access to visual and textual PDF structure, enabling nuanced layout-based filtering (like ignoring headers/footers and accurately detecting font styling).

- **Transformers (Hugging Face) & PyTorch:** Used to generate semantic embeddings with a compact BERT model (`prajjwal1/bert-tiny`). This allows the system to “understand” the meaning of section titles relative to persona goals and tasks, rather than relying solely on keyword matches.

- **Summarization Pipeline (optional):** The code attempts to load a DistilBART summarizer for future enhancement in content summarization, demonstrating extensibility.

- **Regular Expressions:** Drive powerful pattern matching customized per persona, improving precision in section detection.

- **NumPy:** Handles efficient numerical computation for embedding similarity comparisons.

- **collections.Counter and defaultdict:** Help aggregate font sizes and organize extracted headings effectively.

- **Python Standard Libraries (json, pathlib, logging, etc.):** Ensure reliable file handling, error logging, and structured output.



### Output

The output JSON will contain for each collection:

- Metadata about input docs, persona, job, timestamp
- An ordered list of top relevant sections per document, with their importance scores and page numbers
- Detailed extracted text snippets to support further processing or summarization

## Why This Solution Matters

Understanding and extracting relevant information from PDFs is difficult, especially when the relevance depends on the user’s role or task. This system fills that gap by adapting the extraction logic dynamically for specific personas, combining surface text clues with deep semantic understanding.

This approach prevents information overload, helps highlight actionable document parts, and empowers applications in areas like travel planning, HR processing, or catering management, where targeted insights save time and improve decision-making.

## Conclusion

This persona-driven PDF section extractor elegantly merges heuristic PDF layout analysis, pattern matching, and state-of-the-art language modeling to yield highly relevant and ranked document highlights tailored to user needs. Its modular design and clear outputs make it a powerful tool for automating intelligent document understanding in practical, role-specific settings.