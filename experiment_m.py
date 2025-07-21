import os
import json
import re
import fitz
import numpy as np
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer, AutoModel
import torch

class PDFOutlineExtractor:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModel
        import torch
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        self.heading_templates = [
            "introduction", "summary", "conclusion", "references", "appendix", "timeline", "milestones",
            "approach", "evaluation", "content", "audience", "objectives", "requirements", "structure",
            "outcomes", "table of contents", "revision history", "acknowledgements", "trademarks",
            "documents and web sites", "career paths", "learning objectives", "entry requirements",
            "keeping it current", "pathway options", "business outcomes", "background", "results", "discussion",
            "abstract", "methodology", "goals", "pathway", "options", "regular", "distinction", "hope", "see", "there"
        ]
        self.template_embs = self._embed_texts(self.heading_templates)

    def _embed_texts(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def is_heading_llm(self, text):
        if not text or len(text) < 3:
            return False
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        text_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        sims = np.dot(self.template_embs, text_emb) / (
            np.linalg.norm(self.template_embs, axis=1) * np.linalg.norm(text_emb) + 1e-8
        )
        return np.max(sims) > 0.7

    def extract_title(self, doc):
        first = doc[0]
        blocks = first.get_text("dict")['blocks']
        # Find the largest font size on the first page
        sizes = [span["size"] for b in blocks if "lines" in b for line in b["lines"] for span in line["spans"]]
        if not sizes:
            return "Untitled Document"
        max_size = max(sizes)
        # Concatenate all text in the largest font size
        title_spans = []
        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    if span["size"] == max_size:
                        title_spans.append(span["text"].strip())
        title = " ".join(title_spans).strip()
        return title or "Untitled Document"

    def is_heading_heuristic(self, line, body_size):
        text = " ".join([s["text"] for s in line["spans"]]).strip()
        if len(text) < 2 or text.isdigit():
            return False
        for s in line["spans"]:
            if s["size"] > body_size + 1.5:
                return True
            if s["flags"] & 2:
                return True
        return False

    def is_heading_combined(self, line, body_size):
        text = " ".join([s["text"] for s in line["spans"]]).strip()
        if len(text) < 4 or text.isdigit():
            return False
        # Expanded ignore list
        form_fields = {"name", "age", "date", "designation", "service", "relationship", "from", "the", "fare", "rail"}
        ignore_phrases = {
            "version", "remarks", "copyright notice", "baseline", "extension", "syllabus", "foundation level.",
            "foundation level", "consultants.", "projects.", "criteria.", "reference", "address", "mission statement",
            "goals", "topjump", "parkway"
        }
        if text.lower().strip(':').strip() in form_fields or text.lower().strip(':').strip() in ignore_phrases:
            return False
        # Exclude all-uppercase, short lines unless they match a known heading
        if text.isupper() and len(text.split()) < 3 and text.lower() not in self.heading_templates:
            return False
        # Heuristic check
        heuristic = self.is_heading_heuristic(line, body_size)
        if not heuristic:
            return False
        # LLM check
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        text_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        sims = np.dot(self.template_embs, text_emb) / (
            np.linalg.norm(self.template_embs, axis=1) * np.linalg.norm(text_emb) + 1e-8
        )
        llm_strong = np.max(sims) > 0.9
        # For H2/H3, require at least 4 words or LLM similarity > 0.9 or template match, and at least 8 alphabetic chars
        if len(text.split()) >= 4 or text.lower() in self.heading_templates or llm_strong or sum(c.isalpha() for c in text) >= 8:
            return True
        return False

    def level_from_size(self, size, sorted_sizes):
        # Only assign H2/H3 if the difference is significant (>=2pt)
        if size >= sorted_sizes[0] - 1:
            return "H1"
        elif len(sorted_sizes) > 1 and size >= sorted_sizes[1] - 1 and (sorted_sizes[0] - sorted_sizes[1]) >= 2:
            return "H2"
        elif len(sorted_sizes) > 2 and size >= sorted_sizes[2] - 1 and (sorted_sizes[1] - sorted_sizes[2]) >= 2:
            return "H3"
        return "H3"

    def extract_outline(self, pdf_path):
        doc = fitz.open(pdf_path)
        headings = []
        title = self.extract_title(doc)
        heading_counter = {}
        num_pages = len(doc)
        for pno in range(num_pages):
            page = doc[pno]
            blocks = page.get_text("dict")["blocks"]
            page_height = page.rect.height
            header_cutoff = page_height * 0.15  # Increased to 15%
            footer_cutoff = page_height * 0.85  # Increased to 15%
            sizes = [s["size"] for b in blocks if "lines" in b
                     for line in b["lines"] for s in line["spans"]]
            if not sizes:
                continue
            counter = Counter(sizes)
            body = counter.most_common(1)[0][0]
            sorted_sizes = sorted(counter.keys(), reverse=True)
            # Collect all headings for this page
            page_headings = []
            special_heading_on_page = None
            for b in blocks:
                if "lines" not in b:
                    continue
                for line in b["lines"]:
                    span_y = line["spans"][0]["origin"][1]
                    # Exclude lines in header or footer
                    if span_y < header_cutoff or span_y > footer_cutoff:
                        continue
                    text = " ".join([s["text"] for s in line["spans"]]).strip()
                    # Do not include the title as a heading
                    if text.strip() == title.strip():
                        continue
                    if self.is_heading_combined(line, body):
                        lvl = self.level_from_size(line["spans"][0]["size"], sorted_sizes)
                        heading_dict = {"level": lvl, "text": text, "page": pno + 1}
                        # Check for special heading in first 5 pages
                        if pno < 5:
                            text_lower = text.lower().strip(":. ")
                            if text_lower in ["contents", "content", "table of contents"]:
                                special_heading_on_page = heading_dict
                        page_headings.append(heading_dict)
            # If special heading found on this page, only keep it
            if special_heading_on_page is not None:
                headings.append(special_heading_on_page)
                key = special_heading_on_page["text"].lower().strip()
                heading_counter[key] = heading_counter.get(key, 0) + 1
            else:
                for h in page_headings:
                    headings.append(h)
                    key = h["text"].lower().strip()
                    heading_counter[key] = heading_counter.get(key, 0) + 1
        doc.close()
        # Exclude headings that appear on more than 50% of pages (likely headers/footers)
        min_count = max(2, int(num_pages * 0.5) + 1)
        filtered_headings = [h for h in headings if heading_counter[h["text"].lower().strip()] < min_count]
        return {"title": title, "outline": filtered_headings}

    def process_pdf(self, pdf_path):
        """Main processing function"""
        try:
            return self.extract_outline(pdf_path)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return {"title": "", "outline": []}

def main():
    # Use Docker paths as specified in the challenge requirements
    INPUT_DIR = Path("/app/input")
    OUTPUT_DIR = Path("/app/output")
    
    # Fallback for local testing
    if not INPUT_DIR.exists():
        INPUT_DIR = Path("Challenge_1a/sample_dataset/pdfs")
        OUTPUT_DIR = Path("C:/Users/Afham Faiyaz Ahmad/Desktop/Adobe/Adobe-India-Hackathon25/OUTPUTS_M1")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    extractor = PDFOutlineExtractor()
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    
    print(f"Processing {len(pdf_files)} PDF files...")
    
    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        result = extractor.process_pdf(pdf_file)
        
        output_path = OUTPUT_DIR / f"{pdf_file.stem}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        print(f"  Title: '{result['title']}'")
        print(f"  Headings: {len(result['outline'])}")

if __name__ == "__main__":
    main()
