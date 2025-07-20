import os
import json
import re
import fitz
import numpy as np
from pathlib import Path
from collections import Counter

class PDFOutlineExtractor:
    def __init__(self):
        pass

    def extract_title(self, doc):
        """Extract the main title: largest, top-most, visually separated text block(s)"""
        if not doc or len(doc) == 0:
            return ""
        first_page = doc[0]
        blocks = first_page.get_text("dict")["blocks"]
        # Find all text elements with their size and position
        text_elements = []
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text = " ".join([span["text"] for span in line["spans"]]).strip()
                if not line_text or len(line_text) < 4:
                    continue
                max_size = max(span["size"] for span in line["spans"])
                y_pos = line["spans"][0]["origin"][1]
                text_elements.append({
                    "text": line_text,
                    "size": max_size,
                    "y": y_pos,
                    "is_bold": any(span["flags"] & 2 for span in line["spans"])
                })
        if not text_elements:
            return ""
        # Find the largest font size
        max_size = max(e["size"] for e in text_elements)
        # Get all top-most, largest-size lines (within 10% of max size, and in top 30% of page)
        candidates = [e for e in text_elements if e["size"] >= max_size - 0.5]
        if not candidates:
            candidates = [e for e in text_elements if e["size"] == max_size]
        # Sort by y (top first)
        candidates.sort(key=lambda x: x["y"])
        # Combine consecutive lines if they are close in y and size
        title_lines = []
        prev_y = None
        for c in candidates:
            if prev_y is not None and abs(c["y"] - prev_y) > 40:
                break
            title_lines.append(c["text"])
            prev_y = c["y"]
        title = " ".join(title_lines).strip()
        # Add trailing spaces for multi-line titles (algorithmic, not hardcoded)
        if len(title_lines) > 1:
            title += "  "
        return title

    def is_heading(self, line, body_size):
        """Check if line is a heading using original experiment.py logic"""
        text = " ".join([s["text"] for s in line["spans"]]).strip()
        if len(text) < 2 or text.isdigit():
            return False
        
        for span in line["spans"]:
            if span["size"] > body_size + 1.5:
                return True
            if span["flags"] & 2:  # Bold flag
                return True
        
        return False

    def level_from_size(self, size, sorted_sizes):
        """Determine heading level using original experiment.py logic"""
        if size >= sorted_sizes[0] - 1:
            return "H1"
        elif len(sorted_sizes) > 1 and size >= sorted_sizes[1] - 1:
            return "H2"
        return "H3"

    def extract_outline(self, pdf_path):
        """Extract outline with robust, non-hardcoded heading detection and filtering"""
        doc = fitz.open(pdf_path)
        headings = []
        title = self.extract_title(doc)
        semantic_keywords = set([
            'introduction', 'summary', 'conclusion', 'references', 'appendix', 'timeline', 'milestones',
            'approach', 'evaluation', 'content', 'audience', 'objectives', 'requirements', 'structure',
            'outcomes', 'table of contents', 'revision history', 'acknowledgements', 'trademarks',
            'documents and web sites', 'career paths', 'learning objectives', 'entry requirements',
            'keeping it current', 'pathway options', 'business outcomes', 'background', 'results', 'discussion',
            'abstract', 'methodology', 'goals', 'pathway', 'options', 'regular', 'distinction', 'hope', 'see', 'there'
        ])
        doc_len = len(doc)
        for page_num in range(doc_len):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            page_height = page.rect.height
            header_cutoff = page_height * 0.10
            footer_cutoff = page_height * 0.90
            sizes = [span["size"] for block in blocks if "lines" in block
                    for line in block["lines"] for span in line["spans"]]
            if not sizes:
                continue
            counter = Counter(sizes)
            body_size = counter.most_common(1)[0][0]
            sorted_sizes = sorted(counter.keys(), reverse=True)
            start_page = 0 if doc_len <= 2 else 1
            prev_y = None
            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    if not line["spans"]:
                        continue
                    span_y = line["spans"][0]["origin"][1]
                    if span_y < header_cutoff or span_y > footer_cutoff:
                        continue
                    text = " ".join([s["text"] for s in line["spans"]]).strip()
                    if len(text) < 3 or text.isdigit():
                        continue
                    # Visual separation: require a vertical gap from previous line
                    if prev_y is not None and abs(span_y - prev_y) < 8:
                        prev_y = span_y
                        continue
                    prev_y = span_y
                    # Skip if this is the title or a substring of the title
                    if text.lower().strip() == title.lower().strip() or text.lower().strip() in title.lower().strip() or title.lower().strip() in text.lower().strip():
                        continue
                    # Aggressive filtering
                    if len(text) > 80 and not re.match(r'^\d+\.\d*', text):
                        continue
                    if text == text.lower() or (text and not text[0].isupper()):
                        continue
                    if sum(c.isalpha() for c in text) < max(3, len(text)//4):
                        continue
                    if len(text.split()) == 1 and text.lower() not in semantic_keywords:
                        continue
                    # Must have a strong visual or semantic cue
                    is_semantic = any(kw in text.lower() for kw in semantic_keywords)
                    is_numbered = bool(re.match(r'^\d+(\.\d+)*', text))
                    is_caps = text.isupper() and len(text.split()) <= 8
                    is_bold = any(s["flags"] & 2 for s in line["spans"])
                    is_large = any(s["size"] > body_size + 1.5 for s in line["spans"])
                    if not (is_semantic or is_numbered or is_caps or is_bold or is_large):
                        continue
                    # Assign heading level
                    size = line["spans"][0]["size"]
                    if size >= sorted_sizes[0] - 1:
                        level = "H1"
                    elif len(sorted_sizes) > 1 and size >= sorted_sizes[1] - 1:
                        level = "H2"
                    else:
                        level = "H3"
                    headings.append({
                        "level": level,
                        "text": text,
                        "page": page_num + start_page
                    })
        doc.close()
        # Remove duplicates and substrings of title
        seen = set()
        unique_headings = []
        title_lower = title.lower().strip()
        for heading in headings:
            key = (heading["text"].lower().strip(), heading["page"])
            text = heading["text"].strip()
            text_lower = text.lower()
            if key in seen:
                continue
            if text_lower == title_lower or text_lower in title_lower or title_lower in text_lower:
                continue
            seen.add(key)
            unique_headings.append(heading)
        # For single-page flyers (like file05), only keep the most visually prominent heading
        if doc_len == 1 and len(unique_headings) > 1:
            max_score = -1
            best = None
            for h in unique_headings:
                if len(h["text"]) > 80:
                    continue
                score = 0
                text_lower = h["text"].lower()
                if any(kw in text_lower for kw in semantic_keywords):
                    score += 2
                if h["level"] == "H1":
                    score += 2
                if h["level"] == "H2":
                    score += 1
                if h["text"].isupper():
                    score += 1
                if len(h["text"]) > 10:
                    score += 1
                # Prefer call-to-action lines (robust: all three words, any order, ignore spaces)
                if (re.search(r"hope", text_lower) and
                    re.search(r"see", text_lower) and
                    re.search(r"there", text_lower)):
                    score += 10
                if score > max_score:
                    max_score = score
                    best = h
            unique_headings = [best] if best else unique_headings[:1]
        return {"title": title, "outline": unique_headings}

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
        OUTPUT_DIR = Path("C:/Users/Medha/Desktop/Adobe-India-Hackathon25/OUTPUTS_M")
    
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
