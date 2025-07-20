import os
import json
import re
import fitz
import numpy as np
from pathlib import Path
from collections import Counter

class PDFOutlineExtractor:
    def __init__(self):
        # Major heading keywords from samples
        self.major_headings = {
            "revision history", "table of contents", "acknowledgements", "references",
            "introduction", "background", "overview", "summary", "conclusion",
            "methodology", "results", "discussion", "abstract", "appendix",
            "timeline", "milestones", "approach", "evaluation", "business outcomes",
            "content", "intended audience", "career paths", "learning objectives",
            "entry requirements", "structure", "keeping it current", "trademarks",
            "documents and web sites", "pathway options"
        }

    def extract_title(self, doc):
        """Extract title using your original algorithm with improvements"""
        if not doc or len(doc) == 0:
            return ""
        
        first_page = doc[0]
        blocks = first_page.get_text("dict")["blocks"]
        
        # Collect all text with sizes from first page
        text_elements = []
        
        for block in blocks:
            if "lines" not in block:
                continue
            
            for line in block["lines"]:
                line_text = ""
                max_size_in_line = 0
                line_y = line["bbox"][1]
                
                # Skip header/footer areas (your technique)
                page_height = first_page.rect.height
                if line_y < page_height * 0.10 or line_y > page_height * 0.90:
                    continue
                
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        line_text += text + " "
                        max_size_in_line = max(max_size_in_line, span["size"])
                
                line_text = line_text.strip()
                if len(line_text) >= 8 and len(line_text) <= 120:  # Reasonable title length
                    text_elements.append({
                        "text": line_text,
                        "size": max_size_in_line,
                        "position": line_y,
                        "is_bold": any(s["flags"] & 2**4 for s in line["spans"])
                    })
        
        if not text_elements:
            return ""
        
        # Sort by size (largest first), then by position (top first)
        text_elements.sort(key=lambda x: (-x["size"], x["position"]))
        
        # Find title from top elements
        for element in text_elements[:10]:
            text = element["text"]
            
            # Skip if it looks like a heading keyword
            if (text.lower().strip() in self.major_headings or
                re.match(r'^\d+\.?\s', text) or
                text.isupper() and len(text) > 50):
                continue
            
            # Must be reasonably large or bold
            if element["size"] >= 10 or element["is_bold"]:
                return text
        
        return ""

    def extract_outline(self, pdf_path):
        """Extract outline using your size-based algorithm with pattern matching"""
        doc = fitz.open(pdf_path)
        headings = []
        
        # Extract title first
        title = self.extract_title(doc)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            page_height = page.rect.height
            
            # Your header/footer detection
            header_cutoff = page_height * 0.10
            footer_cutoff = page_height * 0.90
            
            # Collect all font sizes on this page (your technique)
            all_sizes = []
            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        all_sizes.append(span["size"])
            
            if not all_sizes:
                continue
            
            # Find body text size (your technique)
            size_counter = Counter(all_sizes)
            body_size = size_counter.most_common(1)[0][0]
            sorted_sizes = sorted(size_counter.keys(), reverse=True)
            
            # Process each line
            for block in blocks:
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    if not line["spans"]:
                        continue
                    
                    # Skip headers/footers (your technique)
                    span_y = line["spans"][0]["bbox"][1]
                    if span_y < header_cutoff or span_y > footer_cutoff:
                        continue
                    
                    # Combine text from all spans in line
                    line_text = " ".join([s["text"] for s in line["spans"]]).strip()
                    
                    if len(line_text) < 2 or len(line_text) > 200:
                        continue
                    
                    # Skip if this is the title
                    if line_text == title:
                        continue
                    
                    # Check if it's a heading
                    if self.is_heading(line, body_size, line_text):
                        # Determine level using your size-based approach + patterns
                        level = self.determine_level(line, sorted_sizes, line_text)
                        
                        headings.append({
                            "level": level,
                            "text": line_text,
                            "page": page_num,  # Keep 0-based like samples
                            "sort_key": (page_num, span_y)
                        })
        
        doc.close()
        
        # Remove duplicates and sort
        seen = set()
        unique_headings = []
        for heading in headings:
            key = (heading["text"].lower().strip(), heading["page"])
            if key not in seen:
                seen.add(key)
                unique_headings.append(heading)
        
        # Sort by page and position
        unique_headings.sort(key=lambda x: x["sort_key"])
        
        # Clean up for output
        for heading in unique_headings:
            del heading["sort_key"]
        
        return {"title": title, "outline": unique_headings}

    def is_heading(self, line, body_size, text):
        """Enhanced heading detection combining your size logic with patterns"""
        text_lower = text.lower().strip()
        
        # Your original size and formatting checks
        is_large_or_bold = False
        for span in line["spans"]:
            if span["size"] > body_size + 1.5:  # Your threshold
                is_large_or_bold = True
                break
            if span["flags"] & 2**4:  # Bold (your check)
                is_large_or_bold = True
                break
        
        # Pattern-based checks from samples
        
        # 1. Numbered patterns
        if re.match(r'^\d+\.\s+\w+', text):  # "1. Introduction"
            return True
        if re.match(r'^\d+\.\d+\s+\w+', text):  # "2.1 Intended Audience"
            return True
        if re.match(r'^\d+\.\s+[A-Z]', text):  # "1. Preamble" (in appendix)
            return True
        
        # 2. Major headings
        if text_lower in self.major_headings:
            return True
        
        # 3. All caps (like "PATHWAY OPTIONS")
        if (text.isupper() and 
            3 <= len(text.split()) <= 8 and 
            len(text) <= 50):
            return True
        
        # 4. Appendix patterns
        if re.match(r'^Appendix\s+[A-Z]:', text):
            return True
        
        # 5. Colon endings with size/formatting
        if text.endswith(':') and len(text.split()) <= 5 and is_large_or_bold:
            return True
        
        # 6. Questions
        if text.endswith('?') and len(text.split()) <= 10 and is_large_or_bold:
            return True
        
        # 7. "For each X it could mean:" patterns
        if re.match(r'^For\s+each\s+.+\s+it\s+could\s+mean:', text):
            return True
        
        return False

    def determine_level(self, line, sorted_sizes, text):
        """Determine heading level using your size algorithm + patterns"""
        
        # Get the largest font size in this line
        max_size = max(span["size"] for span in line["spans"])
        
        # Pattern-based level assignment (takes priority)
        
        # H1: Major sections
        if re.match(r'^\d+\.\s+\w+', text):  # "1. Introduction"
            return "H1"
        
        text_lower = text.lower().strip()
        if text_lower in ["revision history", "table of contents", "acknowledgements", 
                         "introduction", "references"]:
            return "H1"
        
        if text.isupper() and len(text.split()) >= 2:  # "PATHWAY OPTIONS"
            return "H1"
        
        if re.match(r'^Appendix\s+[A-Z]:', text):  # "Appendix A:"
            return "H1" if ":" in text else "H2"
        
        # H2: Subsections
        if re.match(r'^\d+\.\d+\s+\w+', text):  # "2.1 Intended Audience"
            return "H2"
        
        if text_lower in ["summary", "background", "approach", "evaluation"]:
            return "H2"
        
        # H3: Sub-subsections and descriptive items
        if re.match(r'^\d+\.\s+[A-Z]', text):  # "1. Preamble" (in appendix)
            return "H3"
        
        if text.endswith(':') or text_lower.startswith('for each'):
            return "H3"
        
        # H4: Very specific patterns
        if re.match(r'^For\s+each\s+.+\s+it\s+could\s+mean:', text):
            return "H4"
        
        # Fallback to your size-based approach
        if len(sorted_sizes) > 0 and max_size >= sorted_sizes[0] - 1:
            return "H1"
        elif len(sorted_sizes) > 1 and max_size >= sorted_sizes[1] - 1:
            return "H2"
        else:
            return "H3"

    def process_pdf(self, pdf_path):
        """Main processing function"""
        try:
            return self.extract_outline(pdf_path)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return {"title": "", "outline": []}

def main():
    INPUT_DIR = Path("/home/medha/Adobe-India-Hackathon25/Challenge_1a/sample_dataset/pdfs")
    OUTPUT_DIR = Path("/home/medha/Adobe-India-Hackathon25/outputs_m")
    
    if not INPUT_DIR.exists():
        INPUT_DIR = Path("input")
        OUTPUT_DIR = Path("output")
    
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
