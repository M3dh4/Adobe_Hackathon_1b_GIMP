import fitz
import json

def debug_pdf(pdf_path):
    """Debug PDF content to understand structure"""
    doc = fitz.open(pdf_path)
    print(f"\n=== DEBUGGING {pdf_path} ===")
    print(f"Number of pages: {len(doc)}")
    
    for page_num in range(min(3, len(doc))):  # First 3 pages
        page = doc[page_num]
        print(f"\n--- PAGE {page_num + 1} ---")
        
        # Get text blocks
        blocks = page.get_text("dict")["blocks"]
        
        for block_idx, block in enumerate(blocks):
            if "lines" not in block:
                continue
                
            for line_idx, line in enumerate(block["lines"]):
                line_text = ""
                max_size = 0
                is_bold = False
                
                for span in line["spans"]:
                    text = span["text"]
                    size = span["size"]
                    flags = span["flags"]
                    
                    line_text += text
                    max_size = max(max_size, size)
                    if flags & 2**4:  # Bold flag
                        is_bold = True
                
                line_text = line_text.strip()
                if line_text and len(line_text) > 3:
                    print(f"  [{block_idx}:{line_idx}] Size:{max_size:.1f} Bold:{is_bold} | {line_text}")
    
    doc.close()

# Debug the problematic files
debug_pdf("Challenge_1a/sample_dataset/pdfs/file03.pdf")
debug_pdf("Challenge_1a/sample_dataset/pdfs/file04.pdf") 
debug_pdf("Challenge_1a/sample_dataset/pdfs/file05.pdf") 