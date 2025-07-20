import fitz

def debug_title_detection(pdf_path):
    """Debug title detection for a specific PDF"""
    doc = fitz.open(pdf_path)
    first_page = doc[0]
    blocks = first_page.get_text("dict")["blocks"]
    
    print(f"\n=== DEBUGGING TITLE DETECTION FOR {pdf_path} ===")
    
    text_elements = []
    
    for block in blocks:
        if "lines" not in block:
            continue
        
        for line in block["lines"]:
            line_text = ""
            max_size_in_line = 0
            line_y = line["bbox"][1]
            
            # Skip header/footer areas
            page_height = first_page.rect.height
            if line_y < page_height * 0.10 or line_y > page_height * 0.90:
                continue
            
            for span in line["spans"]:
                text = span["text"].strip()
                if text:
                    line_text += text + " "
                    max_size_in_line = max(max_size_in_line, span["size"])
            
            line_text = line_text.strip()
            if len(line_text) >= 8 and len(line_text) <= 200:
                text_elements.append({
                    "text": line_text,
                    "size": max_size_in_line,
                    "position": line_y,
                    "is_bold": any(s["flags"] & 2**4 for s in line["spans"])
                })
    
    # Sort by size (largest first), then by position (top first)
    text_elements.sort(key=lambda x: (-x["size"], x["position"]))
    
    print("Top 10 text elements by size:")
    for i, element in enumerate(text_elements[:10]):
        print(f"  {i+1}. Size:{element['size']:.1f} Bold:{element['is_bold']} | {element['text']}")
    
    doc.close()

# Debug the problematic files
debug_title_detection("Challenge_1a/sample_dataset/pdfs/file04.pdf")
debug_title_detection("Challenge_1a/sample_dataset/pdfs/file05.pdf") 