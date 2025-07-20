import os
import json
import fitz
from collections import Counter

# Input PDF (hardcoded)
PDF_NAME = "/home/medha/Adobe-India-Hackathon25/Challenge_1a/sample_dataset/pdfs/file01.pdf"  # change to file02.pdf etc as needed
PDF_PATH = os.path.join("Challenge_1a", "sample_dataset", "pdfs", PDF_NAME)

# Root-level output folder
OUTPUT_DIR = "sample_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
JSON_PATH = os.path.join(OUTPUT_DIR, PDF_NAME.replace(".pdf", ".json"))

def extract_title(doc):
    first = doc[0]
    blocks = first.get_text("dict")["blocks"]
    max_size = 0
    prev_max=0
    title = ""
    pseudo=""
    for b in blocks:
        print(b,"\n")
        if "lines" not in b: continue
        for line in b["lines"]:
            for span in line["spans"]:
                if span["size"] >= max_size:
                    prev_max=max_size
                    max_size = span["size"]
                    # pseudo=span["text"].strip()
                    title = title + " " + span["text"].strip()
    return title or "Untitled Document"

def is_heading(line, body_size):
    text = " ".join([s["text"] for s in line["spans"]]).strip()
    if len(text) < 2 or text.isdigit(): return False
    for s in line["spans"]:
        if s["size"] > body_size + 1.5: return True
        if s["flags"] & 2 : return True


    return False

def level_from_size(size, sorted_sizes):
    if size >= sorted_sizes[0] - 1:
        return "H1"
    elif len(sorted_sizes) > 1 and size >= sorted_sizes[1] - 1:
        return "H2"
    return "H3"

def extract_outline(path):
    doc = fitz.open(path)
    headings = []

    for pno in range(len(doc)):
        page = doc[pno]
        blocks = page.get_text("dict")["blocks"]
        page_height = page.rect.height

        header_cutoff = page_height * 0.10
        footer_cutoff = page_height * 0.90

        sizes = [s["size"] for b in blocks if "lines" in b
                 for line in b["lines"] for s in line["spans"]]
        if not sizes:
            continue

        counter = Counter(sizes)
        body = counter.most_common(1)[0][0]
        sorted_sizes = sorted(counter.keys(), reverse=True)

        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                span_y = line["spans"][0]["origin"][1]
                if span_y < header_cutoff or span_y > footer_cutoff:
                    continue
                text = " ".join([s["text"] for s in line["spans"]]).strip()
                if is_heading(line, body):
                    lvl = level_from_size(line["spans"][0]["size"], sorted_sizes)
                    headings.append({"level": lvl, "text": text, "page": pno + 1})
    return {"title": extract_title(doc), "outline": headings}


if __name__ == "__main__":
    result = extract_outline(PDF_PATH)
    with open(JSON_PATH, "w", encoding="utf-8") as out:
        json.dump(result, out, indent=2, ensure_ascii=False)
    print(f"✅ Wrote output ➜ {JSON_PATH}")
