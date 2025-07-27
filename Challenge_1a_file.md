# PDF Outline Extractor

## What This Project Does

This project is a smart tool that reads PDF documents and pulls out their structure, the document’s title and its main section headings. Instead of just grabbing raw text, it understands how documents are visually and logically organized. It looks at font sizes, formatting, and where text appears on the page to find titles and headings. Then, it enhances this understanding by using a language model to interpret the meaning of text, ensuring that section titles are identified accurately even if they are phrased in different ways.

In essence, it turns a static PDF into a navigable outline, like a digital table of contents you can work with programmatically.

## How It Works

1. **Reading the PDF Layout:** The program uses the powerful PyMuPDF library to read the PDF pages carefully. Every piece of text is inspected, along with its font size and position on the page. This helps separate the main content from repeated elements like headers and footers.

2. **Finding the Document Title:** On the first page, it finds the largest text, usually the title, and combines all pieces with that font size to capture the full title.

3. **Detecting Headings Using Multiple Clues:**

   - It compares the font sizes of text lines to identify candidates that stand out as headings.
   - It checks if text is bold or significantly larger than normal body text.
   - It ignores lines that look like page numbers, footers, or headers by checking their vertical placement.
   - It uses a pretrained BERT language model to turn heading candidates into numeric vectors (embeddings).
   - It compares these embeddings to a set of vectors derived from common heading phrases (like “Introduction”, “Conclusion”, “Table of Contents”), measuring similarity to confirm if text is likely a heading.

4. **Assigning Heading Levels:** Based on font size differences, the program assigns levels like H1, H2, or H3, representing the hierarchy, similar to chapters, sections, and subsections.

5. **Filtering Out Noise:** It removes headings that appear too frequently across pages (likely page headers or footers) to keep only meaningful section titles.

6. **Outputting the Result:** Finally, it writes a JSON file for each PDF containing the extracted title and an array of headings with their level and page number, ready for use in search, navigation, or further analysis.

## Key Technologies and Why They Matter

- **PyMuPDF (fitz):** Gives direct access to the PDF’s visual and textual layout, essential for understanding structure beyond raw text.

- **Hugging Face Transformers (BERT-tiny):** Enables semantic understanding of heading texts, making the extraction robust to wording variations and noise.

- **PyTorch:** Powers the model computations efficiently.

- **NumPy:** Provides fast, reliable math operations to compare text embeddings via cosine similarity.

- **Python Standard Libraries:** Manage files, JSON output, and basic utilities ensuring smooth, cross-platform operation.



## Why This Matters

PDF documents are everywhere, but automating their understanding has always been tough because PDFs don’t store content hierarchy explicitly. This tool bridges that gap by combining visual cues with intelligent language processing. It’s useful anywhere you need to automate document indexing, improve searchability, or build smart reading tools.
