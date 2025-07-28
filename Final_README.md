# Technical Overview of PDF Processing Algorithms

This document provides an algorithmic explanation of two Python scripts for multilingual PDF content extraction and relevance scoring.

---

## 1. Multilingual PDF Outline Extraction

### **Objective:**

Extract the document title and a structured outline of headings from PDFs in various languages.

### **Algorithm Workflow:**

#### **Model and Template Loading**

* Loads a local, multilingual BERT model to produce embeddings for text.
* Computes and caches vector representations for common heading templates (e.g., "Introduction", "Summary") in several languages.

#### **Language Detection**

* Analyzes script patterns in PDF text samples using regular expressions and language heuristics to determine the document’s language.
* Adjusts processing rules (such as templates and regexes) accordingly.

#### **Title Extraction**

* On the first page, identifies the largest font size in use.
* Concatenates all text fragments using this font to estimate the document’s title, avoiding substring repetitions.

#### **Outline Parsing**

* On each page, collects all text spans, filtering out headers and footers.
* For each text line, evaluates font size, boldness, and placement to identify heading-like candidates.
* Compares the embedding of each candidate heading with the cached heading templates for the detected language using cosine similarity.
* Heading level (H1/H2/H3) is inferred from font size hierarchy.
* Excludes non-informative headings (like "Table of Contents") and adjacent duplicates.

### **Output:**

Produces a JSON listing the document title and all extracted headings, each annotated with its page and level.

---

## 2. PDF Collection Section Relevance Extraction

### **Objective:**

From multiple PDF collections and a persona/job configuration, extract and rank relevant sections for the given persona.

### **Algorithm Workflow:**

#### **Initialization**

* Loads a compact multilingual BERT model and, if available, a summarization model (both local).
* Loads persona/job description and sets up heading/keyword regexes appropriate for the supported languages.

#### **Collection & Language Detection**

* Scans for valid PDF collections (identified by configuration files and docs).
* Detects language using script and regex analysis, adapting section extraction to the main document language.

#### **Section Heading Identification**

* Iterates through PDF pages, skipping headers/footers.
* Considers text lines as candidate headings based on font size, length, and persona/job-related regex matches.

#### **Relevance Scoring**

* Calculates cosine similarity between embeddings of candidate headings and the persona/job description.
* Adds bonus scores for keyword matches from the persona/job config.

#### **Post-processing**

* Deduplicates sections by document and heading.
* Selects top N most relevant sections across all documents.

### **Output:**

For each collection, creates a JSON with:

* Metadata
* Ranked relevant sections (heading, page, score)
* Text snippets near each heading

---

> **Note:**
> Both scripts exclusively use **offline models** and are built for robust **multilingual PDF analysis**.

---
