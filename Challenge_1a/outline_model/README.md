# Advanced PDF Outline Extraction Model

This module is designed for Challenge 1A of the Adobe India Hackathon. It extracts structured outlines (headings like H1, H2, H3) from PDFs, converting unstructured content into a clean, navigable hierarchy.

## Features
- **Layout-aware model integration** (e.g., LayoutLMv3/Donut, with 200MB model size constraint)
- **Statistical/ML-based clustering** of font attributes (size, boldness, indentation) to dynamically classify headings
- **Language-based cues** for section title detection (using common patterns and ML/NLP)
- Modular design for easy extension and experimentation

## Structure
- `outline_extractor.py`: Main extraction logic
- `clustering.py`: Font attribute clustering utilities
- `language_cues.py`: Language-based heading detection
- `model_utils.py`: Layout-aware model integration (stub for now)

## Usage
Import and use the `OutlineExtractor` class from `outline_extractor.py`.

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

---
This module is a work in progress and designed for extensibility and experimentation. 