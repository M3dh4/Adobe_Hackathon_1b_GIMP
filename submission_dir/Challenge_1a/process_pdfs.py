#!/usr/bin/env python3
"""
Challenge 1A: PDF Outline Extraction with Multilingual Support
Final version with strict compliance to Challenge 1A output format requirements
"""

import os
import json
import re
import logging
import time
import unicodedata
from pathlib import Path
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import dependencies with error handling
try:
    import fitz
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    import torch
    logger.info("✅ All dependencies imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import dependencies: {e}")
    exit(1)

class MultilingualPDFOutlineExtractor:
    def __init__(self):
        """Initialize with multilingual support while preserving your working logic"""
        logger.info("Initializing Multilingual PDF Outline Extractor...")
        
        try:
            # Set offline mode for Docker
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            self.torch = torch
            
            # Use multilingual BERT for better language support
            logger.info("Loading multilingual BERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-multilingual-cased",
                local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                "distilbert-base-multilingual-cased", 
                local_files_only=True
            )
            
            # Comprehensive multilingual heading templates
            self.heading_templates = {
                'en': [  # English - your original templates plus more
                    "introduction", "summary", "conclusion", "references", "appendix", "timeline", "milestones",
                    "approach", "evaluation", "content", "audience", "objectives", "requirements", "structure",
                    "outcomes", "table of contents", "revision history", "acknowledgements", "trademarks",
                    "documents and web sites", "career paths", "learning objectives", "entry requirements",
                    "keeping it current", "pathway options", "business outcomes", "background", "results", 
                    "discussion", "abstract", "methodology", "goals", "pathway", "options", "regular", 
                    "distinction", "hope", "see", "there", "overview", "executive summary"
                ],
                'es': [  # Spanish
                    "introducción", "resumen", "conclusión", "referencias", "apéndice", "cronología", "hitos",
                    "enfoque", "evaluación", "contenido", "audiencia", "objetivos", "requisitos", "estructura",
                    "resultados", "índice", "historial", "agradecimientos", "antecedentes", "metodología",
                    "metas", "opciones", "descripción general", "resumen ejecutivo", "discusión"
                ],
                'fr': [  # French
                    "introduction", "résumé", "conclusion", "références", "annexe", "chronologie", "jalons",
                    "approche", "évaluation", "contenu", "audience", "objectifs", "exigences", "structure",
                    "résultats", "table des matières", "historique", "remerciements", "méthodologie",
                    "buts", "options", "aperçu", "résumé exécutif", "discussion"
                ],
                'de': [  # German
                    "einführung", "zusammenfassung", "schlussfolgerung", "referenzen", "anhang", "zeitplan",
                    "meilensteine", "ansatz", "bewertung", "inhalt", "publikum", "ziele", "anforderungen",
                    "struktur", "ergebnisse", "inhaltsverzeichnis", "verlauf", "danksagungen", "methodik",
                    "überblick", "zusammenfassung der geschäftsführung", "diskussion"
                ],
                'it': [  # Italian
                    "introduzione", "riassunto", "conclusione", "riferimenti", "appendice", "cronologia",
                    "tappe", "approccio", "valutazione", "contenuto", "pubblico", "obiettivi", "requisiti",
                    "struttura", "risultati", "indice", "cronologia", "ringraziamenti", "metodologia",
                    "panoramica", "riassunto esecutivo", "discussione"
                ],
                'pt': [  # Portuguese
                    "introdução", "resumo", "conclusão", "referências", "apêndice", "cronograma", "marcos",
                    "abordagem", "avaliação", "conteúdo", "audiência", "objetivos", "requisitos", "estrutura",
                    "resultados", "índice", "histórico", "agradecimentos", "metodologia", "visão geral",
                    "resumo executivo", "discussão"
                ],
                'zh': [  # Chinese (Simplified)
                    "介绍", "摘要", "结论", "参考文献", "附录", "时间表", "里程碑", "方法", "评估", "内容",
                    "受众", "目标", "要求", "结构", "结果", "目录", "历史", "致谢", "方法论", "概述",
                    "执行摘要", "讨论", "第一章", "第二章", "第三章"
                ],
                'ja': [  # Japanese
                    "導入", "要約", "結論", "参考文献", "付録", "タイムライン", "マイルストーン", "アプローチ",
                    "評価", "コンテンツ", "対象者", "目的", "要件", "構造", "結果", "目次", "履歴",
                    "謝辞", "方法論", "概要", "エグゼクティブサマリー", "議論", "第1章", "第2章"
                ],
                'ar': [  # Arabic
                    "مقدمة", "ملخص", "خلاصة", "مراجع", "ملحق", "جدول زمني", "معالم", "نهج", "تقييم",
                    "محتوى", "جمهور", "أهداف", "متطلبات", "هيكل", "نتائج", "فهرس", "تاريخ",
                    "شكر وتقدير", "منهجية", "نظرة عامة", "ملخص تنفيذي", "مناقشة"
                ],
                'ru': [  # Russian
                    "введение", "резюме", "заключение", "ссылки", "приложение", "график", "вехи", "подход",
                    "оценка", "содержание", "аудитория", "цели", "требования", "структура", "результаты",
                    "оглавление", "история", "благодарности", "методология", "обзор", "резюме", "обсуждение"
                ],
                'hi': [  # Hindi
                    "परिचय", "सारांश", "निष्कर्ष", "संदर्भ", "परिशिष्ट", "समयसीमा", "मील के पत्थर",
                    "दृष्टिकोण", "मूल्यांकन", "सामग्री", "दर्शक", "उद्देश्य", "आवश्यकताएं", "संरचना",
                    "परिणाम", "विषय सूची", "इतिहास", "आभार", "कार्यप्रणाली", "अवलोकन", "चर्चा"
                ]
            }
            
            # Language detection patterns for script-based detection
            self.language_patterns = {
                'zh': re.compile(r'[\u4e00-\u9fff]'),  # Chinese characters
                'ja': re.compile(r'[\u3040-\u309f\u30a0-\u30ff]'),  # Hiragana and Katakana
                'ar': re.compile(r'[\u0600-\u06ff]'),  # Arabic script
                'ru': re.compile(r'[\u0400-\u04ff]'),  # Cyrillic script
                'hi': re.compile(r'[\u0900-\u097f]'),  # Devanagari script
                'th': re.compile(r'[\u0e00-\u0e7f]'),  # Thai script
                'ko': re.compile(r'[\uac00-\ud7af]'),  # Korean script
            }
            
            # Precompute embeddings for all language templates
            logger.info("Computing multilingual template embeddings...")
            self.template_embeddings = {}
            for lang, templates in self.heading_templates.items():
                try:
                    embeddings = self._embed_texts(templates)
                    self.template_embeddings[lang] = embeddings
                except Exception as e:
                    logger.warning(f"Failed to compute embeddings for {lang}: {e}")
            
            # Form fields and ignore phrases (keeping your working logic)
            self.form_fields = {
                "name", "age", "date", "designation", "service", "relationship", "from", "the", "fare", "rail"
            }
            self.ignore_phrases = {
                "version", "remarks", "copyright notice", "baseline", "extension", "syllabus", 
                "foundation level.", "foundation level", "consultants.", "projects.", "criteria.", 
                "reference", "address", "mission statement", "goals", "topjump", "parkway"
            }
            
            logger.info("✅ Multilingual PDF Outline Extractor initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize extractor: {e}")
            raise

    def detect_language(self, text_sample: str) -> str:
        """Comprehensive language detection"""
        if not text_sample or len(text_sample) < 10:
            return 'en'
        
        # First check for non-Latin scripts
        for lang, pattern in self.language_patterns.items():
            if pattern.search(text_sample):
                logger.info(f"Detected {lang} via script pattern")
                return lang
        
        # For Latin scripts, use character and word analysis
        text_lower = text_sample.lower()
        
        # Character-based detection for Latin scripts with special characters
        if any(char in text_lower for char in ['ñ', 'á', 'é', 'í', 'ó', 'ú', '¿', '¡']):
            return 'es'  # Spanish
        elif any(char in text_lower for char in ['à', 'â', 'ç', 'è', 'é', 'ê', 'ë', 'î', 'ï', 'ô', 'ù', 'û', 'ü', 'ÿ']):
            return 'fr'  # French
        elif any(char in text_lower for char in ['ä', 'ö', 'ü', 'ß']):
            return 'de'  # German
        elif any(char in text_lower for char in ['à', 'á', 'â', 'ã', 'ç', 'é', 'ê', 'í', 'ó', 'ô', 'õ', 'ú']):
            return 'pt'  # Portuguese
        elif any(char in text_lower for char in ['à', 'è', 'é', 'ì', 'í', 'î', 'ò', 'ó', 'ù', 'ú']):
            return 'it'  # Italian
        
        # Word frequency analysis for better accuracy
        words = re.findall(r'\b[a-zA-Z]+\b', text_lower)[:100]  # First 100 words
        
        if not words:
            return 'en'
        
        # Language-specific common words
        language_words = {
            'en': {'the', 'and', 'or', 'to', 'of', 'in', 'for', 'with', 'on', 'at', 'is', 'are', 'was', 'were'},
            'es': {'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su'},
            'fr': {'le', 'de', 'et', 'à', 'un', 'il', 'être', 'en', 'avoir', 'que', 'pour', 'dans', 'ce', 'son'},
            'de': {'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für'},
            'it': {'il', 'di', 'che', 'e', 'la', 'per', 'un', 'in', 'con', 'del', 'da', 'al', 'le', 'si'},
            'pt': {'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma'}
        }
        
        # Calculate scores for each language
        scores = {}
        for lang, common_words in language_words.items():
            matches = sum(1 for word in words if word in common_words)
            scores[lang] = matches / len(words) if words else 0
        
        # Find language with highest score
        best_lang = max(scores, key=scores.get)
        best_score = scores[best_lang]
        
        # Require minimum confidence for non-English detection
        if best_lang != 'en' and best_score > 0.15:
            return best_lang
        elif scores['en'] > 0.08:  # Lower threshold for English
            return 'en'
        
        return 'en'  # Default to English

    def _embed_texts(self, texts):
        """Embed texts using multilingual BERT (keeping your logic)"""
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def is_heading_llm_multilingual(self, text, detected_lang='en'):
        """Enhanced multilingual LLM heading detection (based on your method)"""
        if not text or len(text) < 3:
            return False
        
        # Get appropriate template embeddings for detected language
        template_embeddings = self.template_embeddings.get(detected_lang, self.template_embeddings['en'])
        
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        text_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        sims = np.dot(template_embeddings, text_emb) / (
            np.linalg.norm(template_embeddings, axis=1) * np.linalg.norm(text_emb) + 1e-8
        )
        return np.max(sims) > 0.7

    def extract_title(self, doc, detected_lang='en'):
        """Your title extraction logic enhanced for multilingual"""
        if not doc or len(doc) == 0:
            return ""  # Return empty string instead of "Untitled Document" for compliance
            
        first = doc[0]
        blocks = first.get_text("dict")['blocks']
        
        # Find the largest font size on the first page (your logic)
        sizes = [span["size"] for b in blocks if "lines" in b for line in b["lines"] for span in line["spans"]]
        if not sizes:
            return ""
        
        max_size = max(sizes)
        
        # Concatenate all text in the largest font size (your logic)
        title_spans = []
        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    if span["size"] == max_size:
                        text = span["text"].strip()
                        if text and len(text) > 1:
                            title_spans.append(text)
        
        title = " ".join(title_spans).strip()
        
        # Clean up title to avoid repetition
        if title:
            # Remove excessive repetition
            words = title.split()
            unique_words = []
            seen = set()
            for word in words:
                if word.lower() not in seen or len(seen) < 3:
                    unique_words.append(word)
                    seen.add(word.lower())
            title = " ".join(unique_words)
        
        return title

    def is_heading_heuristic(self, line, body_size):
        """Your original heuristic logic - keeping it unchanged"""
        text = " ".join([s["text"] for s in line["spans"]]).strip()
        if len(text) < 2 or text.isdigit():
            return False
        for s in line["spans"]:
            if s["size"] > body_size + 1.5:
                return True
            if s["flags"] & 2:
                return True
        return False

    def is_heading_combined_multilingual(self, line, body_size, detected_lang='en'):
        """Your combined logic enhanced for multilingual support"""
        text = " ".join([s["text"] for s in line["spans"]]).strip()
        if len(text) < 4 or text.isdigit():
            return False
        
        # Your form field filtering (keeping exactly as is)
        if text.lower().strip(':').strip() in self.form_fields or text.lower().strip(':').strip() in self.ignore_phrases:
            return False
        
        # Get appropriate templates for detected language
        lang_templates = self.heading_templates.get(detected_lang, self.heading_templates['en'])
        
        # Your uppercase filtering (enhanced for multilingual)
        if text.isupper() and len(text.split()) < 3 and text.lower() not in lang_templates:
            return False
        
        # Your heuristic check (unchanged)
        heuristic = self.is_heading_heuristic(line, body_size)
        if not heuristic:
            return False
        
        # Enhanced LLM check with multilingual support
        template_embeddings = self.template_embeddings.get(detected_lang, self.template_embeddings['en'])
        
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        text_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        sims = np.dot(template_embeddings, text_emb) / (
            np.linalg.norm(template_embeddings, axis=1) * np.linalg.norm(text_emb) + 1e-8
        )
        llm_strong = np.max(sims) > 0.9
        
        # Your final decision logic (keeping exactly as is)
        if len(text.split()) >= 4 or text.lower() in lang_templates or llm_strong or sum(c.isalpha() for c in text) >= 8:
            return True
        return False

    def level_from_size(self, size, sorted_sizes):
        """Your original level detection - keeping unchanged"""
        # Only assign H2/H3 if the difference is significant (>=2pt)
        if size >= sorted_sizes[0] - 1:
            return "H1"
        elif len(sorted_sizes) > 1 and size >= sorted_sizes[1] - 1 and (sorted_sizes[0] - sorted_sizes[1]) >= 2:
            return "H2"
        elif len(sorted_sizes) > 2 and size >= sorted_sizes[2] - 1 and (sorted_sizes[1] - sorted_sizes[2]) >= 2:
            return "H3"
        return "H3"

    def extract_outline(self, pdf_path):
        """Your extraction logic enhanced with language detection and STRICT OUTPUT COMPLIANCE"""
        try:
            doc = fitz.open(pdf_path)
            
            # Sample text for language detection
            sample_text = ""
            for page_num in range(min(3, len(doc))):
                page_text = doc[page_num].get_text()
                sample_text += page_text[:1000]  # First 1000 chars per page
            
            detected_lang = self.detect_language(sample_text)
            logger.info(f"Detected language: {detected_lang}")
            
            headings = []
            title = self.extract_title(doc, detected_lang)
            heading_counter = {}
            num_pages = len(doc)
            
            # Your page processing loop (keeping your logic)
            for pno in range(num_pages):
                page = doc[pno]
                blocks = page.get_text("dict")["blocks"]
                page_height = page.rect.height
                header_cutoff = page_height * 0.15  # Your values
                footer_cutoff = page_height * 0.85  # Your values
                
                sizes = [s["size"] for b in blocks if "lines" in b
                         for line in b["lines"] for s in line["spans"]]
                if not sizes:
                    continue
                    
                counter = Counter(sizes)
                body = counter.most_common(1)[0][0]
                sorted_sizes = sorted(counter.keys(), reverse=True)
                
                # Your heading collection logic
                page_headings = []
                special_heading_on_page = None
                
                for b in blocks:
                    if "lines" not in b:
                        continue
                    for line in b["lines"]:
                        if not line["spans"]:
                            continue
                        span_y = line["spans"][0]["origin"][1]
                        # Your header/footer exclusion
                        if span_y < header_cutoff or span_y > footer_cutoff:
                            continue
                        text = " ".join([s["text"] for s in line["spans"]]).strip()
                        # Your title exclusion
                        if text.strip() == title.strip():
                            continue
                        
                        # Enhanced combined check with multilingual support
                        if self.is_heading_combined_multilingual(line, body, detected_lang):
                            lvl = self.level_from_size(line["spans"][0]["size"], sorted_sizes)
                            heading_dict = {"level": lvl, "text": text, "page": pno + 1}
                            
                            # Enhanced special heading detection for multiple languages
                            if pno < 5:
                                text_lower = text.lower().strip(":. ")
                                # Table of contents patterns for different languages
                                toc_patterns = {
                                    'en': ["contents", "content", "table of contents"],
                                    'es': ["contenido", "índice", "tabla de contenidos"],
                                    'fr': ["sommaire", "table des matières", "contenu"],
                                    'de': ["inhalt", "inhaltsverzeichnis"],
                                    'it': ["indice", "sommario"],
                                    'pt': ["índice", "sumário", "conteúdo"],
                                    'zh': ["目录", "内容"],
                                    'ja': ["目次", "内容"],
                                    'ar': ["المحتويات", "الفهرس"],
                                    'ru': ["содержание", "оглавление"],
                                    'hi': ["विषय सूची", "सामग्री"]
                                }
                                
                                current_toc_patterns = toc_patterns.get(detected_lang, toc_patterns['en'])
                                if any(pattern in text_lower for pattern in current_toc_patterns):
                                    special_heading_on_page = heading_dict
                            
                            page_headings.append(heading_dict)
                
                # Your special heading logic (unchanged)
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
            
            # Your filtering logic (unchanged)
            min_count = max(2, int(num_pages * 0.5) + 1)
            filtered_headings = [h for h in headings if heading_counter[h["text"].lower().strip()] < min_count]
            
            # ✅ FIXED: Strict compliance with Challenge 1A output format
            # Only return title and outline - no extra fields
            return {
                "title": title,
                "outline": filtered_headings
            }
            
        except Exception as e:
            logger.error(f"Error extracting outline from {pdf_path}: {e}")
            return {"title": "", "outline": []}

    def process_pdf(self, pdf_path):
        """Your main processing function with error handling"""
        try:
            return self.extract_outline(pdf_path)
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return {"title": "", "outline": []}

def main():
    """Enhanced main function with multilingual support and your path logic"""
    start_time = time.time()
    
    # Your Docker paths logic (unchanged)
    INPUT_DIR = Path("/app/input")
    OUTPUT_DIR = Path("/app/output")
    
    # Your fallback for local testing (updated path)
    if not INPUT_DIR.exists():
        INPUT_DIR = Path("Challenge_1a/sample_dataset/pdfs")
        OUTPUT_DIR = Path("outputs")  # Simplified for cross-platform compatibility
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting multilingual PDF processing...")
    
    # Initialize multilingual extractor
    try:
        extractor = MultilingualPDFOutlineExtractor()
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {e}")
        return 1
    
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    
    print(f"Processing {len(pdf_files)} PDF files...")
    
    success_count = 0
    language_stats = Counter()
    
    for pdf_file in pdf_files:
        try:
            print(f"Processing: {pdf_file.name}")
            process_start = time.time()
            
            result = extractor.process_pdf(pdf_file)
            
            output_path = OUTPUT_DIR / f"{pdf_file.stem}.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            
            process_time = time.time() - process_start
            
            print(f"  Title: '{result['title']}'")
            print(f"  Headings: {len(result['outline'])}")
            print(f"  Processed in: {process_time:.2f}s")
            
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
            logger.error(f"Error processing {pdf_file.name}: {e}")
    
    total_time = time.time() - start_time
    print(f"\nMultilingual processing complete!")
    print(f"Successfully processed: {success_count}/{len(pdf_files)} files")
    print(f"Total time: {total_time:.2f}s")

if __name__ == "__main__":
    main()
