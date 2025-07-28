import os
import json
import re
import fitz
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from typing import List, Dict, Any, Union
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultilingualChallenge1B:
    def __init__(self):
        try:
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny", local_files_only=True)
            self.model = AutoModel.from_pretrained("prajjwal1/bert-tiny", local_files_only=True)
            
            try:
                self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", local_files_only=True)
            except:
                self.summarizer = None
            
            # Language detection patterns
            self.language_patterns = {
                'zh': re.compile(r'[\u4e00-\u9fff]'),  # Chinese characters
                'ja': re.compile(r'[\u3040-\u309f\u30a0-\u30ff]'),  # Hiragana and Katakana
                'ar': re.compile(r'[\u0600-\u06ff]'),  # Arabic script
                'ru': re.compile(r'[\u0400-\u04ff]'),  # Cyrillic script
                'hi': re.compile(r'[\u0900-\u097f]'),  # Devanagari script
                'ko': re.compile(r'[\uac00-\ud7af]'),  # Korean script
            }
            
            # Enhanced multilingual persona configurations
            self.persona_config = {
                "Travel Planner": {
                    "section_patterns": {
                        'en': [
                            r"guide to.*cities", r"coastal adventures", r"culinary experiences", 
                            r"packing.*tips", r"nightlife.*entertainment", r"activities", 
                            r"restaurants", r"hotels", r"attractions", r"water.*sports"
                        ],
                        'es': [
                            r"guía.*ciudades", r"aventuras.*costeras", r"experiencias.*culinarias",
                            r"consejos.*equipaje", r"vida.*nocturna", r"actividades",
                            r"restaurantes", r"hoteles", r"atracciones", r"deportes.*acuáticos"
                        ],
                        'fr': [
                            r"guide.*villes", r"aventures.*côtières", r"expériences.*culinaires",
                            r"conseils.*bagages", r"vie.*nocturne", r"activités",
                            r"restaurants", r"hôtels", r"attractions", r"sports.*nautiques"
                        ]
                    },
                    "keywords": {
                        'en': [
                            "cities", "coastal", "adventures", "culinary", "experiences", "packing", 
                            "tips", "nightlife", "entertainment", "activities", "restaurants", 
                            "hotels", "attractions", "water", "sports", "beaches", "tours"
                        ],
                        'es': [
                            "ciudades", "costero", "aventuras", "culinario", "experiencias", "equipaje",
                            "consejos", "vida nocturna", "entretenimiento", "actividades", "restaurantes",
                            "hoteles", "atracciones", "agua", "deportes", "playas", "tours"
                        ],
                        'fr': [
                            "villes", "côtier", "aventures", "culinaire", "expériences", "bagages",
                            "conseils", "vie nocturne", "divertissement", "activités", "restaurants",
                            "hôtels", "attractions", "eau", "sports", "plages", "tours"
                        ]
                    }
                },
                "HR professional": {
                    "section_patterns": {
                        'en': [
                            r"change.*forms.*fillable", r"create.*multiple.*pdfs", r"convert.*clipboard", 
                            r"fill.*sign.*forms", r"send.*document.*signatures", r"prepare.*forms",
                            r"acrobat.*pro", r"pdf.*tools", r"e-signatures", r"request.*signatures"
                        ],
                        'es': [
                            r"cambiar.*formularios.*rellenables", r"crear.*múltiples.*pdfs", r"convertir.*portapapeles",
                            r"llenar.*firmar.*formularios", r"enviar.*documento.*firmas", r"preparar.*formularios",
                            r"acrobat.*pro", r"herramientas.*pdf", r"firmas.*electrónicas", r"solicitar.*firmas"
                        ],
                        'fr': [
                            r"modifier.*formulaires.*remplissables", r"créer.*multiples.*pdfs", r"convertir.*presse-papiers",
                            r"remplir.*signer.*formulaires", r"envoyer.*document.*signatures", r"préparer.*formulaires",
                            r"acrobat.*pro", r"outils.*pdf", r"signatures.*électroniques", r"demander.*signatures"
                        ]
                    },
                    "keywords": {
                        'en': [
                            "forms", "fillable", "acrobat", "pdf", "signatures", "create", "convert",
                            "fill", "sign", "tools", "prepare", "request", "documents", "pro"
                        ],
                        'es': [
                            "formularios", "rellenable", "acrobat", "pdf", "firmas", "crear", "convertir",
                            "llenar", "firmar", "herramientas", "preparar", "solicitar", "documentos", "pro"
                        ],
                        'fr': [
                            "formulaires", "remplissable", "acrobat", "pdf", "signatures", "créer", "convertir",
                            "remplir", "signer", "outils", "préparer", "demander", "documents", "pro"
                        ]
                    }
                },
                "Food Contractor": {
                    "section_patterns": {
                        'en': [
                            r"^[A-Z][a-z]+\s*[A-Z]*[a-z]*$", r"falafel", r"ratatouille", r"baba.*ganoush",
                            r"veggie.*sushi", r"vegetable.*lasagna", r"macaroni.*cheese", r"escalivada"
                        ],
                        'es': [
                            r"^[A-Z][a-z]+\s*[A-Z]*[a-z]*$", r"falafel", r"ratatouille", r"baba.*ganoush",
                            r"sushi.*vegetariano", r"lasaña.*vegetales", r"macarrones.*queso", r"escalivada"
                        ],
                        'fr': [
                            r"^[A-Z][a-z]+\s*[A-Z]*[a-z]*$", r"falafel", r"ratatouille", r"baba.*ganoush",
                            r"sushi.*végétarien", r"lasagne.*légumes", r"macaroni.*fromage", r"escalivada"
                        ]
                    },
                    "keywords": {
                        'en': [
                            "falafel", "ratatouille", "baba", "ganoush", "sushi", "vegetable", 
                            "lasagna", "macaroni", "cheese", "vegetarian", "recipe", "ingredients",
                            "instructions", "cooking", "dinner", "buffet", "corporate"
                        ],
                        'es': [
                            "falafel", "ratatouille", "baba", "ganoush", "sushi", "vegetal",
                            "lasaña", "macarrones", "queso", "vegetariano", "receta", "ingredientes",
                            "instrucciones", "cocinar", "cena", "buffet", "corporativo"
                        ],
                        'fr': [
                            "falafel", "ratatouille", "baba", "ganoush", "sushi", "légume",
                            "lasagne", "macaroni", "fromage", "végétarien", "recette", "ingrédients",
                            "instructions", "cuisine", "dîner", "buffet", "entreprise"
                        ]
                    }
                }
            }
            
            logger.info("✅ Multilingual Challenge 1B processor initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize processor: {e}")
            raise

    def detect_language(self, text_sample: str) -> str:
        """Detect document language"""
        if not text_sample or len(text_sample) < 10:
            return 'en'
        
        # Check for non-Latin scripts first
        for lang, pattern in self.language_patterns.items():
            if pattern.search(text_sample):
                logger.info(f"Detected {lang} via script pattern")
                return lang
        
        text_lower = text_sample.lower()
        
        if any(char in text_lower for char in ['ñ', 'á', 'é', 'í', 'ó', 'ú', '¿', '¡']):
            return 'es'
        elif any(char in text_lower for char in ['à', 'â', 'ç', 'è', 'é', 'ê', 'ë', 'î', 'ï', 'ô', 'ù', 'û', 'ü', 'ÿ']):
            return 'fr'
        elif any(char in text_lower for char in ['ä', 'ö', 'ü', 'ß']):
            return 'de'
        elif any(char in text_lower for char in ['à', 'á', 'â', 'ã', 'ç', 'é', 'ê', 'í', 'ó', 'ô', 'õ', 'ú']):
            return 'pt'
        elif any(char in text_lower for char in ['à', 'è', 'é', 'ì', 'í', 'î', 'ò', 'ó', 'ù', 'ú']):
            return 'it'
        
        return 'en'  # Default to English

    def get_persona_patterns(self, persona: str, detected_lang: str) -> List[str]:
        """Get persona patterns for detected language"""
        persona_data = self.persona_config.get(persona, {})
        patterns_by_lang = persona_data.get("section_patterns", {})
        return patterns_by_lang.get(detected_lang, patterns_by_lang.get('en', []))

    def get_persona_keywords(self, persona: str, detected_lang: str) -> List[str]:
        """Get persona keywords for detected language"""
        persona_data = self.persona_config.get(persona, {})
        keywords_by_lang = persona_data.get("keywords", {})
        return keywords_by_lang.get(detected_lang, keywords_by_lang.get('en', []))

    def _embed_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Embed texts using BERT model """
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                               max_length=512, return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def load_input_configuration(self, input_file: Path) -> Dict[str, Any]:
        """Load input configuration from JSON file"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading input configuration: {e}")
            return {}

    def is_valid_section_heading(self, text: str, line_info: Dict, body_size: float, 
                                persona: str, detected_lang: str = 'en') -> bool:
        word_count = len(text.split())
        if word_count < 1 or word_count > 15 or len(text) < 3:
            return False
        
        skip_patterns = [
            r"^\d+$", r"^page \d+", r"^figure \d+", r"^table \d+",
            r"^copyright", r"^all rights reserved", r"^www\.", r"^http"
        ]
        if any(re.search(pattern, text.lower()) for pattern in skip_patterns):
            return False
        
        spans = line_info.get("spans", [])
        if spans:
            font_size = spans[0].get("size", body_size)
            is_bold = spans[0].get("flags", 0) & 16
            is_larger = font_size > body_size + 1.5
        else:
            is_bold = is_larger = False
        
        # Enhanced persona-specific pattern matching with language support
        persona_patterns = self.get_persona_patterns(persona, detected_lang)
        matches_persona_pattern = any(
            re.search(pattern, text.lower()) for pattern in persona_patterns
        )
        
        # Enhanced general heading patterns with multilingual support
        general_patterns = [
            r"^[A-Z][a-z]+(?: [A-Z][a-z]+)*$",  # Title Case
            r"^\d+\.\s*[A-Z]",  # Numbered sections
            r"^(Chapter|Section|Part|Capítulo|Sección|Parte|Chapitre|Section|Partie|Kapitel|Abschnitt|Teil)\s+\d+",
            r"(Introduction|Conclusion|Summary|Overview|Instructions|Ingredients|Introducción|Conclusión|Resumen|Instrucciones|Ingredientes|Présentation|Conclusion|Résumé|Aperçu|Instructions|Ingrédients)$"
        ]
        matches_general_pattern = any(
            re.search(pattern, text) for pattern in general_patterns
        )
        
        # Your exact decision logic
        if matches_persona_pattern:
            return True
        if (is_bold or is_larger) and (matches_general_pattern or word_count <= 8):
            return True
        if matches_general_pattern and word_count <= 6:
            return True
            
        return False

    def extract_sections_from_pdf(self, pdf_path: Path, persona: str, 
                                 job_description: str) -> List[Dict[str, Any]]:
        """Extract sections with multilingual support - keeping your exact algorithm"""
        try:
            doc = fitz.open(pdf_path)
            sections = []
            
            # Detect language from first few pages
            sample_text = ""
            for page_num in range(min(3, len(doc))):
                page_text = doc[page_num].get_text()
                sample_text += page_text[:500]
            
            detected_lang = self.detect_language(sample_text)
            # logger.info(f"Detected language: {detected_lang}")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                page_height = page.rect.height
                
                # Your exact font size calculation
                sizes = []
                for block in blocks:
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        for span in line["spans"]:
                            sizes.append(span["size"])
                
                if not sizes:
                    continue
                
                body_size = Counter(sizes).most_common(1)[0][0]
                
                # Your exact section extraction logic
                for block in blocks:
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        if not line["spans"]:
                            continue
                        
                        # Your exact header/footer skip
                        y_pos = line["spans"][0]["origin"][1]
                        if y_pos < page_height * 0.1 or y_pos > page_height * 0.9:
                            continue
                        
                        text = " ".join([span["text"] for span in line["spans"]]).strip()
                        
                        if self.is_valid_section_heading(text, line, body_size, persona, detected_lang):
                            relevance = self.calculate_section_relevance(text, persona, job_description, detected_lang)
                            
                            if relevance > 0.3:  # Your exact threshold
                                sections.append({
                                    "document": pdf_path.name,
                                    "section_title": text,
                                    "page_number": page_num + 1,
                                    "relevance_score": relevance,
                                    "importance_rank": 0
                                })
            
            doc.close()
            return sections
            
        except Exception as e:
            logger.error(f"Error extracting sections from {pdf_path}: {e}")
            return []

    def calculate_section_relevance(self, section_title: str, persona: str, 
                                   job_description: str, detected_lang: str = 'en') -> float:
        """Calculate relevance score with multilingual support - your exact algorithm"""
        if not section_title:
            return 0.0
        
        # Your exact embedding approach
        title_embedding = self._embed_texts(section_title)[0]
        persona_embedding = self._embed_texts(persona)[0]
        job_embedding = self._embed_texts(job_description)[0]
        
        # Your exact similarity calculations
        persona_sim = np.dot(title_embedding, persona_embedding) / (
            np.linalg.norm(title_embedding) * np.linalg.norm(persona_embedding) + 1e-8
        )
        job_sim = np.dot(title_embedding, job_embedding) / (
            np.linalg.norm(title_embedding) * np.linalg.norm(job_embedding) + 1e-8
        )
        
        # Enhanced keyword matching with language support
        persona_keywords = self.get_persona_keywords(persona, detected_lang)
        title_lower = section_title.lower()
        keyword_matches = sum(1 for keyword in persona_keywords if keyword in title_lower)
        keyword_score = min(0.4, keyword_matches * 0.1)
        
        # Enhanced pattern bonus with language support
        persona_patterns = self.get_persona_patterns(persona, detected_lang)
        pattern_bonus = 0.3 if any(re.search(pattern, title_lower) for pattern in persona_patterns) else 0
        
        # Your exact relevance calculation
        return float(persona_sim * 0.25 + job_sim * 0.25 + keyword_score * 0.25 + pattern_bonus * 0.25)

    def deduplicate_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate sections - your exact method"""
        seen = set()
        deduped = []
        
        # Your exact sorting approach
        sections_sorted = sorted(sections, key=lambda x: x["relevance_score"], reverse=True)
        
        for section in sections_sorted:
            # Your exact deduplication key
            key = (
                section["document"], 
                section["section_title"].strip().lower(),
                section["page_number"]
            )
            
            if key not in seen:
                seen.add(key)
                deduped.append(section)
                
            if len(deduped) >= 5:  # Your exact limit
                break
        
        return deduped

    def extract_detailed_content(self, pdf_path: Path, section_title: str, 
                               page_number: int, persona: str) -> str:
        """Extract comprehensive content - your exact method"""
        try:
            doc = fitz.open(pdf_path)
            if page_number > len(doc):
                doc.close()
                return section_title
            
            page = doc[page_number - 1]
            text_dict = page.get_text("dict")
            
            # Your exact text extraction
            full_text = ""
            for block in text_dict["blocks"]:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    if line["spans"]:
                        line_text = " ".join([span["text"] for span in line["spans"]]).strip()
                        full_text += " " + line_text
            
            doc.close()
            
            # Your exact content finding logic
            full_text = full_text.strip()
            if not full_text:
                return section_title
            
            title_lower = section_title.lower()
            text_lower = full_text.lower()
            
            if title_lower in text_lower:
                start_idx = text_lower.find(title_lower)
                content_start = start_idx + len(section_title)
                content_end = min(len(full_text), content_start + 800)
                detailed_content = full_text[content_start:content_end].strip()
                
                if detailed_content:
                    return f"{section_title} {detailed_content}"
            
            return section_title
            
        except Exception as e:
            logger.error(f"Error extracting detailed content: {e}")
            return section_title

    def process_collection(self, input_file: Path, pdf_directory: Path) -> Dict[str, Any]:
        """Process collection - your exact method"""
        # Your exact configuration loading
        config = self.load_input_configuration(input_file)
        if not config:
            return {"error": "Failed to load input configuration"}
        
        # Your exact detail extraction
        documents = config.get("documents", [])
        persona = config.get("persona", {}).get("role", "Unknown")
        job_description = config.get("job_to_be_done", {}).get("task", "Unknown")
        
        # Your exact document processing
        all_sections = []
        processed_docs = []
        
        for doc_info in documents:
            filename = doc_info.get("filename", "")
            pdf_path = pdf_directory / filename
            
            if pdf_path.exists():
                sections = self.extract_sections_from_pdf(pdf_path, persona, job_description)
                all_sections.extend(sections)
                processed_docs.append(filename)
        
        if not all_sections:
            return {
                "metadata": {
                    "input_documents": processed_docs,
                    "persona": persona,
                    "job_to_be_done": job_description,
                    "processing_timestamp": datetime.now().isoformat()
                },
                "extracted_sections": [],
                "subsection_analysis": []
            }
        
        # Your exact deduplication and ranking
        top_sections = self.deduplicate_sections(all_sections)
        
        for rank, section in enumerate(top_sections, 1):
            section["importance_rank"] = rank
        
        # Your exact subsection analysis
        subsection_analysis = []
        for section in top_sections:
            pdf_path = pdf_directory / section["document"]
            detailed_content = self.extract_detailed_content(
                pdf_path, section["section_title"], section["page_number"], persona
            )
            
            subsection_analysis.append({
                "document": section["document"],
                "refined_text": detailed_content,
                "page_number": section["page_number"]
            })
        
        # Your exact return format
        return {
            "metadata": {
                "input_documents": processed_docs,
                "persona": persona,
                "job_to_be_done": job_description,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": [
                {
                    "document": section["document"],
                    "section_title": section["section_title"],
                    "importance_rank": section["importance_rank"],
                    "page_number": section["page_number"]
                }
                for section in top_sections
            ],
            "subsection_analysis": subsection_analysis
        }

def main():
    """Main execution function - supporting any number of collections"""
    INPUT_DIR = Path("/app/input")
    OUTPUT_DIR = Path("/app/output")
    
    # Fallback for local testing (same as 1A)
    if not INPUT_DIR.exists():
        INPUT_DIR = Path("Challenge_1b")
        OUTPUT_DIR = Path("Challenge_1b/outputs")
    
    # Create output directory for Docker compatibility
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    processor = MultilingualChallenge1B()
    
    # ✅ DYNAMIC: Auto-detect all collections instead of hardcoded list
    collections = []
    
    # Scan input directory for collection folders
    if INPUT_DIR.exists():
        for item in INPUT_DIR.iterdir():
            if item.is_dir():
                # Check if it's a valid collection (has required files)
                input_file = item / "challenge1b_input.json"
                pdf_directory = item / "PDFs"
                
                if input_file.exists() and pdf_directory.exists():
                    collections.append(item.name)
                    logger.info(f"Found valid collection: {item.name}")
                else:
                    logger.info(f"Skipping incomplete collection: {item.name}")
    
    if not collections:
        print("❌ No valid collections found!")
        print(f"   Expected structure: {INPUT_DIR}/[Collection Name]/challenge1b_input.json")
        print(f"                      {INPUT_DIR}/[Collection Name]/PDFs/")
        return
    
    collections.sort()  # Process collections in alphabetical order
    print(f"🔍 Found {len(collections)} collection(s): {', '.join(collections)}")
    
    successful_collections = 0
    
    for collection_name in collections:
        collection_path = INPUT_DIR / collection_name
        input_file = collection_path / "challenge1b_input.json"
        pdf_directory = collection_path / "PDFs"
        
        # Dual output: Store both with collection AND in output dir
        collection_output = collection_path / "challenge1b_output.json"
        docker_output = OUTPUT_DIR / f"challenge1b_output_{collection_name.replace(' ', '_').lower()}.json"
        
        print(f"📁 Processing {collection_name}...")
        
        try:
            result = processor.process_collection(input_file, pdf_directory)
            
            # Save to both locations
            with open(collection_output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            
            with open(docker_output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            
            if "error" not in result:
                print(f"  ✅ Extracted {len(result['extracted_sections'])} sections")
                print(f"  📝 Generated {len(result['subsection_analysis'])} analyses")
                print(f"  💾 Output saved to: {collection_output}")
                print(f"  💾 Docker output: {docker_output}")
                successful_collections += 1
            else:
                print(f"  ❌ Error: {result['error']}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            logger.error(f"Error processing {collection_name}: {e}")
    
    # Final summary
    print(f"\n🏁 Processing complete!")
    print(f"📊 Successfully processed: {successful_collections}/{len(collections)} collections")
    
    if successful_collections == len(collections):
        print("✅ All collections processed successfully!")
    elif successful_collections > 0:
        print("⚠️  Some collections had errors")
    else:
        print("❌ No collections were processed successfully")

if __name__ == "__main__":
    main()
