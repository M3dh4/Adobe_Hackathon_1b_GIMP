import os
import json
import re
import fitz
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersonaBasedPDFSummarizer:
    def __init__(self):
        """Initialize the persona-based PDF summarizer"""
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        self.torch = torch
        
        # Initialize models for text understanding and summarization
        self.tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        self.model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        
        # Initialize summarization pipeline
        try:
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                tokenizer="facebook/bart-large-cnn"
            )
        except:
            # Fallback to a smaller model if BART is not available
            self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        
        # Define persona templates and their associated keywords
        self.personas = {
            "student": {
                "keywords": [
                    "learning", "education", "study", "assignment", "project", "course", 
                    "curriculum", "objective", "skill", "knowledge", "understanding",
                    "example", "exercise", "practice", "tutorial", "guide", "basic",
                    "fundamental", "concept", "theory", "application", "homework"
                ],
                "focus": "educational content, learning objectives, practical examples",
                "summary_style": "Clear explanations with examples and key concepts highlighted"
            },
            "researcher": {
                "keywords": [
                    "methodology", "analysis", "data", "result", "conclusion", "research",
                    "study", "experiment", "hypothesis", "finding", "evidence", "statistical",
                    "survey", "investigation", "literature", "reference", "citation",
                    "framework", "model", "algorithm", "evaluation", "validation"
                ],
                "focus": "research methods, findings, data analysis, conclusions",
                "summary_style": "Technical detail with emphasis on methodology and results"
            },
            "business_analyst": {
                "keywords": [
                    "business", "strategy", "market", "analysis", "revenue", "profit",
                    "cost", "investment", "roi", "performance", "metric", "kpi",
                    "process", "efficiency", "optimization", "requirement", "stakeholder",
                    "risk", "opportunity", "competitive", "growth", "value"
                ],
                "focus": "business impact, metrics, strategic insights, recommendations",
                "summary_style": "Executive summary with key metrics and business implications"
            },
            "technical_developer": {
                "keywords": [
                    "implementation", "code", "system", "architecture", "design",
                    "algorithm", "database", "api", "framework", "library", "tool",
                    "development", "programming", "software", "technical", "specification",
                    "integration", "deployment", "testing", "debugging", "performance"
                ],
                "focus": "technical implementation, system design, development processes",
                "summary_style": "Technical specifications with implementation details"
            },
            "project_manager": {
                "keywords": [
                    "timeline", "milestone", "deliverable", "resource", "budget",
                    "schedule", "team", "coordination", "communication", "status",
                    "progress", "risk", "dependency", "stakeholder", "requirement",
                    "scope", "planning", "execution", "monitoring", "control"
                ],
                "focus": "project timelines, deliverables, resource allocation, risks",
                "summary_style": "Project-focused with timelines, deliverables, and action items"
            }
        }
        
        # Precompute persona embeddings
        self.persona_embeddings = {}
        for persona, data in self.personas.items():
            embeddings = self._embed_texts(data["keywords"])
            self.persona_embeddings[persona] = np.mean(embeddings, axis=0)
        
        # Content section templates for better extraction
        self.section_templates = [
            "introduction", "summary", "conclusion", "methodology", "results",
            "discussion", "background", "objectives", "requirements", "approach",
            "implementation", "evaluation", "recommendations", "future work"
        ]
        self.section_embeddings = self._embed_texts(self.section_templates)

    def _embed_texts(self, texts):
        """Embed texts using BERT model"""
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                               max_length=512, return_tensors="pt")
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def calculate_persona_relevance(self, text: str, persona: str) -> float:
        """Calculate how relevant text is to a specific persona"""
        if not text or persona not in self.personas:
            return 0.0
        
        # Embed the text
        text_embedding = self._embed_texts(text)[0]
        persona_embedding = self.persona_embeddings[persona]
        
        # Calculate cosine similarity
        similarity = np.dot(text_embedding, persona_embedding) / (
            np.linalg.norm(text_embedding) * np.linalg.norm(persona_embedding) + 1e-8
        )
        
        # Keyword matching bonus
        persona_keywords = self.personas[persona]["keywords"]
        text_lower = text.lower()
        keyword_matches = sum(1 for keyword in persona_keywords if keyword in text_lower)
        keyword_bonus = min(0.3, keyword_matches * 0.05)  # Up to 30% bonus
        
        return float(similarity) + keyword_bonus

    def extract_pdf_content(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract structured content from a PDF"""
        try:
            doc = fitz.open(pdf_path)
            content = {
                "title": "",
                "sections": [],
                "full_text": "",
                "metadata": {
                    "pages": len(doc),
                    "file_name": pdf_path.name
                }
            }
            
            # Extract title from first page
            if len(doc) > 0:
                first_page = doc[0]
                blocks = first_page.get_text("dict")['blocks']
                sizes = [span["size"] for b in blocks if "lines" in b 
                        for line in b["lines"] for span in line["spans"]]
                if sizes:
                    max_size = max(sizes)
                    title_parts = []
                    for b in blocks:
                        if "lines" not in b:
                            continue
                        for line in b["lines"]:
                            for span in line["spans"]:
                                if span["size"] == max_size:
                                    title_parts.append(span["text"].strip())
                    content["title"] = " ".join(title_parts).strip() or pdf_path.stem
                else:
                    content["title"] = pdf_path.stem
            
            # Extract content by sections
            current_section = {"heading": "Introduction", "content": "", "page": 1}
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" not in block:
                        continue
                    
                    for line in block["lines"]:
                        if not line["spans"]:
                            continue
                        
                        text = " ".join([s["text"] for s in line["spans"]]).strip()
                        if not text:
                            continue
                        
                        # Check if this might be a section heading
                        if self._is_section_heading(line, text):
                            # Save previous section if it has content
                            if current_section["content"].strip():
                                content["sections"].append(current_section)
                            
                            # Start new section
                            current_section = {
                                "heading": text,
                                "content": "",
                                "page": page_num + 1
                            }
                        else:
                            # Add to current section content
                            current_section["content"] += " " + text
            
            # Add the last section
            if current_section["content"].strip():
                content["sections"].append(current_section)
            
            # Create full text
            content["full_text"] = " ".join([section["content"] for section in content["sections"]])
            
            doc.close()
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content from {pdf_path}: {e}")
            return {
                "title": pdf_path.stem,
                "sections": [],
                "full_text": "",
                "metadata": {"pages": 0, "file_name": pdf_path.name, "error": str(e)}
            }

    def _is_section_heading(self, line, text):
        """Determine if a line is likely a section heading"""
        if len(text.split()) > 10:  # Too long to be a heading
            return False
        
        spans = line["spans"]
        if not spans:
            return False
        
        # Check for formatting indicators (bold, larger font)
        first_span = spans[0]
        is_bold = first_span.get("flags", 0) & 16  # Bold flag
        
        # Check against section templates
        text_embedding = self._embed_texts(text)[0]
        similarities = np.dot(self.section_embeddings, text_embedding) / (
            np.linalg.norm(self.section_embeddings, axis=1) * 
            np.linalg.norm(text_embedding) + 1e-8
        )
        
        has_section_similarity = np.max(similarities) > 0.6
        
        return is_bold or has_section_similarity

    def extract_persona_relevant_content(self, documents: List[Dict], persona: str, 
                                       max_sections: int = 10) -> List[Dict]:
        """Extract most relevant content sections for a specific persona"""
        relevant_sections = []
        
        for doc in documents:
            for section in doc["sections"]:
                relevance_score = self.calculate_persona_relevance(
                    section["content"], persona
                )
                
                if relevance_score > 0.3:  # Minimum relevance threshold
                    relevant_sections.append({
                        "document": doc["title"],
                        "file_name": doc["metadata"]["file_name"],
                        "section_heading": section["heading"],
                        "content": section["content"],
                        "page": section["page"],
                        "relevance_score": relevance_score
                    })
        
        # Sort by relevance and return top sections
        relevant_sections.sort(key=lambda x: x["relevance_score"], reverse=True)
        return relevant_sections[:max_sections]

    def generate_persona_summary(self, relevant_content: List[Dict], persona: str) -> Dict:
        """Generate a persona-specific summary from relevant content"""
        if not relevant_content:
            return {
                "persona": persona,
                "summary": "No relevant content found for this persona.",
                "key_points": [],
                "documents_used": [],
                "total_sections": 0
            }
        
        # Combine content for summarization
        combined_text = ""
        documents_used = set()
        key_points = []
        
        for item in relevant_content:
            combined_text += f"{item['section_heading']}: {item['content']}\n\n"
            documents_used.add(item["file_name"])
            
            # Extract key points (first sentence of each section)
            sentences = item["content"].split(". ")
            if sentences:
                key_points.append(f"• {sentences[0].strip()}")
        
        # Generate summary based on persona
        summary_text = self._generate_summary_text(combined_text, persona)
        
        return {
            "persona": persona,
            "summary": summary_text,
            "key_points": key_points[:10],  # Top 10 key points
            "documents_used": list(documents_used),
            "total_sections": len(relevant_content),
            "relevance_scores": [item["relevance_score"] for item in relevant_content]
        }

    def _generate_summary_text(self, text: str, persona: str) -> str:
        """Generate summary text using the summarization model"""
        if not text.strip():
            return "No content available for summarization."
        
        try:
            # Truncate text if too long for the model
            max_length = 1024  # Adjust based on model capacity
            if len(text) > max_length:
                text = text[:max_length] + "..."
            
            # Generate summary with persona-specific parameters
            persona_config = self.personas.get(persona, {})
            
            summary_result = self.summarizer(
                text,
                max_length=150,
                min_length=50,
                do_sample=False,
                num_beams=4
            )
            
            summary = summary_result[0]["summary_text"]
            
            # Add persona-specific context
            style_note = persona_config.get("summary_style", "")
            if style_note:
                summary = f"[{style_note}]\n\n{summary}"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Summary generation failed: {str(e)}"

    def process_multiple_pdfs(self, pdf_paths: List[Path], persona: str) -> Dict:
        """Process multiple PDFs for a specific persona"""
        logger.info(f"Processing {len(pdf_paths)} PDFs for persona: {persona}")
        
        # Extract content from all PDFs
        documents = []
        for pdf_path in pdf_paths:
            logger.info(f"Extracting content from: {pdf_path.name}")
            content = self.extract_pdf_content(pdf_path)
            if content["sections"]:  # Only add if we extracted sections
                documents.append(content)
        
        if not documents:
            return {
                "error": "No content could be extracted from the provided PDFs",
                "persona": persona,
                "processed_files": 0
            }
        
        # Extract persona-relevant content
        relevant_content = self.extract_persona_relevant_content(documents, persona)
        
        # Generate persona-specific summary
        summary_result = self.generate_persona_summary(relevant_content, persona)
        
        # Add metadata
        summary_result.update({
            "processed_files": len(documents),
            "total_documents": len(pdf_paths),
            "persona_definition": self.personas.get(persona, {}),
            "processing_timestamp": str(Path.cwd())  # Placeholder for timestamp
        })
        
        return summary_result


def main():
    """Main execution function"""
    # Use Docker paths as specified in the challenge requirements
    INPUT_DIR = Path("/app/input")
    OUTPUT_DIR = Path("/app/output")
    
    # Fallback for local testing
    if not INPUT_DIR.exists():
        INPUT_DIR = Path("Challenge_1b/Collection 1/pdfs")
        OUTPUT_DIR = Path("C:/Users/garvt/OneDrive/Desktop/Adobe-India-Hackathon25/OUTPUTS_M1B")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize the summarizer
    summarizer = PersonaBasedPDFSummarizer()
    
    # Get all PDF files
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in the input directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF files for processing...")
    
    # Define personas to process (you can modify this based on requirements)
    personas_to_process = ["student", "researcher", "business_analyst", "technical_developer", "project_manager"]
    
    # Process for each persona
    for persona in personas_to_process:
        print(f"\n🎭 Processing for persona: {persona.upper()}")
        
        try:
            # Generate persona-specific summary
            result = summarizer.process_multiple_pdfs(pdf_files, persona)
            
            # Save results
            output_file = OUTPUT_DIR / f"{persona}_summary.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            
            # Print summary info
            if "error" not in result:
                print(f"  ✅ Generated summary from {result['total_sections']} relevant sections")
                print(f"  📄 Used {result['processed_files']} documents")
                print(f"  💾 Saved to: {output_file.name}")
                
                # Print a brief preview of the summary
                summary_preview = result["summary"][:200] + "..." if len(result["summary"]) > 200 else result["summary"]
                print(f"  📝 Summary preview: {summary_preview}")
            else:
                print(f"  ❌ Error: {result['error']}")
                
        except Exception as e:
            print(f"  ❌ Error processing persona {persona}: {e}")
            logger.error(f"Error processing persona {persona}: {e}")
    
    # Generate a combined report
    print(f"\n📊 Generating combined report...")
    combined_results = {}
    
    for persona in personas_to_process:
        result_file = OUTPUT_DIR / f"{persona}_summary.json"
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                combined_results[persona] = json.load(f)
    
    # Save combined results
    combined_output = OUTPUT_DIR / "all_personas_summary.json"
    with open(combined_output, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Processing complete! Results saved to: {OUTPUT_DIR}")
    print(f"📋 Combined report: {combined_output.name}")


if __name__ == "__main__":
    main()
