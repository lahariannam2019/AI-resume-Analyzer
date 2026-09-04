"""
PDF Resume Parser using PyMuPDF (fitz) with section detection and text cleaning.
"""

import fitz  # PyMuPDF
import re
from typing import Dict, List, Tuple, Any
from backend.models import ResumeParseResult, SectionDetail

STANDARD_SECTIONS = [
    "Summary",
    "Education",
    "Skills",
    "Experience",
    "Projects",
    "Certifications",
    "Achievements"
]

SECTION_PATTERNS = {
    "Summary": [
        r"^(executive\s+)?summary", r"^professional\s+summary", r"^profile", r"^about\s+me", r"^objective"
    ],
    "Education": [
        r"^education", r"^academic\s+background", r"^academic\ revival", r"^qualifications", r"^education\s+&\s+credentials"
    ],
    "Skills": [
        r"^skills", r"^technical\s+skills", r"^core\s+competencies", r"^technologies", r"^skills\s+&\s+tools", r"^expertise"
    ],
    "Experience": [
        r"^experience", r"^work\s+experience", r"^employment\s+history", r"^professional\s+experience", r"^work\s+history", r"^internships"
    ],
    "Projects": [
        r"^projects", r"^key\s+projects", r"^academic\s+projects", r"^personal\s+projects", r"^portfolio"
    ],
    "Certifications": [
        r"^certifications", r"^licenses", r"^certificates", r"^courses\s+&\s+certifications"
    ],
    "Achievements": [
        r"^achievements", r"^awards", r"^honors", r"^publications", r"^key\s+achievements"
    ]
}


def clean_text(raw_text: str) -> str:
    """Clean extra spaces, strange symbols, and normalize line breaks."""
    if not raw_text:
        return ""
    # Normalize unicode whitespace & newlines
    text = re.sub(r'[\r\t\f\v]', ' ', raw_text)
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove excessive blank lines
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Tuple[str, int]:
    """Extract full raw text from PDF byte content."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    text_chunks = []
    
    for page in doc:
        page_text = page.get_text("text")
        if page_text:
            text_chunks.append(page_text)
            
    full_text = "\n".join(text_chunks)
    doc.close()
    return clean_text(full_text), total_pages


def extract_text_from_pdf_path(filepath: str) -> Tuple[str, int]:
    """Extract full raw text from PDF filepath."""
    with open(filepath, "rb") as f:
        return extract_text_from_pdf_bytes(f.read())


def identify_sections(full_text: str) -> Dict[str, str]:
    """Segment resume text into identified standard sections."""
    lines = full_text.split('\n')
    sections: Dict[str, List[str]] = {sec: [] for sec in STANDARD_SECTIONS}
    sections["General"] = []
    
    current_section = "General"
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        # Check if line matches a known section header
        found_section = None
        # Header candidate usually short (< 40 chars)
        if len(stripped) < 45:
            clean_line = re.sub(r'[^\w\s]', '', stripped.lower()).strip()
            for sec_name, patterns in SECTION_PATTERNS.items():
                for pat in patterns:
                    if re.search(pat, clean_line):
                        found_section = sec_name
                        break
                if found_section:
                    break
        
        if found_section:
            current_section = found_section
        else:
            sections[current_section].append(stripped)
            
    # Combine lines per section
    result = {}
    for sec, text_lines in sections.items():
        combined = "\n".join(text_lines).strip()
        if combined:
            result[sec] = combined
            
    return result


def parse_resume(pdf_bytes: bytes, filename: str = "uploaded_resume.pdf") -> ResumeParseResult:
    """Full parsing pipeline returning ResumeParseResult."""
    clean_txt, pages = extract_text_from_pdf_bytes(pdf_bytes)
    sections_map = identify_sections(clean_txt)
    
    words = clean_txt.split()
    word_count = len(words)
    char_count = len(clean_txt)
    
    detected_list: List[SectionDetail] = []
    for sec_name in STANDARD_SECTIONS:
        present = sec_name in sections_map and len(sections_map[sec_name]) > 0
        sec_text = sections_map.get(sec_name, "")
        sec_words = len(sec_text.split()) if sec_text else 0
        preview = sec_text[:120] + "..." if len(sec_text) > 120 else sec_text
        
        detected_list.append(SectionDetail(
            name=sec_name,
            present=present,
            word_count=sec_words,
            preview=preview
        ))
        
    return ResumeParseResult(
        filename=filename,
        total_pages=pages,
        char_count=char_count,
        word_count=word_count,
        clean_text=clean_txt,
        sections=sections_map,
        detected_sections=detected_list
    )
