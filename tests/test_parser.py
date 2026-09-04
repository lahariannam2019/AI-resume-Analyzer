"""
Unit tests for PDF parsing & section identification.
"""

import pytest
import fitz  # PyMuPDF
from backend.parser import clean_text, identify_sections, parse_resume


def test_clean_text():
    raw = "  Hello   World!\r\n\n\nThis is a   test.  "
    cleaned = clean_text(raw)
    assert "Hello World!" in cleaned
    assert "  " not in cleaned


def test_identify_sections():
    sample_text = """
    Jane Doe
    jane@example.com

    SUMMARY
    Experienced AI Engineer specializing in Natural Language Processing and LLMs.

    EXPERIENCE
    Senior Developer at TechCorp (2021-Present)
    Built PyTorch recommendation pipelines.

    SKILLS
    Python, PyTorch, FastApi, SQL, Docker

    EDUCATION
    BS Computer Science, Tech University
    """
    sections = identify_sections(sample_text)
    assert "Summary" in sections
    assert "Experience" in sections
    assert "Skills" in sections
    assert "Education" in sections
    assert "PyTorch" in sections["Skills"]


def test_parse_resume_synthetic_pdf():
    # Create an in-memory PDF document using PyMuPDF
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "John Doe\nSummary\nData Scientist with Python experience.\nSkills\nPython, SQL, Machine Learning")
    pdf_bytes = doc.tobytes()
    doc.close()

    result = parse_resume(pdf_bytes, filename="test_resume.pdf")
    assert result.filename == "test_resume.pdf"
    assert result.total_pages == 1
    assert "John Doe" in result.clean_text
    assert "Summary" in result.sections or "General" in result.sections
