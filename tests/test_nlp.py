"""
Unit tests for NLP skill extraction and semantic similarity calculations.
"""

import pytest
from backend.nlp_analyzer import extract_skills, calculate_semantic_similarity, extract_keywords_and_phrases


def test_extract_skills():
    text = "Proficient in Python, JS, React.js, PostgreSQL, Docker, AWS, and Machine Learning."
    extracted = extract_skills(text)
    names = {s.name for s in extracted}
    
    assert "Python" in names
    assert "JavaScript" in names  # Normalized from JS
    assert "React" in names       # Normalized from React.js
    assert "PostgreSQL" in names
    assert "Docker" in names
    assert "Amazon Web Services" in names or "AWS" in names
    assert "Machine Learning" in names


def test_calculate_semantic_similarity():
    text1 = "Python developer with experience in Django, REST APIs, PostgreSQL, and Git."
    text2 = "Looking for a Python developer proficient in Django, RESTful API design, and SQL."
    
    score = calculate_semantic_similarity(text1, text2)
    assert score > 15.0  # Should show significant semantic overlap
    
    unrelated_text = "Looking for a certified professional chef to bake cakes and manage kitchen staff."
    unrelated_score = calculate_semantic_similarity(text1, unrelated_text)
    assert score > unrelated_score


def test_extract_keywords_and_phrases():
    jd = "Seeking a Frontend Developer skilled in React, TypeScript, state management, and user interface design."
    keywords = extract_keywords_and_phrases(jd)
    assert len(keywords) > 0
