"""
Unit tests for Matching Engine (skills comparison, keyword coverage, project relevance).
"""

import pytest
from backend.matcher import analyze_skill_match, analyze_keywords, analyze_projects


def test_analyze_skill_match():
    resume = "Skills: Python, React, PostgreSQL, Git, Docker."
    jd = "Requirements: Python, React, TypeScript, AWS, Docker."
    
    res = analyze_skill_match(resume, jd)
    matched_names = {s.name for s in res.matched_skills}
    missing_names = {s.name for s in res.missing_skills}
    
    assert "Python" in matched_names
    assert "React" in matched_names
    assert "TypeScript" in missing_names
    assert res.match_percentage > 0


def test_analyze_keywords():
    resume = "Developed responsive React web applications with REST APIs and state management."
    jd = "Seeking developer experienced in React web applications, REST APIs, and responsive design."
    
    kw_res = analyze_keywords(resume, jd)
    assert len(kw_res.jd_top_keywords) > 0
    assert kw_res.coverage_percentage > 0.0


def test_analyze_projects():
    sections = {
        "Projects": "AI Resume Matcher\nBuilt an NLP resume analyzer using Python, spaCy, and FastAPI."
    }
    jd = "Requirements: Python, FastAPI, NLP."
    projects = analyze_projects("", jd, sections)
    assert len(projects) > 0
    assert projects[0].relevance_score > 50.0
