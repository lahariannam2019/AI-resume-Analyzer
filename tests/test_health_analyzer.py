"""
Unit tests for Resume Health Evaluation Engine, Dual Analysis Modes, and Custom Job Role Handling.
"""

import pytest
import os
from backend.models import ResumeParseResult, SectionDetail
from backend.health_analyzer import evaluate_resume_health
from backend.nlp_analyzer import extract_role_title_from_jd
from backend.database import init_db, create_user, save_analysis, get_history, clear_user_history


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    test_db_file = os.path.join(tmp_path, "test_dual_mode.db")
    monkeypatch.setattr("backend.database.DB_PATH", test_db_file)
    init_db()


def test_evaluate_resume_health_scoring():
    """Verify Resume Health evaluation calculates sub-scores, strengths, and recommendations."""
    parse_info = ResumeParseResult(
        filename="test_resume.pdf",
        total_pages=1,
        char_count=1200,
        word_count=220,
        clean_text="Jane Doe. Email: jane@example.com. LinkedIn: linkedin.com/in/janedoe. Summary: Experienced Software Engineer. Experience: Built scalable microservices using Python and React. Increased performance by 40%. Projects: Developed AI resume parser using spaCy, Docker, and PostgreSQL. Skills: Python, React, PostgreSQL, Docker, Git.",
        sections={
            "Summary": "Experienced Software Engineer.",
            "Experience": "Built scalable microservices using Python and React. Increased performance by 40%.",
            "Projects": "Developed AI resume parser using spaCy, Docker, and PostgreSQL.",
            "Skills": "Python, React, PostgreSQL, Docker, Git."
        },
        detected_sections=[
            SectionDetail(name="Summary", present=True, word_count=3),
            SectionDetail(name="Experience", present=True, word_count=10),
            SectionDetail(name="Projects", present=True, word_count=9),
            SectionDetail(name="Skills", present=True, word_count=5),
            SectionDetail(name="Education", present=False, word_count=0)
        ]
    )
    
    health_result = evaluate_resume_health(parse_info)
    
    assert "overall_health_score" in health_result
    assert 0.0 <= health_result["overall_health_score"] <= 100.0
    assert "category_scores" in health_result
    assert "ATS Readiness" in health_result["category_scores"]
    assert "Content Quality" in health_result["category_scores"]
    assert len(health_result["strengths"]) >= 1
    assert len(health_result["recommended_next_steps"]) >= 1
    assert health_result["detected_links"]["linkedin"] == "linkedin.com/in/janedoe"


def test_extract_role_title_from_jd():
    """Verify role title auto-extraction from JD text."""
    jd1 = "We are seeking a Senior DevOps Engineer to manage Kubernetes clusters."
    assert extract_role_title_from_jd(jd1) is not None
    
    jd2 = "Position: Cybersecurity Analyst\nResponsibilities include threat monitoring and incident response."
    assert extract_role_title_from_jd(jd2) == "Cybersecurity Analyst"


def test_dual_mode_history_isolation():
    """Verify SQLite history correctly isolates Resume Health vs Job Match entries for a user."""
    user = create_user("Mode User", "mode@example.com", "pass123456")
    user_id = user["user_id"]
    
    # Save Resume Health entry
    health_entry = {
        "analysis_id": "anl_health_01",
        "analysis_type": "resume_health",
        "filename": "health.pdf",
        "health_info": {"overall_health_score": 82.0, "status": "Good"}
    }
    save_analysis(health_entry, user_id=user_id)
    
    # Save Job Match entry
    match_entry = {
        "analysis_id": "anl_match_01",
        "analysis_type": "job_match",
        "filename": "match.pdf",
        "target_role": "Cybersecurity Analyst",
        "score_breakdown": {"overall_score": 75.0, "tier": "Strong Match"}
    }
    save_analysis(match_entry, user_id=user_id)
    
    history = get_history(user_id=user_id)
    assert len(history) == 2
    
    types = [h["analysis_type"] for h in history]
    assert "resume_health" in types
    assert "job_match" in types
