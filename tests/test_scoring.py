"""
Unit tests for multi-factor explainable scoring engine.
"""

import pytest
from backend.models import SkillMatchResult, KeywordMatchResult, ProjectRelevance, SkillItem
from backend.scoring import calculate_overall_score


def test_calculate_overall_score():
    skills_res = SkillMatchResult(
        matched_skills=[SkillItem(name="Python", category="Languages")],
        missing_skills=[SkillItem(name="AWS", category="Cloud")],
        match_percentage=75.0
    )
    
    keywords_res = KeywordMatchResult(
        matched_keywords=["Python", "FastAPI"],
        missing_keywords=["Docker"],
        jd_top_keywords=["Python", "FastAPI", "Docker"],
        coverage_percentage=66.7
    )
    
    projects = [
        ProjectRelevance(title="NLP App", relevance_score=80.0, matched_terms=["Python"])
    ]
    
    sections = {"Experience": "Worked 2 years", "Projects": "Built 3 apps"}
    semantic_similarity = 70.0
    
    score = calculate_overall_score(
        skills_res=skills_res,
        keywords_res=keywords_res,
        projects=projects,
        semantic_similarity=semantic_similarity,
        sections=sections
    )
    
    assert 0.0 <= score.overall_score <= 100.0
    assert score.skill_score == 75.0
    assert score.keyword_score == 66.7
    assert score.tier in ["Exceptional Match", "Strong Match", "Moderate Fit", "Needs Targeted Optimization", "Low Match"]
    assert len(score.explanation) == 4
