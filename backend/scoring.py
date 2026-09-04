"""
Explainable Multi-Factor Scoring Engine for AI Resume Analyzer.
"""

from typing import Dict, List, Any
from backend.models import ScoreBreakdown, SkillMatchResult, KeywordMatchResult, ProjectRelevance


def calculate_overall_score(
    skills_res: SkillMatchResult,
    keywords_res: KeywordMatchResult,
    projects: List[ProjectRelevance],
    semantic_similarity: float,
    sections: Dict[str, str]
) -> ScoreBreakdown:
    """
    Computes a deterministic, explainable match score out of 100 based on weighted factors:
    - 40% Skill Match
    - 25% Keyword Match
    - 20% Relevant Experience / Projects
    - 15% Semantic Similarity
    """
    # 1. Skill Score (40%)
    skill_score = min(max(skills_res.match_percentage, 0.0), 100.0)
    
    # 2. Keyword Coverage Score (25%)
    keyword_score = min(max(keywords_res.coverage_percentage, 0.0), 100.0)
    
    # 3. Experience & Projects Relevance Score (20%)
    if projects:
        avg_proj_score = sum(p.relevance_score for p in projects) / len(projects)
    else:
        avg_proj_score = 40.0
        
    has_exp_section = "Experience" in sections and len(sections["Experience"]) > 30
    has_proj_section = "Projects" in sections and len(sections["Projects"]) > 30
    
    section_bonus = 0.0
    if has_exp_section: section_bonus += 15.0
    if has_proj_section: section_bonus += 15.0
    
    experience_score = min(avg_proj_score + section_bonus, 100.0)
    
    # 4. Semantic Similarity Score (15%)
    semantic_score = min(max(semantic_similarity, 0.0), 100.0)
    
    # Calculate weighted overall score
    overall = (
        (0.40 * skill_score) +
        (0.25 * keyword_score) +
        (0.20 * experience_score) +
        (0.15 * semantic_score)
    )
    overall_score = round(min(max(overall, 0.0), 100.0), 1)
    
    # Assign tier rating
    if overall_score >= 85.0:
        tier = "Exceptional Match"
    elif overall_score >= 75.0:
        tier = "Strong Match"
    elif overall_score >= 60.0:
        tier = "Moderate Fit"
    elif overall_score >= 45.0:
        tier = "Needs Targeted Optimization"
    else:
        tier = "Low Match"
        
    # Generate transparent explanations
    explanations: List[str] = []
    
    explanations.append(f"Skill Match ({skill_score:.1f}%): You matched {len(skills_res.matched_skills)} out of {len(skills_res.jd_skills)} target job skills.")
    
    if keywords_res.jd_top_keywords:
        explanations.append(f"Keyword Coverage ({keyword_score:.1f}%): Your resume covers {len(keywords_res.matched_keywords)} of top {len(keywords_res.jd_top_keywords)} job posting keyphrases.")
    
    explanations.append(f"Experience & Projects ({experience_score:.1f}%): Evaluated based on project alignment and section presence.")
    explanations.append(f"Semantic Alignment ({semantic_score:.1f}%): TF-IDF vector similarity between full resume text and target job description.")

    return ScoreBreakdown(
        overall_score=overall_score,
        skill_score=round(skill_score, 1),
        keyword_score=round(keyword_score, 1),
        experience_score=round(experience_score, 1),
        semantic_score=round(semantic_score, 1),
        tier=tier,
        explanation=explanations
    )
