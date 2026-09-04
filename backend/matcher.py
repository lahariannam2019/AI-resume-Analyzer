"""
Matching Engine for comparing Resume content against Target Job Description.
"""

import re
from typing import List, Dict, Set, Any
from backend.models import SkillMatchResult, KeywordMatchResult, ProjectRelevance, SkillItem
from backend.nlp_analyzer import extract_skills, extract_keywords_and_phrases


def analyze_skill_match(resume_text: str, jd_text: str) -> SkillMatchResult:
    """Compare skills extracted from Resume vs Job Description."""
    resume_skills = extract_skills(resume_text)
    jd_skills = extract_skills(jd_text)
    
    resume_skill_names = {s.name for s in resume_skills}
    
    matched_skills: List[SkillItem] = []
    missing_skills: List[SkillItem] = []
    
    for s in jd_skills:
        if s.name in resume_skill_names:
            matched_skills.append(s)
        else:
            missing_skills.append(s)
            
    if jd_skills:
        match_pct = (len(matched_skills) / len(jd_skills)) * 100.0
    else:
        # If no explicit skills found in JD, fall back to default baseline match
        match_pct = 80.0 if resume_skills else 50.0
        
    return SkillMatchResult(
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        resume_skills=resume_skills,
        jd_skills=jd_skills,
        match_percentage=round(match_pct, 2)
    )


def analyze_keywords(resume_text: str, jd_text: str) -> KeywordMatchResult:
    """Analyze top JD keyphrase coverage in the resume text."""
    jd_keywords = extract_keywords_and_phrases(jd_text, top_n=20)
    
    if not jd_keywords:
        return KeywordMatchResult(
            matched_keywords=[],
            missing_keywords=[],
            jd_top_keywords=[],
            coverage_percentage=100.0
        )
        
    resume_lower = resume_text.lower()
    matched: List[str] = []
    missing: List[str] = []
    
    for kw in jd_keywords:
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        if re.search(pattern, resume_lower):
            matched.append(kw)
        else:
            missing.append(kw)
            
    coverage_pct = (len(matched) / len(jd_keywords)) * 100.0 if jd_keywords else 100.0
    
    return KeywordMatchResult(
        matched_keywords=matched,
        missing_keywords=missing,
        jd_top_keywords=jd_keywords,
        coverage_percentage=round(coverage_pct, 2)
    )


def analyze_projects(resume_text: str, jd_text: str, sections: Dict[str, str]) -> List[ProjectRelevance]:
    """Identify project entries and assess relevance to target JD."""
    projects_text = sections.get("Projects", "")
    if not projects_text:
        # Search for project bullet points in full text if section was not demarcated
        project_lines = [line for line in resume_text.split('\n') if 'project' in line.lower() or 'built' in line.lower() or 'developed' in line.lower()]
        projects_text = "\n".join(project_lines[:6])
        
    if not projects_text:
        return []
        
    jd_terms = [t.lower() for t in extract_keywords_and_phrases(jd_text, top_n=15)]
    
    # Split projects text by double newlines or explicit bullet delimiters
    raw_entries = re.split(r'\n\s*\n|\n(?=[•\*\-\#]|\bProject\b)', projects_text)
    entries = [p.strip() for p in raw_entries if len(p.strip()) > 15]
    
    if not entries and projects_text.strip():
        entries = [projects_text.strip()]
    
    results: List[ProjectRelevance] = []
    for idx, entry in enumerate(entries[:5]):
        first_line = entry.split('\n')[0].strip()
        title = first_line[:60] if len(first_line) > 60 else first_line
        if not title:
            title = f"Project #{idx+1}"
            
        matched_terms = []
        entry_lower = entry.lower()
        for term in jd_terms:
            if term in entry_lower:
                matched_terms.append(term.title())
                
        # Calculate score based on matched terms count
        score = min(len(matched_terms) * 25.0 + 30.0, 100.0) if matched_terms else 35.0
        
        results.append(ProjectRelevance(
            title=title,
            relevance_score=round(score, 1),
            matched_terms=matched_terms
        ))
        
    return results
