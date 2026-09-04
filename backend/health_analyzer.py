"""
Resume Health Evaluation Engine.
Evaluates overall resume readiness, section structure, ATS readability, content quality, 
skills range, project presence, and quantifiable metrics WITHOUT requiring a specific job description.
"""

import re
from typing import Dict, List, Any, Tuple, Optional
from backend.models import ResumeParseResult, SkillItem
from backend.nlp_analyzer import extract_skills


STANDARD_SECTIONS_COUNT = 7.0


def evaluate_resume_health(parse_info: ResumeParseResult) -> Dict[str, Any]:
    """
    Computes a deterministic, rule-based Resume Health Score (out of 100) 
    and detailed category diagnostics.
    """
    clean_text = parse_info.clean_text
    sections_map = parse_info.sections
    detected_sections = parse_info.detected_sections
    
    words = clean_text.split()
    word_count = len(words)
    
    # 1. ATS Readiness Score (20%)
    present_headers = [sec.name for sec in detected_sections if sec.present]
    essential_headers = ["Summary", "Education", "Skills", "Experience", "Projects"]
    missing_essential = [h for h in essential_headers if h not in present_headers]
    
    header_score = (len(present_headers) / STANDARD_SECTIONS_COUNT) * 100.0
    ats_score = min(max(header_score + (10.0 if word_count >= 180 else -20.0), 0.0), 100.0)
    
    # 2. Content Quality & Impact (20%)
    action_verbs = {
        "built", "developed", "designed", "engineered", "implemented", "led", 
        "managed", "optimized", "scaled", "created", "spearheaded", "increased", 
        "reduced", "improved", "automated", "launched", "formulated", "deployed"
    }
    found_verbs = {w.lower() for w in words if w.lower() in action_verbs}
    verb_score = min(len(found_verbs) * 12.0 + 30.0, 100.0)
    
    # Check bullet metric presence (numbers, percentages, dollar amounts, multipliers)
    metrics_matches = re.findall(r'\b\d+(?:\.\d+)?%|\$\d+|\b\d+\s*(?:users|clients|ms|s|x|xfold|k|m|b)\b', clean_text, re.IGNORECASE)
    metric_count = len(metrics_matches)
    metric_score = min(metric_count * 20.0 + 30.0, 100.0)
    
    content_quality_score = round(0.5 * verb_score + 0.5 * metric_score, 1)
    
    # 3. Structure & Formatting (15%)
    structure_score = 90.0 if len(missing_essential) == 0 else max(90.0 - (len(missing_essential) * 15.0), 40.0)
    if word_count < 120:
        structure_score -= 20.0
    elif word_count > 1100:
        structure_score -= 10.0
    structure_score = min(max(structure_score, 0.0), 100.0)
    
    # 4. Skills Range & Diversity (15%)
    extracted_skills = extract_skills(clean_text)
    skill_categories = {s.category for s in extracted_skills}
    skill_score = min(len(extracted_skills) * 7.0 + len(skill_categories) * 10.0 + 20.0, 100.0) if extracted_skills else 35.0
    
    # 5. Project Presence & Detail (15%)
    projects_text = sections_map.get("Projects", "")
    has_projects_sec = bool(projects_text)
    project_score = min(len(projects_text.split()) * 0.8 + (30.0 if has_projects_sec else 0.0), 100.0) if projects_text else (45.0 if "project" in clean_text.lower() else 25.0)
    
    # 6. Experience & Accomplishments (15%)
    exp_text = sections_map.get("Experience", "")
    has_exp_sec = bool(exp_text)
    exp_score = min(len(exp_text.split()) * 0.5 + (35.0 if has_exp_sec else 0.0), 100.0) if exp_text else (40.0 if "work" in clean_text.lower() or "intern" in clean_text.lower() else 25.0)
    
    # Overall Weighted Health Score
    overall_health = (
        (0.20 * ats_score) +
        (0.20 * content_quality_score) +
        (0.15 * structure_score) +
        (0.15 * skill_score) +
        (0.15 * project_score) +
        (0.15 * exp_score)
    )
    overall_health_score = round(min(max(overall_health, 0.0), 100.0), 1)
    
    # Status rating
    if overall_health_score >= 85.0:
        status = "Exceptional — Highly job-ready"
    elif overall_health_score >= 75.0:
        status = "Good — A few targeted improvements will make it stronger"
    elif overall_health_score >= 60.0:
        status = "Moderate — Needs stronger impact metrics and skills"
    else:
        status = "Needs Attention — Missing essential sections or bullet detail"
        
    # Detect Links & Contact Info
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', clean_text)
    linkedin_match = re.search(r'linkedin\.com/in/[a-zA-Z0-9_-]+', clean_text, re.IGNORECASE)
    github_match = re.search(r'github\.com/[a-zA-Z0-9_-]+', clean_text, re.IGNORECASE)
    portfolio_match = re.search(r'https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', clean_text)
    
    detected_links = {
        "email": email_match.group(0) if email_match else None,
        "linkedin": linkedin_match.group(0) if linkedin_match else None,
        "github": github_match.group(0) if github_match else None,
        "portfolio": portfolio_match.group(0) if portfolio_match else None
    }
    
    # Generate Strengths
    strengths: List[str] = []
    if len(extracted_skills) >= 4:
        strengths.append(f"Strong technical skills section with {len(extracted_skills)} categorized skills ({', '.join([s.name for s in extracted_skills[:4]])}).")
    if has_projects_sec:
        strengths.append("Dedicated Projects section highlighting hands-on technical work.")
    if has_exp_sec:
        strengths.append("Clear Work Experience section with chronological history.")
    if metric_count >= 1:
        strengths.append(f"Includes {metric_count} quantifiable accomplishment metrics (e.g., percentages, figures).")
    if detected_links["linkedin"] or detected_links["github"]:
        strengths.append("Contains professional online profiles (LinkedIn / GitHub).")
    if len(strengths) < 2:
        strengths.append("Valid PDF format with readable text content.")
        
    # Generate Areas to Improve
    areas_to_improve: List[str] = []
    if missing_essential:
        areas_to_improve.append(f"Missing standard section headings: {', '.join(missing_essential)}.")
    if metric_count < 2:
        areas_to_improve.append("Lacks quantifiable outcomes (e.g., 'Increased performance by 30%', 'Managed 10k users').")
    if len(found_verbs) < 4:
        areas_to_improve.append("Use stronger action verbs (e.g., 'Engineered', 'Spearheaded', 'Optimized') at the start of bullet points.")
    if not detected_links["linkedin"] and not detected_links["github"]:
        areas_to_improve.append("Add links to your LinkedIn profile, GitHub repository, or online portfolio.")
    if word_count < 180:
        areas_to_improve.append("Resume content is concise (<180 words). Expand on responsibilities and project features.")

    # Recommended Next Steps
    recommended_next_steps = [
        "Rewrite generic bullet points to include measurable outcomes and concrete figures.",
        "Ensure standard headings (Summary, Experience, Projects, Skills, Education) are formatted clearly for ATS parsers.",
        "Add explicit links to your GitHub repositories or live project demos."
    ]

    category_scores = {
        "ATS Readiness": round(ats_score, 1),
        "Content Quality": round(content_quality_score, 1),
        "Structure & Formatting": round(structure_score, 1),
        "Skills Range": round(skill_score, 1),
        "Projects Detail": round(project_score, 1),
        "Experience Depth": round(exp_score, 1)
    }

    return {
        "overall_health_score": overall_health_score,
        "status": status,
        "category_scores": category_scores,
        "strengths": strengths,
        "areas_to_improve": areas_to_improve,
        "recommended_next_steps": recommended_next_steps,
        "detected_links": detected_links,
        "extracted_skills": [s.model_dump() for s in extracted_skills] if extracted_skills and hasattr(extracted_skills[0], 'model_dump') else [{"name": s.name, "category": s.category} for s in extracted_skills]
    }
