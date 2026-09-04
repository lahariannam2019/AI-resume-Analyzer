"""
Contextual Actionable AI Suggestion Generator for Resume Optimization.
"""

import re
from typing import List, Dict, Any
from backend.models import SuggestionItem, SkillMatchResult, KeywordMatchResult, ResumeParseResult


def generate_suggestions(
    parse_info: ResumeParseResult,
    skills_res: SkillMatchResult,
    keywords_res: KeywordMatchResult,
    target_role: str,
    jd_text: str
) -> List[SuggestionItem]:
    """
    Generates intelligent, actionable resume improvement suggestions based on analysis results.
    """
    suggestions: List[SuggestionItem] = []
    resume_text = parse_info.clean_text
    
    # 1. Missing High-Priority Skills Suggestion
    if skills_res.missing_skills:
        missing_names = [s.name for s in skills_res.missing_skills[:5]]
        missing_str = ", ".join(missing_names)
        suggestions.append(SuggestionItem(
            category="Skills Optimization",
            impact="High",
            title=f"Add Missing Key Technologies ({len(skills_res.missing_skills)} missing)",
            message=f"The job description explicitly asks for: {missing_str}. If you have experience with these, highlight them prominently in your Skills and Projects sections."
        ))

    # 2. Missing Keyphrase Coverage
    if keywords_res.missing_keywords:
        missing_kw = keywords_res.missing_keywords[:6]
        missing_kw_str = ", ".join([f"'{k}'" for k in missing_kw])
        suggestions.append(SuggestionItem(
            category="ATS Keyword Alignment",
            impact="High",
            title="Incorporate Critical Role Keyphrases",
            message=f"Your resume is missing key industry terms found in the job description: {missing_kw_str}. Weave these naturally into your work experience bullet points."
        ))

    # 3. Quantifiable Impact & Metrics Analysis
    # Search for numbers, percentages, dollar amounts, or metric symbols
    has_metrics = bool(re.search(r'\b\d+(?:\.\d+)?%|\$\d+|\b\d+\s*(?:users|clients|ms|s|x|xfold|k|m|b)\b', resume_text, re.IGNORECASE))
    if not has_metrics:
        suggestions.append(SuggestionItem(
            category="Impact & Accomplishments",
            impact="High",
            title="Add Quantifiable Results & Metrics",
            message="Your project and experience descriptions lack measurable outcomes. Recruiters favor bullet points with numbers—e.g., 'Reduced load time by 40%', 'Increased user engagement by 25%', or 'Managed 50k+ records'."
        ))
    else:
        suggestions.append(SuggestionItem(
            category="Impact & Accomplishments",
            impact="Medium",
            title="Strengthen Measurable Achievements",
            message="Great job including metrics! Ensure every major project highlights business impact or performance improvements using concrete figures."
        ))

    # 4. Resume Section Health Check
    missing_sections = [sec.name for sec in parse_info.detected_sections if not sec.present]
    if missing_sections:
        sections_str = ", ".join(missing_sections)
        suggestions.append(SuggestionItem(
            category="Structure & Formatting",
            impact="High",
            title=f"Missing Recommended Sections: {sections_str}",
            message=f"Consider adding explicit headings for {sections_str}. ATS parsers rely on standard section headers to correctly parse your resume."
        ))

    # 5. Role-Specific Tailored Recommendations
    jd_lower = jd_text.lower()
    role_lower = target_role.lower()
    
    # UI / UX Specific Advice
    if "ux" in jd_lower or "ui" in jd_lower or "design" in jd_lower or "ui" in role_lower:
        if "figma" not in resume_text.lower() and "user research" not in resume_text.lower():
            suggestions.append(SuggestionItem(
                category="UI/UX Relevance",
                impact="Medium",
                title="Highlight Design Process & Wireframing",
                message="For UI/UX roles, recruiters look for user research, usability testing, design systems, and Figma/Sketch portfolio links. Emphasize user-centered design methodologies."
            ))

    # AI / ML Specific Advice
    if "machine learning" in jd_lower or "ai" in jd_lower or "nlp" in jd_lower or "ml" in role_lower:
        if "evaluation" not in resume_text.lower() and "accuracy" not in resume_text.lower() and "f1" not in resume_text.lower():
            suggestions.append(SuggestionItem(
                category="AI/ML Relevance",
                impact="Medium",
                title="Detail Model Evaluation & Architecture Choices",
                message="Specify model architectures, framework versions (PyTorch/TensorFlow/scikit-learn), and performance metrics (F1-score, Accuracy, Latency) for your machine learning projects."
            ))

    # 6. Concise Bullet Length Check
    avg_word_len = len(resume_text.split())
    if avg_word_len < 150:
        suggestions.append(SuggestionItem(
            category="Content Depth",
            impact="High",
            title="Resume Content Is Too Brief",
            message="Your resume contains under 150 words. Expand on your project responsibilities, technical architecture, and individual contributions to provide sufficient material for matching."
        ))
    elif avg_word_len > 1000:
        suggestions.append(SuggestionItem(
            category="Content Brevity",
            impact="Medium",
            title="Consider Condensing Resume Length",
            message="Your resume is quite verbose (1000+ words). Keep bullet points concise, impactful, and tailored to fit within 1-2 pages for maximum reader engagement."
        ))

    return suggestions
