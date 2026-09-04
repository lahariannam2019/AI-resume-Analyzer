"""
Pydantic schemas and data models for AI Resume Analyzer (supporting Dual Modes & Auth).
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


# --- Authentication Schemas ---

class UserSignUp(BaseModel):
    full_name: str = Field(..., min_length=2, description="Full name of user")
    email: str = Field(..., description="Valid email address")
    password: str = Field(..., min_length=6, description="Password min 6 characters")
    confirm_password: str = Field(..., description="Must match password")


class UserLogin(BaseModel):
    email: str
    password: str


class UserProfile(BaseModel):
    user_id: str
    full_name: str
    email: str
    created_at: str
    total_analyses: int = 0


class AuthResponse(BaseModel):
    user: UserProfile
    token: str
    message: str = "Authentication successful"


# --- Analysis & Matching Schemas ---

class SectionDetail(BaseModel):
    name: str
    present: bool
    word_count: int = 0
    preview: str = ""


class SkillItem(BaseModel):
    name: str
    category: str


class SkillMatchResult(BaseModel):
    matched_skills: List[SkillItem] = []
    missing_skills: List[SkillItem] = []
    resume_skills: List[SkillItem] = []
    jd_skills: List[SkillItem] = []
    match_percentage: float = 0.0


class KeywordMatchResult(BaseModel):
    matched_keywords: List[str] = []
    missing_keywords: List[str] = []
    jd_top_keywords: List[str] = []
    coverage_percentage: float = 0.0


class ProjectRelevance(BaseModel):
    title: str
    relevance_score: float
    matched_terms: List[str] = []


class ScoreBreakdown(BaseModel):
    overall_score: float = Field(..., description="Weighted overall score out of 100")
    skill_score: float = Field(..., description="Skill match component (40%)")
    keyword_score: float = Field(..., description="Keyword coverage component (25%)")
    experience_score: float = Field(..., description="Relevant experience/project score (20%)")
    semantic_score: float = Field(..., description="TF-IDF Cosine Similarity score (15%)")
    tier: str = Field(..., description="Score rating category, e.g., 'Strong Match'")
    explanation: List[str] = Field(default=[], description="Bullet point explanations for the score")


class SuggestionItem(BaseModel):
    category: str = Field(..., description="e.g., 'Skills', 'Metrics', 'Formatting'")
    impact: str = Field(..., description="'High', 'Medium', or 'Low'")
    title: str
    message: str


class ResumeParseResult(BaseModel):
    filename: str
    total_pages: int
    char_count: int
    word_count: int
    clean_text: str
    sections: Dict[str, str]
    detected_sections: List[SectionDetail]


class AnalysisResult(BaseModel):
    analysis_id: Optional[str] = None
    analysis_type: str = Field(default="job_match", description="'resume_health' or 'job_match'")
    user_id: str
    filename: str
    target_role: Optional[str] = "Target Job Role"
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    parse_info: ResumeParseResult
    score_breakdown: Optional[ScoreBreakdown] = None
    skills_analysis: Optional[SkillMatchResult] = None
    keywords_analysis: Optional[KeywordMatchResult] = None
    project_analysis: List[ProjectRelevance] = []
    suggestions: List[SuggestionItem] = []
    health_info: Optional[Dict[str, Any]] = None


class HistoryItem(BaseModel):
    id: int
    analysis_id: str
    analysis_type: str
    user_id: str
    filename: str
    target_role: Optional[str] = None
    overall_score: float
    tier: str
    matched_skills_count: int
    missing_skills_count: int
    timestamp: str
