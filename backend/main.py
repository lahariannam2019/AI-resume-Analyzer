"""
FastAPI REST API server for AI Resume Analyzer with Dual Analysis Modes (Resume Health & Job Match)
and Secure JWT Authorization & User Scoping.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import datetime
import uuid

from backend.models import (
    AnalysisResult, UserSignUp, UserLogin, UserProfile, AuthResponse, HistoryItem
)
from backend.auth import (
    hash_password, verify_password, validate_email_format, validate_password_strength,
    create_access_token, decode_access_token
)
from backend.parser import parse_resume
from backend.nlp_analyzer import calculate_semantic_similarity, extract_role_title_from_jd
from backend.health_analyzer import evaluate_resume_health
from backend.matcher import analyze_skill_match, analyze_keywords, analyze_projects
from backend.scoring import calculate_overall_score
from backend.suggestions import generate_suggestions
from backend.database import (
    init_db, create_user, get_user_by_email, get_user_by_id, get_user_stats,
    save_analysis, get_history, get_analysis_by_id, clear_user_history
)

app = FastAPI(
    title="AI Resume Analyzer API",
    description="Dual Mode AI Resume Analyzer API (Resume Health & Custom Job Match) with JWT Auth",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_db_client():
    init_db()


def get_authenticated_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    FastAPI Security Dependency: Derives current user identity strictly from the Bearer JWT token.
    Raises 401 Unauthorized if missing, malformed, or invalid.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authentication token required in 'Authorization: Bearer <token>' header."
        )
    token = authorization.split(" ")[1]
    payload = decode_access_token(token)
    if not payload or "user_id" not in payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid, expired, or corrupted authentication token."
        )
    user = get_user_by_id(payload["user_id"])
    if not user:
        raise HTTPException(
            status_code=401,
            detail="User account associated with token no longer exists."
        )
    return user


@app.get("/")
def read_root():
    return {
        "message": "AI Resume Analyzer Dual-Mode API is running.",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "AI Resume Analyzer API",
        "timestamp": datetime.datetime.now().isoformat()
    }


# --- Authentication Endpoints ---

@app.post("/auth/signup", response_model=AuthResponse)
def signup(user_data: UserSignUp):
    """Register a new user account."""
    if not validate_email_format(user_data.email):
        raise HTTPException(status_code=400, detail="Invalid email address format.")
        
    is_valid_pwd, pwd_msg = validate_password_strength(user_data.password)
    if not is_valid_pwd:
        raise HTTPException(status_code=400, detail=pwd_msg)
        
    if user_data.password != user_data.confirm_password:
        raise HTTPException(status_code=400, detail="Password and Confirm Password do not match.")
        
    existing = get_user_by_email(user_data.email)
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email address already exists.")
        
    pwd_hash = hash_password(user_data.password)
    
    try:
        user_info = create_user(user_data.full_name, user_data.email, pwd_hash)
        token = create_access_token({"user_id": user_info["user_id"], "email": user_info["email"]})
        
        profile = UserProfile(
            user_id=user_info["user_id"],
            full_name=user_info["full_name"],
            email=user_info["email"],
            created_at=user_info["created_at"],
            total_analyses=0
        )
        return AuthResponse(user=profile, token=token, message="Account created successfully!")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@app.post("/auth/login", response_model=AuthResponse)
def login(credentials: UserLogin):
    """Authenticate existing user credentials."""
    if not credentials.email or not credentials.password:
        raise HTTPException(status_code=400, detail="Email and password are required.")
        
    user = get_user_by_email(credentials.email)
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
        
    token = create_access_token({"user_id": user["user_id"], "email": user["email"]})
    stats = get_user_stats(user["user_id"])
    
    profile = UserProfile(
        user_id=user["user_id"],
        full_name=user["full_name"],
        email=user["email"],
        created_at=user["created_at"],
        total_analyses=stats["total_analyses"]
    )
    return AuthResponse(user=profile, token=token, message="Login successful!")


@app.get("/auth/me", response_model=UserProfile)
def get_current_user_profile(current_user: Dict[str, Any] = Depends(get_authenticated_user)):
    """Retrieve authenticated user profile derived strictly from JWT Bearer token."""
    stats = get_user_stats(current_user["user_id"])
    return UserProfile(
        user_id=current_user["user_id"],
        full_name=current_user["full_name"],
        email=current_user["email"],
        created_at=current_user["created_at"],
        total_analyses=stats["total_analyses"]
    )


# --- Analysis Endpoints: Dual Modes (Resume Health & Job Match) ---

@app.post("/extract-resume")
async def extract_resume(file: UploadFile = File(...)):
    """Extract clean text and sections from PDF resume."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        return parse_resume(content, filename=file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {str(e)}")


@app.post("/analyze/health")
async def analyze_resume_health_endpoint(
    file: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    """
    MODE A: RESUME HEALTH EVALUATION.
    Evaluates resume strength, section completeness, ATS readiness, content quality, and impact metrics
    WITHOUT requiring a job description or target job title.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    user_id = "guest_user"
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        payload = decode_access_token(token)
        if payload and "user_id" in payload:
            user_id = payload["user_id"]
            
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")
            
        parse_info = parse_resume(content, filename=file.filename)
        if len(parse_info.clean_text.strip()) < 20:
            raise HTTPException(status_code=400, detail="Extracted resume text is too short or unreadable.")
            
        health_diagnostics = evaluate_resume_health(parse_info)
        analysis_id = f"anl_{uuid.uuid4().hex[:10]}"
        timestamp = datetime.datetime.now().isoformat()
        
        result_dict = {
            "analysis_id": analysis_id,
            "analysis_type": "resume_health",
            "user_id": user_id,
            "filename": file.filename,
            "target_role": "General Resume Health",
            "timestamp": timestamp,
            "parse_info": parse_info.model_dump(),
            "health_info": health_diagnostics
        }
        
        save_analysis(result_dict, user_id=user_id)
        return result_dict
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume Health analysis failed: {str(e)}")


@app.post("/analyze/match")
async def analyze_job_match_endpoint(
    file: UploadFile = File(...),
    job_description: str = Form(...),
    target_role: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None)
):
    """
    MODE B: JOB MATCH EVALUATION.
    Compares resume against a specific target job role and description.
    Supports arbitrary custom job titles or auto-extracts title if omitted.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    if not job_description or not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description text is required.")
        
    user_id = "guest_user"
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        payload = decode_access_token(token)
        if payload and "user_id" in payload:
            user_id = payload["user_id"]
            
    # Auto-extract role title if omitted
    resolved_role = target_role.strip() if target_role and target_role.strip() else extract_role_title_from_jd(job_description) or "Target Job Role"
            
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")
            
        parse_info = parse_resume(content, filename=file.filename)
        if len(parse_info.clean_text.strip()) < 20:
            raise HTTPException(status_code=400, detail="Extracted resume text is too short or unreadable.")
            
        skills_res = analyze_skill_match(parse_info.clean_text, job_description)
        keywords_res = analyze_keywords(parse_info.clean_text, job_description)
        projects_res = analyze_projects(parse_info.clean_text, job_description, parse_info.sections)
        semantic_sim = calculate_semantic_similarity(parse_info.clean_text, job_description)
        
        score_breakdown = calculate_overall_score(
            skills_res=skills_res,
            keywords_res=keywords_res,
            projects=projects_res,
            semantic_similarity=semantic_sim,
            sections=parse_info.sections
        )
        
        suggestions = generate_suggestions(
            parse_info=parse_info,
            skills_res=skills_res,
            keywords_res=keywords_res,
            target_role=resolved_role,
            jd_text=job_description
        )
        
        analysis_id = f"anl_{uuid.uuid4().hex[:10]}"
        timestamp = datetime.datetime.now().isoformat()
        
        result_dict = {
            "analysis_id": analysis_id,
            "analysis_type": "job_match",
            "user_id": user_id,
            "filename": file.filename,
            "target_role": resolved_role,
            "timestamp": timestamp,
            "parse_info": parse_info.model_dump(),
            "score_breakdown": score_breakdown.model_dump(),
            "skills_analysis": skills_res.model_dump(),
            "keywords_analysis": keywords_res.model_dump(),
            "project_analysis": [p.model_dump() for p in projects_res],
            "suggestions": [s.model_dump() for s in suggestions]
        }
        
        save_analysis(result_dict, user_id=user_id)
        return result_dict
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Job match analysis failed: {str(e)}")


@app.post("/analyze")
async def analyze_resume_endpoint(
    file: UploadFile = File(...),
    analysis_type: str = Form("job_match"),
    job_description: Optional[str] = Form(None),
    target_role: Optional[str] = Form(None),
    authorization: Optional[str] = Header(None)
):
    """General analysis route routing to Resume Health or Job Match."""
    if analysis_type == "resume_health":
        return await analyze_resume_health_endpoint(file=file, authorization=authorization)
    else:
        if not job_description or not job_description.strip():
            raise HTTPException(status_code=400, detail="Job description text is required for Job Match mode.")
        return await analyze_job_match_endpoint(
            file=file,
            job_description=job_description,
            target_role=target_role,
            authorization=authorization
        )


@app.get("/history")
def get_user_analysis_history(
    current_user: Dict[str, Any] = Depends(get_authenticated_user),
    limit: int = Query(50, ge=1, le=100)
):
    """Retrieve history records strictly belonging to the authenticated user derived from JWT."""
    user_id = current_user["user_id"]
    try:
        history = get_history(user_id=user_id, limit=limit)
        return {"user_id": user_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


@app.get("/history/{analysis_id}")
def get_user_analysis_details(
    analysis_id: str,
    current_user: Dict[str, Any] = Depends(get_authenticated_user)
):
    """Retrieve detailed analysis record by ID strictly if owned by the authenticated user."""
    user_id = current_user["user_id"]
    result = get_analysis_by_id(analysis_id, user_id=user_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis record not found for this user.")
    return result


@app.delete("/history")
def clear_user_history_endpoint(current_user: Dict[str, Any] = Depends(get_authenticated_user)):
    """Delete all history records strictly for the authenticated user derived from JWT."""
    user_id = current_user["user_id"]
    deleted_count = clear_user_history(user_id=user_id)
    return {"message": "History cleared successfully", "deleted_count": deleted_count}
