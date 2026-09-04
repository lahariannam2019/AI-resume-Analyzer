"""
Streamlit SaaS Dashboard Application for AI RESUME ANALYZER.
Dual Analysis Modes: RESUME HEALTH & JOB MATCH (supporting ANY custom role entry, pasted/uploaded JDs).
Secure Persistent JWT Authentication with Clean URL Bar (Zero Token Exposure),
Collapsible/Expandable Sidebar Controls, & User Isolation.
"""

import streamlit as st
import os
import sys
import time
import pandas as pd
from typing import Dict, Any, Optional

# Ensure parent directory is in sys.path so modules under backend/ can be imported directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.auth import (
    hash_password, verify_password, validate_email_format, validate_password_strength,
    create_access_token, decode_access_token
)
from backend.database import (
    init_db, create_user, get_user_by_email, get_user_by_id, get_user_stats,
    save_analysis, get_history, get_analysis_by_id, clear_user_history
)
from backend.parser import parse_resume
from backend.nlp_analyzer import calculate_semantic_similarity, extract_role_title_from_jd
from backend.health_analyzer import evaluate_resume_health
from backend.matcher import analyze_skill_match, analyze_keywords, analyze_projects
from backend.scoring import calculate_overall_score
from backend.suggestions import generate_suggestions

# Page Configuration
st.set_page_config(
    page_title="AI Resume Analyzer | Dual-Mode Career Copilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom SaaS CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize SQLite Database
init_db()

# Popular Role Quick-Start Presets (Shortcuts ONLY — users can enter ANY custom role!)
POPULAR_ROLES_PRESETS = {
    "UI / UX Designer": {
        "title": "UI / UX Designer",
        "description": """We are seeking a passionate UI/UX Designer to craft intuitive, user-centered digital products.

Key Responsibilities:
- Conduct user research, usability testing, and translate insights into wireframes and interactive prototypes.
- Design responsive web and mobile interfaces adhering to modern design systems and accessibility guidelines.
- Collaborate closely with frontend developers and product managers for seamless design handoff.

Requirements:
- Strong proficiency in Figma, Sketch, or Adobe XD for UI design, wireframing, and prototyping.
- Proven experience with User Research, Usability Testing, User Journeys, and Information Architecture.
- Solid understanding of HTML/CSS capabilities and design system components.
- Excellent communication, storytelling, and visual hierarchy skills."""
    },
    "Frontend Developer": {
        "title": "Frontend Developer",
        "description": """Seeking a Frontend Developer skilled in building modern, performant web applications.

Key Responsibilities:
- Build responsive, accessible user interfaces using React, TypeScript, and Tailwind CSS.
- Optimize web app performance, core web vitals, and cross-browser compatibility.
- Integrate REST APIs and GraphQL services.

Requirements:
- Strong proficiency in JavaScript, TypeScript, React, HTML5, CSS3, and Git.
- Experience with Redux/Zustand state management and Vite/Webpack build tools.
- Understanding of UI/UX design handoff and responsive web layouts."""
    },
    "Full Stack Developer": {
        "title": "Full Stack Developer",
        "description": """Looking for a Full Stack Developer to build end-to-end SaaS applications.

Key Responsibilities:
- Develop responsive frontends using React and robust backends using Node.js or FastAPI.
- Design database schemas, write optimized SQL queries, and deploy cloud infrastructure.

Requirements:
- Hands-on experience with JavaScript, TypeScript, Python, React, and Node.js/FastAPI.
- Strong knowledge of PostgreSQL, MongoDB, Redis, REST APIs, and Docker.
- Experience with Git, CI/CD, and AWS deployment."""
    },
    "Data Analyst": {
        "title": "Data Analyst",
        "description": """Seeking a Data Analyst to extract actionable business insights from complex datasets.

Key Responsibilities:
- Perform exploratory data analysis and build interactive dashboards in Tableau or Power BI.
- Write complex SQL queries, manage ETL data pipelines, and conduct A/B testing.

Requirements:
- High proficiency in SQL, Python, Pandas, NumPy, and Data Visualization.
- Experience with Tableau, Power BI, A/B Testing, and BigQuery/Snowflake.
- Excellent analytical thinking and presentation skills."""
    },
    "ML Engineer": {
        "title": "Machine Learning Engineer",
        "description": """Seeking an ML Engineer to build and deploy machine learning and NLP models.

Key Responsibilities:
- Develop machine learning pipelines, fine-tune LLMs/Transformers, and implement feature engineering.
- Deploy models as scalable REST APIs using FastAPI and Docker.

Requirements:
- Proficiency in Python, PyTorch, TensorFlow, scikit-learn, and spaCy.
- Experience with NLP, Deep Learning, Natural Language Processing, and SQL.
- Familiarity with FastAPI, Docker, and model evaluation metrics."""
    },
    "Product Designer": {
        "title": "Product Designer",
        "description": """Seeking a Product Designer to drive end-to-end user experience and visual design.

Key Responsibilities:
- Lead product design from initial user discovery and wireframing through high-fidelity visual design.
- Define and expand our design system components and interaction patterns.

Requirements:
- Expertise in Figma, User Research, Wireframing, Prototyping, and Design Systems.
- Deep understanding of interaction design, user journeys, and responsive layouts."""
    }
}


# --- Secure Browser LocalStorage JS Helpers for Zero-URL Token Persistence ---

def set_local_storage_token(token: str):
    """Store token in browser localStorage via JS bridge without exposing token in URL bar."""
    js_code = f"""
    <script>
        try {{
            window.parent.localStorage.setItem('auth_token', '{token}');
        }} catch (e) {{
            console.error('Failed to store auth token in localStorage:', e);
        }}
    </script>
    """
    st.components.v1.html(js_code, height=0, width=0)


def clear_local_storage_token():
    """Remove token from browser localStorage via JS bridge upon Logout."""
    js_code = """
    <script>
        try {
            window.parent.localStorage.removeItem('auth_token');
        } catch (e) {
            console.error('Failed to clear auth token from localStorage:', e);
        }
    </script>
    """
    st.components.v1.html(js_code, height=0, width=0)


def init_persistent_auth():
    """
    Secure Persistent Auth Initialization:
    Restores authenticated user session while ensuring the URL bar NEVER exposes JWT tokens.
    """
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user" not in st.session_state:
        st.session_state["user"] = None
    if "sidebar_collapsed" not in st.session_state:
        st.session_state["sidebar_collapsed"] = False

    # 1. If already authenticated in Streamlit session_state, ensure URL bar is 100% clean
    if st.session_state.get("authenticated") and st.session_state.get("user"):
        if "auth_token" in st.query_params:
            st.query_params.clear()
        return

    # 2. Check if a temporary query param token exists (from F5 refresh bridge)
    token = st.query_params.get("auth_token") or st.session_state.get("auth_token")
    
    if token:
        payload = decode_access_token(token)
        if payload and "user_id" in payload:
            user = get_user_by_id(payload["user_id"])
            if user:
                st.session_state["authenticated"] = True
                st.session_state["user"] = {
                    "user_id": user["user_id"],
                    "full_name": user["full_name"],
                    "email": user["email"],
                    "created_at": user["created_at"]
                }
                st.session_state["auth_token"] = token
                # IMMEDIATELY clear token from URL query params so URL bar stays clean!
                st.query_params.clear()
                return

    # 3. If unauthenticated, execute JS restore bridge to check browser localStorage on page refresh
    if not st.session_state.get("authenticated"):
        js_restore = """
        <script>
            try {
                let savedToken = window.parent.localStorage.getItem('auth_token');
                let currentUrl = new URL(window.parent.location.href);
                if (savedToken && !currentUrl.searchParams.has('auth_token')) {
                    currentUrl.searchParams.set('auth_token', savedToken);
                    window.parent.location.replace(currentUrl.toString());
                }
            } catch (e) {
                console.error('LocalStorage auth restore error:', e);
            }
        </script>
        """
        st.components.v1.html(js_restore, height=0, width=0)

    # Clear invalid/expired session
    if st.session_state.get("authenticated"):
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        if "auth_token" in st.session_state:
            del st.session_state["auth_token"]
        st.query_params.clear()


def render_login_signup():
    """Render centered authentication card for Login & Signup."""
    st.markdown("""<div style="text-align: center; margin-top: 2rem;">
<div style="font-size: 2.4rem; font-weight: 800; color: #f8fafc;">⚡ AI RESUME ANALYZER</div>
<div style="font-size: 1.05rem; color: #94a3b8; max-width: 520px; margin: 0.5rem auto 1.5rem auto;">
Evaluate your general resume health or match your resume against any target job role.
</div>
</div>""", unsafe_allow_html=True)
    
    col_left, col_center, col_right = st.columns([1, 2.2, 1])
    
    with col_center:
        auth_tab1, auth_tab2 = st.tabs(["🔑 Log In", "📝 Create Account"])
        
        with auth_tab1:
            st.markdown("#### Welcome Back")
            st.caption("Enter your email and password to access your workspace.")
            
            with st.form("login_form"):
                login_email = st.text_input("Email Address", placeholder="name@example.com")
                login_password = st.text_input("Password", type="password", placeholder="••••••••")
                submit_login = st.form_submit_button("Log In")
                
                if submit_login:
                    if not login_email or not login_password:
                        st.error("Please enter both email address and password.")
                    elif not validate_email_format(login_email):
                        st.error("Please enter a valid email address.")
                    else:
                        user = get_user_by_email(login_email)
                        if not user or not verify_password(login_password, user["password_hash"]):
                            st.error("Invalid email or password.")
                        else:
                            token = create_access_token({"user_id": user["user_id"], "email": user["email"]})
                            st.session_state["authenticated"] = True
                            st.session_state["user"] = {
                                "user_id": user["user_id"],
                                "full_name": user["full_name"],
                                "email": user["email"],
                                "created_at": user["created_at"]
                            }
                            st.session_state["auth_token"] = token
                            
                            # Store token securely in localStorage and keep URL 100% clean
                            set_local_storage_token(token)
                            st.query_params.clear()
                            
                            st.success(f"Welcome back, {user['full_name']}!")
                            time.sleep(0.3)
                            st.rerun()
                            
        with auth_tab2:
            st.markdown("#### Create Your Account")
            st.caption("Start analyzing your resume with personalized history tracking.")
            
            with st.form("signup_form"):
                signup_name = st.text_input("Full Name", placeholder="Jane Doe")
                signup_email = st.text_input("Email Address", placeholder="jane@example.com")
                signup_password = st.text_input("Password (min 6 chars)", type="password", placeholder="••••••••")
                signup_confirm = st.text_input("Confirm Password", type="password", placeholder="••••••••")
                submit_signup = st.form_submit_button("Create Account")
                
                if submit_signup:
                    if not signup_name or not signup_email or not signup_password or not signup_confirm:
                        st.error("All fields are required.")
                    elif not validate_email_format(signup_email):
                        st.error("Please enter a valid email address.")
                    elif signup_password != signup_confirm:
                        st.error("Password and Confirm Password do not match.")
                    else:
                        is_valid_pwd, pwd_msg = validate_password_strength(signup_password)
                        if not is_valid_pwd:
                            st.error(pwd_msg)
                        else:
                            try:
                                pwd_hash = hash_password(signup_password)
                                new_user = create_user(signup_name, signup_email, pwd_hash)
                                token = create_access_token({"user_id": new_user["user_id"], "email": new_user["email"]})
                                st.session_state["authenticated"] = True
                                st.session_state["user"] = new_user
                                st.session_state["auth_token"] = token
                                
                                set_local_storage_token(token)
                                st.query_params.clear()
                                
                                st.success("Account created successfully!")
                                time.sleep(0.3)
                                st.rerun()
                            except ValueError as ve:
                                st.error(str(ve))
                            except Exception as e:
                                st.error(f"Signup failed: {str(e)}")


def render_user_dashboard(user: Dict[str, Any]):
    """Render personalized dashboard metrics for logged-in user."""
    user_id = user["user_id"]
    stats = get_user_stats(user_id)
    
    st.markdown(f"""<div class="welcome-banner">
<div class="welcome-title">Welcome back, {user['full_name']} 👋</div>
<div class="welcome-sub">Let's make your resume stronger. Evaluate general resume health or match against any job role.</div>
</div>""", unsafe_allow_html=True)
    
    col_card1, col_card2 = st.columns(2)
    
    with col_card1:
        st.markdown("""<div class="mode-select-card">
<div style="font-size: 2rem; margin-bottom: 0.4rem;">📄</div>
<div style="font-size: 1.25rem; font-weight: 800; color: #f8fafc;">RESUME HEALTH</div>
<div style="font-size: 0.92rem; color: #94a3b8; margin: 0.4rem 0 1.2rem 0; min-height: 48px;">
Get an overall assessment of your resume readiness, structure, formatting, skills range, and bullet impact without targeting a specific job.
</div>
</div>""", unsafe_allow_html=True)
        if st.button("📄 Review My Resume Health", key="dash_btn_health"):
            st.session_state["active_analysis_mode"] = "resume_health"
            st.session_state["nav_target"] = "Resume Analyzer"
            st.rerun()
            
    with col_card2:
        st.markdown("""<div class="mode-select-card">
<div style="font-size: 2rem; margin-bottom: 0.4rem;">🎯</div>
<div style="font-size: 1.25rem; font-weight: 800; color: #f8fafc;">JOB MATCH</div>
<div style="font-size: 0.92rem; color: #94a3b8; margin: 0.4rem 0 1.2rem 0; min-height: 48px;">
Compare your resume against any target job role or pasted description to evaluate skill overlap, keyword coverage, and recommendations.
</div>
</div>""", unsafe_allow_html=True)
        if st.button("🎯 Match a Target Job", key="dash_btn_match"):
            st.session_state["active_analysis_mode"] = "job_match"
            st.session_state["nav_target"] = "Resume Analyzer"
            st.rerun()

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("### Evaluation Summary")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="dash-stat-card">
<div class="dash-stat-lbl">Total Analyses</div>
<div class="dash-stat-val" style="color: #818cf8;">{stats['total_analyses']}</div>
</div>""", unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""<div class="dash-stat-card">
<div class="dash-stat-lbl">Resume Health Reviews</div>
<div class="dash-stat-val" style="color: #38bdf8;">{stats['health_count']}</div>
</div>""", unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""<div class="dash-stat-card">
<div class="dash-stat-lbl">Job Match Evaluations</div>
<div class="dash-stat-val" style="color: #34d399;">{stats['job_match_count']}</div>
</div>""", unsafe_allow_html=True)
        
    with c4:
        recent_date = stats['recent_time'][:10] if stats['recent_time'] != "N/A" else "None"
        st.markdown(f"""<div class="dash-stat-card">
<div class="dash-stat-lbl">Last Activity</div>
<div class="dash-stat-val" style="font-size: 1.35rem; color: #fbbf24;">{recent_date}</div>
</div>""", unsafe_allow_html=True)


def render_user_profile(user: Dict[str, Any]):
    """Render user profile details cleanly using native Streamlit UI elements."""
    user_id = user["user_id"]
    stats = get_user_stats(user_id)
    
    st.markdown("## Your Profile & Account")
    st.markdown("---")
    
    col_p1, col_p2 = st.columns([1.5, 1])
    
    with col_p1:
        st.subheader(f"👤 {user['full_name']}")
        st.caption(f"Email: {user['email']}")
        
        st.markdown("---")
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("User ID", user['user_id'][:12])
        with m2:
            st.metric("Account Created", user['created_at'][:10])
        with m3:
            st.metric("Total Evaluations", stats['total_analyses'])
            
    with col_p2:
        st.subheader("Account Controls")
        st.caption("Manage your logged-in session.")
        if st.button("🚪 Log Out of Account"):
            clear_local_storage_token()
            st.session_state["authenticated"] = False
            st.session_state["user"] = None
            if "auth_token" in st.session_state:
                del st.session_state["auth_token"]
            st.query_params.clear()
            if "current_analysis" in st.session_state:
                del st.session_state["current_analysis"]
            st.rerun()


def render_sidebar():
    """Render Sidebar Navigation with clean native navigation layout."""
    user = st.session_state.get("user", {})
    full_name = user.get("full_name", "User")
    
    st.sidebar.markdown(f"""<div style="margin-bottom: 1.2rem; margin-top: 0.5rem;">
<div style="font-size: 1.3rem; font-weight: 800; color: #f8fafc;">⚡ AI Resume Analyzer</div>
<div style="font-size: 0.82rem; color: #34d399; margin-top: 0.3rem;">👤 Signed in as <strong>{full_name}</strong></div>
</div>""", unsafe_allow_html=True)
    
    st.sidebar.markdown("<div style='font-size:0.7rem; font-weight:700; color:#64748b; letter-spacing:1px;'>WORKSPACE</div>", unsafe_allow_html=True)
    
    nav_options = ["Dashboard", "Resume Analyzer", "Analysis History", "Profile", "How It Works"]
    icon_map = {
        "Dashboard": "📊 Dashboard",
        "Resume Analyzer": "⚡ Resume Analyzer",
        "Analysis History": "📜 Analysis History",
        "Profile": "👤 Profile",
        "How It Works": "ℹ️ How It Works"
    }
    
    default_index = 0
    if "nav_target" in st.session_state:
        target = st.session_state.pop("nav_target")
        if target in nav_options:
            default_index = nav_options.index(target)

    nav_selection = st.sidebar.radio(
        "Navigation",
        nav_options,
        format_func=lambda x: icon_map[x],
        index=default_index,
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout Account"):
        clear_local_storage_token()
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        if "auth_token" in st.session_state:
            del st.session_state["auth_token"]
        st.query_params.clear()
        if "current_analysis" in st.session_state:
            del st.session_state["current_analysis"]
        st.rerun()
        
    return nav_selection


def render_resume_health_results(res: Dict[str, Any]):
    """Render comprehensive dashboard for RESUME HEALTH evaluation results."""
    st.markdown("""<div style="margin-top: 1.5rem; margin-bottom: 1.2rem;">
<h2 style="font-size: 1.8rem; font-weight: 800; color: #f8fafc; margin-bottom: 0.2rem;">📄 Resume Health Analysis</h2>
<div style="color: #94a3b8; font-size: 1rem;">Overall resume strength, ATS readiness, and diagnostic feedback.</div>
</div>""", unsafe_allow_html=True)
    
    health_info = res.get("health_info", {})
    overall_score = health_info.get("overall_health_score", 0.0)
    status_text = health_info.get("status", "Good")
    filename = res.get("filename", "resume.pdf")
    
    col_score, col_cats = st.columns([1, 2])
    
    with col_score:
        st.markdown(f"""<div class="score-hero-container" style="text-align: center;">
<div style="font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.8rem;">Resume Health Score</div>
<div class="score-display-ring">
<div class="score-num">{overall_score:.0f}</div>
<div class="score-pct">/ 100</div>
</div>
<div class="tier-tag tier-strong" style="margin-top: 0.8rem;">{status_text}</div>
<div style="margin-top: 1rem; font-size: 0.85rem; color: #94a3b8;">
File: <code style="color: #818cf8;">{filename}</code>
</div>
</div>""", unsafe_allow_html=True)
        
    with col_cats:
        st.markdown("#### Dimension Diagnostic Sub-Scores")
        categories = health_info.get("category_scores", {})
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="submetric-card">
<div class="submetric-lbl">ATS Readiness</div>
<div class="submetric-val" style="color: #34d399;">{categories.get('ATS Readiness', 0):.1f}%</div>
<div style="font-size: 0.8rem; color: #94a3b8;">Standard headings & text structure</div>
</div>""", unsafe_allow_html=True)
            
            st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""<div class="submetric-card">
<div class="submetric-lbl">Structure & Formatting</div>
<div class="submetric-val" style="color: #818cf8;">{categories.get('Structure & Formatting', 0):.1f}%</div>
<div style="font-size: 0.8rem; color: #94a3b8;">Section distribution & word balance</div>
</div>""", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)

            st.markdown(f"""<div class="submetric-card">
<div class="submetric-lbl">Projects Detail</div>
<div class="submetric-val" style="color: #38bdf8;">{categories.get('Projects Detail', 0):.1f}%</div>
<div style="font-size: 0.8rem; color: #94a3b8;">Technical project presence</div>
</div>""", unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""<div class="submetric-card">
<div class="submetric-lbl">Content Quality</div>
<div class="submetric-val" style="color: #fbbf24;">{categories.get('Content Quality', 0):.1f}%</div>
<div style="font-size: 0.8rem; color: #94a3b8;">Action verbs & quantifiable metrics</div>
</div>""", unsafe_allow_html=True)
            
            st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
            
            st.markdown(f"""<div class="submetric-card">
<div class="submetric-lbl">Skills Range</div>
<div class="submetric-val" style="color: #a78bfa;">{categories.get('Skills Range', 0):.1f}%</div>
<div style="font-size: 0.8rem; color: #94a3b8;">Taxonomy & breadth of tools</div>
</div>""", unsafe_allow_html=True)

            st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)

            st.markdown(f"""<div class="submetric-card">
<div class="submetric-lbl">Experience Depth</div>
<div class="submetric-val" style="color: #f472b6;">{categories.get('Experience Depth', 0):.1f}%</div>
<div style="font-size: 0.8rem; color: #94a3b8;">Work history & chronology</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    
    col_str, col_imp = st.columns(2)
    
    with col_str:
        st.markdown("### Top Strengths")
        strengths = health_info.get("strengths", [])
        for s in strengths:
            st.markdown(f"- ✅ {s}")
            
    with col_imp:
        st.markdown("### Areas to Improve")
        areas = health_info.get("areas_to_improve", [])
        for a in areas:
            st.markdown(f"- ⚠️ {a}")
            
    st.markdown("---")
    st.markdown("### Recommended Next Steps")
    steps = health_info.get("recommended_next_steps", [])
    for idx, step in enumerate(steps, 1):
        st.markdown(f"**0{idx}.** {step}")


def render_job_match_results(res: Dict[str, Any]):
    """Render comprehensive dashboard for JOB MATCH evaluation results."""
    st.markdown("""<div style="margin-top: 1.5rem; margin-bottom: 1.2rem;">
<h2 style="font-size: 1.8rem; font-weight: 800; color: #f8fafc; margin-bottom: 0.2rem;">🎯 Target Job Match Analysis</h2>
<div style="color: #94a3b8; font-size: 1rem;">Comparison against target role requirements.</div>
</div>""", unsafe_allow_html=True)
    
    score_data = res.get("score_breakdown", {})
    overall = score_data.get("overall_score", 0.0)
    target_role = res.get("target_role", "Target Role")
    filename = res.get("filename", "resume.pdf")
    
    tier_class = "tier-strong"
    tier_label = "Strong Match"
    if overall >= 85:
        tier_class, tier_label = "tier-excellent", "Excellent Match"
    elif overall >= 60:
        tier_class, tier_label = "tier-moderate", "Moderate Match"
    elif overall < 60:
        tier_class, tier_label = "tier-needs-work", "Needs Improvement"
        
    col_score, col_expl = st.columns([1, 2])
    
    with col_score:
        st.markdown(f"""<div class="score-hero-container" style="text-align: center;">
<div style="font-size: 0.8rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.8rem;">Match Score</div>
<div class="score-display-ring">
<div class="score-num">{overall:.0f}</div>
<div class="score-pct">%</div>
</div>
<div class="tier-tag {tier_class}">{tier_label}</div>
<div style="margin-top: 1rem; font-size: 0.85rem; color: #94a3b8;">
Target Role: <strong style="color: #f8fafc;">{target_role}</strong><br>
File: <code style="color: #818cf8;">{filename}</code>
</div>
</div>""", unsafe_allow_html=True)
        
    with col_expl:
        st.markdown("#### Score Component Breakdown")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="submetric-card">
<div class="submetric-lbl">Skill Match (40% Weight)</div>
<div class="submetric-val" style="color: #34d399;">{score_data.get('skill_score', 0):.1f}%</div>
<div style="font-size: 0.8rem; color: #94a3b8;">Direct match against required technologies</div>
</div>""", unsafe_allow_html=True)
            st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
            st.markdown(f"""<div class="submetric-card">
<div class="submetric-lbl">Project Relevance (20% Weight)</div>
<div class="submetric-val" style="color: #818cf8;">{score_data.get('experience_score', 0):.1f}%</div>
<div style="font-size: 0.8rem; color: #94a3b8;">Project alignment & section presence</div>
</div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="submetric-card">
<div class="submetric-lbl">Keyword Coverage (25% Weight)</div>
<div class="submetric-val" style="color: #fbbf24;">{score_data.get('keyword_score', 0):.1f}%</div>
<div style="font-size: 0.8rem; color: #94a3b8;">High-frequency job description keyphrases</div>
</div>""", unsafe_allow_html=True)
            st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
            st.markdown(f"""<div class="submetric-card">
<div class="submetric-lbl">Semantic Similarity (15% Weight)</div>
<div class="submetric-val" style="color: #38bdf8;">{score_data.get('semantic_score', 0):.1f}%</div>
<div style="font-size: 0.8rem; color: #94a3b8;">TF-IDF vector cosine similarity</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Skills Alignment", "🔑 Keyword Coverage", "🛠️ Projects & Section Audit", "💡 AI Recommendations"])
    
    with tab1:
        sa = res.get("skills_analysis", {})
        matched_skills = sa.get("matched_skills", [])
        missing_skills = sa.get("missing_skills", [])
        col_m, col_u = st.columns(2)
        with col_m:
            st.markdown(f"#### Matched Skills ({len(matched_skills)})")
            if matched_skills:
                chips_html = '<div class="chip-group">' + "".join([f'<span class="skill-chip matched">✓ {s["name"]}</span>' for s in matched_skills]) + '</div>'
                st.markdown(chips_html, unsafe_allow_html=True)
            else:
                st.info("No explicit skills from the job description were detected in your resume.")
        with col_u:
            st.markdown(f"#### Missing Skills ({len(missing_skills)})")
            if missing_skills:
                chips_html = '<div class="chip-group">' + "".join([f'<span class="skill-chip missing">✗ {s["name"]}</span>' for s in missing_skills]) + '</div>'
                st.markdown(chips_html, unsafe_allow_html=True)
            else:
                st.success("All explicit technical skills requested in the job description are present on your resume!")

    with tab2:
        ka = res.get("keywords_analysis", {})
        matched_kw = ka.get("matched_keywords", [])
        missing_kw = ka.get("missing_keywords", [])
        ck1, ck2 = st.columns(2)
        with ck1:
            st.markdown(f"#### Matched Keyphrases ({len(matched_kw)})")
            for kw in matched_kw:
                st.markdown(f"- ✅ **`{kw}`**")
        with ck2:
            st.markdown(f"#### Missing Keyphrases ({len(missing_kw)})")
            for kw in missing_kw:
                st.markdown(f"- ❌ `{kw}`")

    with tab3:
        sec_col, proj_col = st.columns([1, 2])
        with sec_col:
            st.markdown("#### Detected Sections")
            sections = res.get("parse_info", {}).get("detected_sections", [])
            for sec in sections:
                status_class = "audit-status-yes" if sec["present"] else "audit-status-no"
                status_text = "✓ Present" if sec["present"] else "✗ Missing"
                st.markdown(f"""<div class="audit-pill"><span class="audit-name">{sec['name']}</span><span class="{status_class}">{status_text}</span></div>""", unsafe_allow_html=True)
        with proj_col:
            st.markdown("#### Project Relevance Analysis")
            projs = res.get("project_analysis", [])
            if projs:
                for p in projs:
                    terms_str = ", ".join(p.get("matched_terms", [])) if p.get("matched_terms") else "General domain alignment"
                    with st.expander(f"📌 {p['title']} — Relevance Score: {p['relevance_score']}%"):
                        st.markdown(f"**Matched Role Terms**: {terms_str}")
            else:
                st.info("No explicit project section entries were parsed.")

    with tab4:
        suggs = res.get("suggestions", [])
        if suggs:
            for idx, s in enumerate(suggs, start=1):
                st.markdown(f"""<div class="rec-card"><div class="rec-num">0{idx}</div><div class="rec-content"><div class="rec-title">{s['title']}</div><div class="rec-msg">{s['message']}</div></div></div>""", unsafe_allow_html=True)
        else:
            st.success("Your resume aligns exceptionally well with the target job description!")


def main():
    init_persistent_auth()
        
    if not st.session_state["authenticated"] or not st.session_state["user"]:
        render_login_signup()
        return
        
    current_user = st.session_state["user"]
    nav_selection = render_sidebar()
    
    if nav_selection == "Dashboard":
        render_user_dashboard(current_user)
        
    elif nav_selection == "Resume Analyzer":
        st.markdown("""<div class="hero-wrapper">
<div class="hero-pill">⚡ Dual-Mode AI Resume Analyzer</div>
<div class="hero-heading">Choose How You'd Like to Analyze Your Resume</div>
<div class="hero-subheading">
Evaluate general Resume Health without targeting a job, or run a Job Match analysis against any target role.
</div>
</div>""", unsafe_allow_html=True)
        
        mode = st.radio(
            "Select Analysis Mode:",
            ["📄 Resume Health (No Job Required)", "🎯 Job Match (Target Role / JD)"],
            index=0 if st.session_state.get("active_analysis_mode") == "resume_health" else 1,
            horizontal=True
        )
        
        if "Resume Health" in mode:
            st.session_state["active_analysis_mode"] = "resume_health"
            st.markdown("### Upload Resume for Health Review")
            uploaded_resume = st.file_uploader(
                "Drag & drop your PDF resume",
                type=["pdf"],
                key="health_pdf_uploader",
                help="Upload PDF (Max 10MB)"
            )
            
            if uploaded_resume:
                st.markdown(f"✓ Ready: `{uploaded_resume.name}` ({round(uploaded_resume.size / 1024, 1)} KB)")
                
            if st.button("🚀 Analyze Resume Health"):
                if not uploaded_resume:
                    st.warning("⚠️ Please upload a PDF resume first.")
                else:
                    try:
                        content = uploaded_resume.read()
                        parse_info = parse_resume(content, filename=uploaded_resume.name)
                        health_diag = evaluate_resume_health(parse_info)
                        
                        import datetime
                        res_dict = {
                            "analysis_id": f"anl_{os.urandom(4).hex()}",
                            "analysis_type": "resume_health",
                            "user_id": current_user["user_id"],
                            "filename": uploaded_resume.name,
                            "target_role": "General Resume Health",
                            "timestamp": datetime.datetime.now().isoformat(),
                            "parse_info": parse_info.model_dump(),
                            "health_info": health_diag
                        }
                        save_analysis(res_dict, user_id=current_user["user_id"])
                        st.session_state["current_analysis"] = res_dict
                        st.success("Resume Health analysis complete!")
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        
            if st.session_state.get("current_analysis") and st.session_state["current_analysis"].get("analysis_type") == "resume_health":
                render_resume_health_results(st.session_state["current_analysis"])

        else:
            st.session_state["active_analysis_mode"] = "job_match"
            st.markdown("### Upload Resume & Match to a Job")
            
            col_r, col_j = st.columns(2)
            
            with col_r:
                uploaded_resume = st.file_uploader(
                    "Upload your PDF resume",
                    type=["pdf"],
                    key="match_pdf_uploader"
                )
                if uploaded_resume:
                    st.markdown(f"✓ Ready: `{uploaded_resume.name}` ({round(uploaded_resume.size / 1024, 1)} KB)")

            with col_j:
                st.markdown("#### How would you like to provide the target job?")
                match_input_tab1, match_input_tab2, match_input_tab3 = st.tabs([
                    "1. Role Search", "2. Paste JD", "3. Upload JD File"
                ])
                
                selected_role = ""
                final_jd_text = ""
                
                with match_input_tab1:
                    role_text_entry = st.text_input("🔍 Search or enter ANY job role (e.g. UX Researcher, DevOps, Product Manager):", placeholder="e.g., Cybersecurity Analyst")
                    st.caption("Popular Role Shortcuts (Click to populate sample JD):")
                    
                    pop_cols = st.columns(3)
                    for idx, (p_role, p_data) in enumerate(POPULAR_ROLES_PRESETS.items()):
                        col_target = pop_cols[idx % 3]
                        if col_target.button(p_role, key=f"pop_btn_{idx}"):
                            st.session_state["custom_role_title"] = p_data["title"]
                            st.session_state["custom_jd_text"] = p_data["description"]
                            
                    if role_text_entry:
                        st.session_state["custom_role_title"] = role_text_entry
                        
                    role_input = st.text_input("Target Role Title", key="custom_role_title")
                    jd_input = st.text_area("Job Description", height=150, key="custom_jd_text")
                    selected_role = role_input
                    final_jd_text = jd_input
                    
                with match_input_tab2:
                    pasted_jd = st.text_area("Paste full Job Description text here:", height=200, key="pasted_jd_key")
                    optional_role = st.text_input("Target Role Title (Optional)", placeholder="Auto-extracted if left blank", key="optional_role_key")
                    if pasted_jd:
                        final_jd_text = pasted_jd
                        selected_role = optional_role or extract_role_title_from_jd(pasted_jd) or "Target Job Role"

                with match_input_tab3:
                    uploaded_jd_file = st.file_uploader("Upload Job Description (PDF/TXT)", type=["pdf", "txt"], key="jd_file_uploader")
                    uploaded_role = st.text_input("Target Role Title (Optional)", key="uploaded_role_key")
                    if uploaded_jd_file:
                        try:
                            jd_bytes = uploaded_jd_file.read()
                            if uploaded_jd_file.name.endswith(".pdf"):
                                parsed_jd = parse_resume(jd_bytes, filename=uploaded_jd_file.name)
                                final_jd_text = parsed_jd.clean_text
                            else:
                                final_jd_text = jd_bytes.decode("utf-8", errors="ignore")
                            selected_role = uploaded_role or extract_role_title_from_jd(final_jd_text) or "Target Job Role"
                            st.info(f"Loaded JD Document: {uploaded_jd_file.name}")
                        except Exception as ex:
                            st.error(f"Failed to read JD file: {str(ex)}")

            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            if st.button("🚀 Analyze Job Match"):
                if not uploaded_resume:
                    st.warning("⚠️ Please upload a PDF resume.")
                elif not final_jd_text or not final_jd_text.strip():
                    st.warning("⚠️ Please enter, paste, or upload a Job Description.")
                else:
                    try:
                        content = uploaded_resume.read()
                        parse_info = parse_resume(content, filename=uploaded_resume.name)
                        skills_res = analyze_skill_match(parse_info.clean_text, final_jd_text)
                        keywords_res = analyze_keywords(parse_info.clean_text, final_jd_text)
                        projects_res = analyze_projects(parse_info.clean_text, final_jd_text, parse_info.sections)
                        semantic_sim = calculate_semantic_similarity(parse_info.clean_text, final_jd_text)
                        
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
                            target_role=selected_role or "Target Job Role",
                            jd_text=final_jd_text
                        )
                        
                        import datetime
                        res_dict = {
                            "analysis_id": f"anl_{os.urandom(4).hex()}",
                            "analysis_type": "job_match",
                            "user_id": current_user["user_id"],
                            "filename": uploaded_resume.name,
                            "target_role": selected_role or "Target Job Role",
                            "timestamp": datetime.datetime.now().isoformat(),
                            "parse_info": parse_info.model_dump(),
                            "score_breakdown": score_breakdown.model_dump(),
                            "skills_analysis": skills_res.model_dump(),
                            "keywords_analysis": keywords_res.model_dump(),
                            "project_analysis": [p.model_dump() for p in projects_res],
                            "suggestions": [s.model_dump() for s in suggestions]
                        }
                        
                        save_analysis(res_dict, user_id=current_user["user_id"])
                        st.session_state["current_analysis"] = res_dict
                        st.success("Job Match analysis complete!")
                    except Exception as e:
                        st.error(f"Job match failed: {str(e)}")
                        
            if st.session_state.get("current_analysis") and st.session_state["current_analysis"].get("analysis_type") == "job_match":
                render_job_match_results(st.session_state["current_analysis"])

    elif nav_selection == "Analysis History":
        st.markdown("## Your Analysis History")
        st.markdown("Review and reload your previous Resume Health & Job Match evaluations.")
        st.markdown("---")
        
        user_history = get_history(user_id=current_user["user_id"], limit=50)
        
        if user_history:
            df = pd.DataFrame(user_history)
            df_display = df[["analysis_id", "analysis_type", "filename", "target_role", "overall_score", "tier", "timestamp"]]
            st.dataframe(df_display, use_container_width=True)
            
            col_load, col_clear = st.columns([2, 1])
            with col_load:
                sel_id = st.selectbox("Select Analysis ID to inspect:", df["analysis_id"].tolist())
                if st.button("Load Selected Analysis"):
                    record = get_analysis_by_id(sel_id, user_id=current_user["user_id"])
                    if record:
                        st.session_state["current_analysis"] = record
                        st.success("Loaded history record! View results below.")
                        if record.get("analysis_type") == "resume_health":
                            render_resume_health_results(record)
                        else:
                            render_job_match_results(record)
                        
            with col_clear:
                st.markdown("#### Clear History")
                confirm_clear = st.checkbox("Confirm permanent deletion of your history records", key="confirm_clear_chk")
                if st.button("🗑️ Clear My History"):
                    if confirm_clear:
                        deleted_count = clear_user_history(user_id=current_user["user_id"])
                        if "current_analysis" in st.session_state:
                            del st.session_state["current_analysis"]
                        st.success(f"Cleared {deleted_count} history records.")
                        time.sleep(0.3)
                        st.rerun()
                    else:
                        st.warning("Please check the confirmation box before clearing history.")
        else:
            st.info("No saved evaluation history found for your account.")

    elif nav_selection == "Profile":
        render_user_profile(current_user)

    elif nav_selection == "How It Works":
        st.markdown("## How AI Resume Analyzer Works")
        st.markdown("Learn how our AI evaluates Resume Health and matches resumes against target job roles.")
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 📄 Mode A: Resume Health")
            st.markdown("""
            - **No Job Role Required**: Evaluates general resume quality and ATS readiness.
            - **ATS Readiness**: Checks standard section headings and text structure.
            - **Content Quality**: Evaluates action verbs and quantifiable metric impact.
            - **Skills Range**: Scans taxonomy coverage across tools, languages, and frameworks.
            """)
        with c2:
            st.markdown("### 🎯 Mode B: Job Match")
            st.markdown("""
            - **Any Custom Job Title**: Search or enter any role (e.g. UX Researcher, DevOps, AI Engineer).
            - **Multiple Input Options**: Role search, pasted JD text, or uploaded JD document.
            - **Explainable Match Scoring**: Weighted score based on Skills (40%), Keywords (25%), Projects (20%), and Cosine Similarity (15%).
            """)

        st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
        with st.expander("🛠️ Technical Architecture & Stack"):
            st.markdown("""
            - **Frontend**: Streamlit, Custom SaaS CSS Design System
            - **Backend API**: Python 3.11, FastAPI, Pydantic v2
            - **Security**: bcrypt password hashing, persistent JWT token restoration, user-isolated SQLite queries
            - **NLP Engine**: spaCy (`en_core_web_sm`), scikit-learn (TF-IDF vectorizer, Cosine Similarity)
            - **PDF Extraction**: PyMuPDF (`fitz`)
            """)


if __name__ == "__main__":
    main()
