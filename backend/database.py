"""
SQLite Database manager with multi-user user management, dual mode support (Resume Health & Job Match),
and scoped user-history isolation.
"""

import sqlite3
import json
import os
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "analyzer_history.db")


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    
    # 2. Create analysis_history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT UNIQUE NOT NULL,
            analysis_type TEXT NOT NULL DEFAULT 'job_match',
            user_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            target_role TEXT,
            overall_score REAL NOT NULL,
            tier TEXT NOT NULL,
            matched_skills_count INTEGER NOT NULL DEFAULT 0,
            missing_skills_count INTEGER NOT NULL DEFAULT 0,
            timestamp TEXT NOT NULL,
            details_json TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        );
    """)
    
    # Schema migration checks for legacy databases
    cursor.execute("PRAGMA table_info(analysis_history)")
    columns = [row[1] for row in cursor.fetchall()]
    if "user_id" not in columns:
        cursor.execute("ALTER TABLE analysis_history ADD COLUMN user_id TEXT DEFAULT 'legacy_user'")
    if "analysis_type" not in columns:
        cursor.execute("ALTER TABLE analysis_history ADD COLUMN analysis_type TEXT DEFAULT 'job_match'")
        
    conn.commit()
    conn.close()


# --- User Authentication DB Queries ---

def create_user(full_name: str, email: str, password_hash: str) -> Dict[str, Any]:
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    email_clean = email.strip().lower()
    
    cursor.execute("SELECT id FROM users WHERE lower(email) = ?", (email_clean,))
    if cursor.fetchone():
        conn.close()
        raise ValueError("An account with this email address already exists.")
        
    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    created_at = datetime.now().isoformat()
    
    cursor.execute("""
        INSERT INTO users (user_id, full_name, email, password_hash, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, full_name.strip(), email_clean, password_hash, created_at))
    
    conn.commit()
    conn.close()
    
    return {
        "user_id": user_id,
        "full_name": full_name.strip(),
        "email": email_clean,
        "created_at": created_at,
        "total_analyses": 0
    }


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE lower(email) = ?", (email.strip().lower(),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def get_user_stats(user_id: str) -> Dict[str, Any]:
    """Retrieve summary dashboard metrics strictly for a single user."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) as total_count, 
               AVG(overall_score) as avg_score, 
               MAX(overall_score) as max_score,
               MAX(timestamp) as recent_time,
               SUM(CASE WHEN analysis_type = 'resume_health' THEN 1 ELSE 0 END) as health_count,
               SUM(CASE WHEN analysis_type = 'job_match' THEN 1 ELSE 0 END) as match_count
        FROM analysis_history
        WHERE user_id = ?
    """, (user_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    total = row["total_count"] if row and row["total_count"] else 0
    avg_sc = round(row["avg_score"], 1) if row and row["avg_score"] is not None else 0.0
    best_sc = round(row["max_score"], 1) if row and row["max_score"] is not None else 0.0
    recent = row["recent_time"] if row and row["recent_time"] else "N/A"
    health_cnt = row["health_count"] if row and row["health_count"] else 0
    match_cnt = row["match_count"] if row and row["match_count"] else 0
    
    return {
        "total_analyses": total,
        "avg_score": avg_sc,
        "best_score": best_sc,
        "recent_time": recent,
        "health_count": health_cnt,
        "job_match_count": match_cnt
    }


# --- User-Scoped Analysis History DB Queries ---

def save_analysis(analysis_data: Dict[str, Any], user_id: str) -> str:
    """Save an analysis record strictly linked to the authenticated user's user_id."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    analysis_id = analysis_data.get("analysis_id", f"anl_{int(datetime.now().timestamp())}")
    analysis_type = analysis_data.get("analysis_type", "job_match")
    filename = analysis_data.get("filename", "resume.pdf")
    
    if analysis_type == "resume_health":
        target_role = "General Resume Health"
        overall_score = analysis_data.get("health_info", {}).get("overall_health_score", 0.0)
        tier = analysis_data.get("health_info", {}).get("status", "N/A")
        matched_count = len(analysis_data.get("health_info", {}).get("extracted_skills", []))
        missing_count = 0
    else:
        target_role = analysis_data.get("target_role", "Target Role")
        overall_score = analysis_data.get("score_breakdown", {}).get("overall_score", 0.0)
        tier = analysis_data.get("score_breakdown", {}).get("tier", "N/A")
        matched_count = len(analysis_data.get("skills_analysis", {}).get("matched_skills", []))
        missing_count = len(analysis_data.get("skills_analysis", {}).get("missing_skills", []))
        
    timestamp = analysis_data.get("timestamp", datetime.now().isoformat())
    
    analysis_data["user_id"] = user_id
    analysis_data["analysis_type"] = analysis_type
    details_json = json.dumps(analysis_data)

    cursor.execute("""
        INSERT OR REPLACE INTO analysis_history 
        (analysis_id, analysis_type, user_id, filename, target_role, overall_score, tier, matched_skills_count, missing_skills_count, timestamp, details_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (analysis_id, analysis_type, user_id, filename, target_role, overall_score, tier, matched_count, missing_count, timestamp, details_json))
    
    conn.commit()
    conn.close()
    return analysis_id


def get_history(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve history records strictly belonging to user_id."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, analysis_id, analysis_type, user_id, filename, target_role, overall_score, tier, 
               matched_skills_count, missing_skills_count, timestamp
        FROM analysis_history
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_analysis_by_id(analysis_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve single analysis details strictly if owned by user_id."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT details_json FROM analysis_history WHERE analysis_id = ? AND user_id = ?
    """, (analysis_id, user_id))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row["details_json"])
    return None


def clear_user_history(user_id: str) -> int:
    """Delete all analysis records belonging strictly to user_id."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM analysis_history WHERE user_id = ?", (user_id,))
    deleted_count = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted_count
