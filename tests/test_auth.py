"""
Comprehensive unit tests for Multi-User Authentication, Password Hashing, User-Specific Data Isolation, and IDOR Security Controls.
"""

import pytest
import os
import sqlite3
from fastapi.testclient import TestClient

from backend.main import app
from backend.auth import (
    hash_password, verify_password, validate_email_format, validate_password_strength,
    create_access_token, decode_access_token
)
from backend.database import (
    init_db, create_user, get_user_by_email, get_user_by_id, get_user_stats,
    save_analysis, get_history, get_analysis_by_id, clear_user_history
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    """Fixture to ensure test suite operates on an isolated temporary SQLite database."""
    test_db_file = os.path.join(tmp_path, "test_analyzer_history.db")
    monkeypatch.setattr("backend.database.DB_PATH", test_db_file)
    init_db()


def test_password_hashing():
    raw_pwd = "MySecretPassword123"
    pwd_hash = hash_password(raw_pwd)
    
    assert pwd_hash != raw_pwd
    assert pwd_hash.startswith("$2b$") or pwd_hash.startswith("$2a$")
    assert verify_password(raw_pwd, pwd_hash) is True
    assert verify_password("WrongPassword", pwd_hash) is False


def test_email_validation():
    assert validate_email_format("user@example.com") is True
    assert validate_email_format("john.doe@company.co.in") is True
    assert validate_email_format("invalid-email") is False
    assert validate_email_format("@missingusername.com") is False
    assert validate_email_format("") is False


def test_signup_success_and_duplicate_prevention():
    user = create_user("Test User", "test@example.com", hash_password("pass123"))
    assert user["full_name"] == "Test User"
    assert user["email"] == "test@example.com"
    assert user["user_id"].startswith("usr_")
    
    # Attempt duplicate signup
    with pytest.raises(ValueError, match="already exists"):
        create_user("Another Name", "TEST@EXAMPLE.COM", hash_password("pass123"))


def test_user_authentication_flow():
    pwd_hash = hash_password("secure123")
    user = create_user("Alice Wonderland", "alice@domain.com", pwd_hash)
    
    fetched = get_user_by_email("alice@domain.com")
    assert fetched is not None
    assert fetched["user_id"] == user["user_id"]
    assert verify_password("secure123", fetched["password_hash"]) is True
    assert verify_password("wrong", fetched["password_hash"]) is False


def test_jwt_token_generation_and_decoding():
    payload = {"user_id": "usr_999", "email": "jwt@test.com"}
    token = create_access_token(payload)
    assert isinstance(token, str)
    
    decoded = decode_access_token(token)
    assert decoded["user_id"] == "usr_999"
    assert decoded["email"] == "jwt@test.com"


def test_multi_user_history_isolation():
    """
    MANDATORY MULTI-USER ISOLATION TEST:
    Verify User A sees ONLY Analysis A, and User B sees ONLY Analysis B.
    """
    user_a = create_user("User A", "usera@test.com", hash_password("passA123"))
    user_b = create_user("User B", "userb@test.com", hash_password("passB123"))
    
    # Save Analysis A under User A
    analysis_a_data = {
        "analysis_id": "anl_user_a_001",
        "filename": "resume_user_a.pdf",
        "target_role": "Frontend Developer",
        "score_breakdown": {"overall_score": 88.5, "tier": "Exceptional Match"},
        "skills_analysis": {"matched_skills": [{"name": "React", "category": "Web"}]},
        "timestamp": "2026-09-03T10:00:00"
    }
    save_analysis(analysis_a_data, user_id=user_a["user_id"])
    
    # Save Analysis B under User B
    analysis_b_data = {
        "analysis_id": "anl_user_b_001",
        "filename": "resume_user_b.pdf",
        "target_role": "AI Engineer",
        "score_breakdown": {"overall_score": 92.0, "tier": "Exceptional Match"},
        "skills_analysis": {"matched_skills": [{"name": "PyTorch", "category": "AI/ML"}]},
        "timestamp": "2026-09-03T11:00:00"
    }
    save_analysis(analysis_b_data, user_id=user_b["user_id"])
    
    # Fetch history for User A
    history_a = get_history(user_id=user_a["user_id"])
    assert len(history_a) == 1
    assert history_a[0]["analysis_id"] == "anl_user_a_001"
    assert history_a[0]["filename"] == "resume_user_a.pdf"
    
    # Fetch history for User B
    history_b = get_history(user_id=user_b["user_id"])
    assert len(history_b) == 1
    assert history_b[0]["analysis_id"] == "anl_user_b_001"
    assert history_b[0]["filename"] == "resume_user_b.pdf"
    
    # Verify User A cannot fetch User B's detail by ID
    assert get_analysis_by_id("anl_user_b_001", user_id=user_a["user_id"]) is None
    assert get_analysis_by_id("anl_user_a_001", user_id=user_a["user_id"]) is not None


def test_clear_user_history_isolation():
    user_a = create_user("User A", "cleara@test.com", hash_password("passA123"))
    user_b = create_user("User B", "clearb@test.com", hash_password("passB123"))
    
    save_analysis({"analysis_id": "anl_a", "filename": "a.pdf"}, user_id=user_a["user_id"])
    save_analysis({"analysis_id": "anl_b", "filename": "b.pdf"}, user_id=user_b["user_id"])
    
    # Clear User A's history
    deleted = clear_user_history(user_id=user_a["user_id"])
    assert deleted == 1
    
    # User A has 0 history entries
    assert len(get_history(user_id=user_a["user_id"])) == 0
    
    # User B STILL HAS their history entry intact!
    assert len(get_history(user_id=user_b["user_id"])) == 1


def test_authorization_security_idor_prevention():
    """
    SECURITY AUDIT TEST:
    Verify protected endpoints reject unauthenticated calls (401 Unauthorized),
    and malicious parameter tampering (User A attempting to query User B's ID) is completely blocked.
    """
    user_a = create_user("User A", "sec_a@test.com", hash_password("passwordA123"))
    user_b = create_user("User B", "sec_b@test.com", hash_password("passwordB123"))
    
    token_a = create_access_token({"user_id": user_a["user_id"], "email": user_a["email"]})
    token_b = create_access_token({"user_id": user_b["user_id"], "email": user_b["email"]})
    
    # Save analysis for User B
    save_analysis({
        "analysis_id": "anl_sec_b",
        "filename": "private_b.pdf",
        "target_role": "Target B",
        "score_breakdown": {"overall_score": 90.0, "tier": "Exceptional Match"}
    }, user_id=user_b["user_id"])
    
    # 1. Unauthenticated calls MUST return 401 Unauthorized
    res_unauth_me = client.get("/auth/me")
    assert res_unauth_me.status_code == 401
    
    res_unauth_hist = client.get("/history")
    assert res_unauth_hist.status_code == 401
    
    res_unauth_del = client.delete("/history")
    assert res_unauth_del.status_code == 401
    
    # 2. Parameter Tampering Attack: User A calls /history with User A token but appends ?user_id=USER_B
    headers_a = {"Authorization": f"Bearer {token_a}"}
    res_tamper = client.get(f"/history?user_id={user_b['user_id']}", headers=headers_a)
    assert res_tamper.status_code == 200
    # Response MUST contain User A's user_id (0 records), ignoring client parameter tampering completely!
    assert res_tamper.json()["user_id"] == user_a["user_id"]
    assert len(res_tamper.json()["history"]) == 0
    
    # 3. IDOR Attack: User A tries to view User B's specific analysis by ID using User A token
    res_idor_view = client.get("/history/anl_sec_b", headers=headers_a)
    assert res_idor_view.status_code == 404  # Not found for User A
    
    # 4. IDOR Delete Attack: User A attempts to clear history with User A token
    res_delete_a = client.delete("/history", headers=headers_a)
    assert res_delete_a.status_code == 200
    assert res_delete_a.json()["deleted_count"] == 0  # 0 deleted for User A
    
    # Verify User B's record is STILL intact!
    headers_b = {"Authorization": f"Bearer {token_b}"}
    res_check_b = client.get("/history", headers=headers_b)
    assert res_check_b.status_code == 200
    assert len(res_check_b.json()["history"]) == 1
    assert res_check_b.json()["history"][0]["analysis_id"] == "anl_sec_b"
