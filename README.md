# AI Resume Analyzer
### AI/ML + NLP + UI/UX Career Copilot

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)](https://streamlit.io/)
[![spaCy](https://img.shields.io/badge/spaCy-3.7.0-09A3D5.svg)](https://spacy.io/)
[![Tests](https://img.shields.io/badge/Tests-21%20Passed-success.svg)]()
[![Security](https://img.shields.io/badge/Security-JWT%20%2B%20Bcrypt-lock.svg)]()

A full-stack, production-ready web application that combines **AI/NLP analysis**, **modern SaaS UI/UX design**, **multi-user authentication**, and **explainable scoring algorithms** to help job seekers evaluate and optimize their resumes.

---

## 1. Project Overview

Job seekers face two fundamental challenges during their job search:
1. **General Resume Quality**: *"Is my resume well-structured, ATS-friendly, and impact-driven overall?"*
2. **Job-Specific Alignment**: *"How well does my resume match a specific job role I want to apply for?"*

**AI Resume Analyzer** solves both challenges by providing two distinct, user-centered evaluation modes:

- **📄 Resume Health Mode**: Evaluates overall resume structure, ATS readability, vocabulary impact, skills range, and quantifiable metrics **without requiring a job description**.
- **🎯 Job Match Mode**: Compares a resume against **any target job role or job description**, calculating skill overlap, keyword coverage, TF-IDF semantic similarity, and actionable recommendations.

---

## 2. Key Features

### 📄 Resume Health Mode
- **Overall Resume Health Score**: Out of 100 with qualitative status (`Exceptional`, `Good`, `Moderate`, `Needs Attention`).
- **ATS Readiness Audit**: Verifies standard section headings and ATS-friendly text formatting.
- **Content Quality**: Evaluates action verb frequency and bullet point readability.
- **Structure & Formatting**: Analyzes section word count balance and layout completeness.
- **Skills Range**: Scans taxonomy coverage across technical tools, languages, and frameworks.
- **Project Detail**: Assesses technical project presence and bullet descriptions.
- **Actionable Feedback**: Identifies Top Strengths, Areas to Improve, and Recommended Next Steps.
- **Contact Info & Profile Link Detection**: Scans for email, LinkedIn, GitHub, and portfolio links.

### 🎯 Job Match Mode
- **Any Custom Job Role**: Search or type any target role title (e.g. *UX Researcher*, *DevOps Engineer*, *Cybersecurity Analyst*, *Product Manager*).
- **Popular Role Shortcuts**: Pre-configured quick-start presets (`UI / UX Designer`, `Frontend Developer`, `Full Stack Developer`, `Data Analyst`, `ML Engineer`, `Product Designer`).
- **Flexible Job Input**:
  - *Option 1*: Role search with editable sample job description.
  - *Option 2*: Paste raw job description text (auto-extracts role title if present).
  - *Option 3*: Upload job description document (PDF/TXT).
- **Explainable Match Scoring**: Weighted score based on Skills (40%), Keywords (25%), Projects (20%), and Cosine Similarity (15%).
- **Side-by-Side Skill & Keyword Coverage**: Visual badge matrix of matched vs. missing skills and keyphrases.

### 🔐 Multi-User Authentication & Security
- **Account Management**: User Signup, Login, Session Management, and Logout.
- **Password Hashing**: Secure salted password hashing using `bcrypt` (12 rounds).
- **JWT Authorization**: Cryptographic JSON Web Tokens (`HS256`) for request validation.
- **User-Scoped History**: SQLite database queries strictly isolated by authenticated `user_id`.
- **IDOR / BOLA Prevention**: Protected endpoints derive user identity 100% from verified JWT tokens.
- **Zero-URL Token Exposure**: Tokens stored in browser storage (`localStorage`) to keep URL bar clean (`http://localhost:8502/`).

### 📊 SaaS Dashboard & User Experience
- **Personalized Workspace**: Summary cards for Total Analyses, Health Reviews, Job Match Evaluations, and Last Activity.
- **Analysis History**: Inspect and reload previous evaluations into the active dashboard view.
- **User Profile**: View user ID, registration date, and account controls.
- **Collapsible Navigation Sidebar**: Interactive expanded and collapsed navigation modes.

---

## 3. User Flow

```
                 +-----------------------+
                 |  SIGN UP / LOG IN     |
                 +-----------------------+
                             |
                             v
                 +-----------------------+
                 |   SaaS DASHBOARD      |
                 +-----------------------+
                    /                 \
                   /                   \
                  v                     v
      +-------------------+     +-------------------+
      |   RESUME HEALTH   |     |     JOB MATCH     |
      | (No Job Required) |     |  (Any Custom Job) |
      +-------------------+     +-------------------+
                |                         |
                v                         v
      +-------------------+     +-------------------+
      | Health Score &    |     | Match Score % &   |
      | Category Feedback |     | Skills Alignment  |
      +-------------------+     +-------------------+
                   \                   /
                    \                 /
                     v               v
             +-------------------------------+
             |   SAVED ANALYSIS HISTORY      |
             +-------------------------------+
```

---

## 4. UI/UX Design System

Designed to deliver a modern, consumer-facing SaaS product experience:

- **SaaS Dashboard Approach**: Clean card-based information architecture built for quick visual scanning.
- **Strong Visual Hierarchy**: Distinct typographic sizing, clear section headings, and status badges.
- **Custom Design System (`style.css`)**: Dark-mode color palette (`#0b0f19` background, `#151c2c` cards, `#6366f1` accent indigo, `#10b981` emerald).
- **Zero Viewport Overflow**: Responsive flex layout rules (`max-width: 100vw`, `overflow-x: hidden`, `word-break: break-word`) eliminating horizontal scrollbars.
- **User-Centered Dual-Mode Workflow**: Clear decision point (*"What would you like to analyze?"*) allowing users to choose between Resume Health and Job Match.
- **Empty & Feedback States**: Balanced empty dashboard states, loading indicators, and success notifications.

---

## 5. AI/ML & NLP Implementation

The core intelligence is driven by deterministic NLP algorithms and statistical text vectorization without relying on paid external LLM APIs:

- **spaCy (`en_core_web_sm`)**: Used for POS tagging, entity recognition, and noun chunk extraction.
- **TF-IDF Vectorization (`scikit-learn`)**: Converts resume text and job descriptions into term-frequency inverse-document-frequency vectors.
- **Cosine Similarity**: Measures semantic distance between resume TF-IDF vectors and target job requirements (0.0% to 100.0%).
- **Skills Taxonomy (`skills.json`)**: Pre-built catalog of 200+ canonical technical/soft skills with alias normalization (e.g. `JS` $\rightarrow$ `JavaScript`, `React.js` $\rightarrow$ `React`).
- **Rule-Based Scoring Engine**: Multi-factor weighted formula calculating transparent scores out of 100.
- **PyMuPDF (`fitz`)**: Fast PDF document parsing, page extraction, and text normalization.

---

## 6. System Architecture

```
                  +----------------------------------+
                  | Streamlit Frontend (frontend/app.py) |
                  +----------------------------------+
                                   |
                             HTTP / REST API
                                   v
                  +----------------------------------+
                  | FastAPI REST Backend (backend/main.py)|
                  +----------------------------------+
                                   |
                  +----------------------------------+
                  |  JWT Security & Business Logic   |
                  +----------------------------------+
                                   |
          +------------------------+------------------------+
          |                        |                        |
          v                        v                        v
+------------------+     +-------------------+    +--------------------+
|  PyMuPDF Parser  |     |  spaCy + TF-IDF   |    | Resume Health &    |
|  (backend/parser)|     |  (backend/nlp)    |    | Scoring Engines    |
+------------------+     +-------------------+    +--------------------+
                                   |
                                   v
                  +----------------------------------+
                  | SQLite Database (data/*.db)      |
                  +----------------------------------+
```

---

## 7. Security Controls

- **Cryptographic Password Hashing**: Passwords stored as bcrypt salted hashes (never plaintext).
- **JWT Authorization**: Token payload decoded and validated using `HS256` secret verification.
- **User-Scoped Queries**: All SQLite queries enforce `WHERE user_id = ?` to prevent cross-tenant data leakage.
- **IDOR / BOLA Prevention**: Client-supplied `user_id` query parameters are ignored; caller identity is derived 100% from the Bearer JWT token header.
- **Zero URL Bar Exposure**: Tokens saved in browser `localStorage` to prevent token leakage in URL query strings.
- **Environment Configuration**: Production secrets loaded via `os.getenv("JWT_SECRET")`.

---

## 8. Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Frontend** | Streamlit, Custom SaaS CSS System, HTML5 |
| **Backend** | Python 3.11, FastAPI, Pydantic v2 |
| **AI / NLP** | spaCy (`en_core_web_sm`), scikit-learn, TF-IDF, Cosine Similarity |
| **Data & Storage** | SQLite 3, pandas |
| **Document Processing** | PyMuPDF (`fitz`) |
| **Security & Auth** | bcrypt, PyJWT |
| **Testing** | pytest, FastAPI TestClient |

---

## 9. Testing & Quality Assurance

The codebase contains **21 automated unit tests** achieving 100% pass rate:

```bash
python -m pytest tests/ -v
```

### Test Coverage Categories:
- **Authentication**: Password hashing, verification, signup validation, login flow.
- **JWT Tokens**: Token generation, expiration decoding, secret verification.
- **Multi-User Isolation**: Scoped SQLite history query separation between users.
- **Authorization & IDOR**: Verifies parameter tampering attempts are blocked.
- **Resume Health**: Sub-category calculations, link detection, recommendations.
- **NLP & Matching**: Skill taxonomy extraction, TF-IDF cosine similarity, keyword extraction.
- **Document Parsing**: PyMuPDF synthetic PDF parsing and section identification.
- **Scoring**: Weighted score formula and rating tier classification.

---

## 10. Project Structure

```
hopeful-faraday/
├── backend/
│   ├── __init__.py
│   ├── auth.py              # Password hashing & JWT token management
│   ├── database.py          # SQLite database schema, user management & migration
│   ├── health_analyzer.py   # Mode A: Resume Health evaluation engine
│   ├── main.py              # FastAPI REST API server & JWT security dependency
│   ├── matcher.py           # Skill matching, keyword coverage & project analysis
│   ├── models.py            # Pydantic v2 schemas for auth, health & job match
│   ├── nlp_analyzer.py      # spaCy & TF-IDF Cosine Similarity engine + role extraction
│   ├── parser.py            # PyMuPDF (fitz) text & section extraction
│   ├── scoring.py           # Weighted scoring algorithm
│   └── suggestions.py       # Rule-based AI recommendations
├── data/
│   ├── skills.json          # 200+ taxonomy skills & canonical alias mappings
│   └── analyzer_history.db  # Local SQLite database (git-ignored)
├── frontend/
│   ├── app.py               # Streamlit SaaS dashboard & persistent auth bridge
│   └── style.css            # Custom CSS design system (zero viewport overflow)
├── tests/
│   ├── test_auth.py         # Auth, JWT, multi-user isolation & IDOR tests
│   ├── test_health_analyzer.py # Resume Health & dual mode tests
│   ├── test_matcher.py      # Matcher engine unit tests
│   ├── test_nlp.py          # spaCy & TF-IDF similarity unit tests
│   ├── test_parser.py       # PDF parsing unit tests
│   └── test_scoring.py      # Scoring algorithm unit tests
├── uploads/
│   └── .gitkeep             # Directory placeholder for uploads (git-ignored)
├── .gitignore               # Production gitignore rules
├── README.md                # Project documentation
└── requirements.txt         # Runtime dependencies
```

---

## 11. Local Setup Instructions

### Prerequisites
- Python 3.11+ installed

### Step 1: Clone Repository & Create Virtual Environment
```powershell
# Activate virtual environment (Windows)
.venv\Scripts\activate
```

### Step 2: Install Dependencies & Download spaCy Model
```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Step 3: Run Automated Test Suite
```powershell
python -m pytest tests/ -v
```

### Step 4: Launch FastAPI REST API Server
```powershell
uvicorn backend.main:app --reload --port 8000
```

### Step 5: Launch Streamlit SaaS Frontend Application
```powershell
streamlit run frontend/app.py --server.port 8501
```

Access the application in your web browser at `http://localhost:8501`.

---

## 12. Application Screenshots

*(Screenshots will be added here)*

- **Dashboard**: `[Screenshot Placeholder: Personalized User Dashboard]`
- **Resume Health**: `[Screenshot Placeholder: Resume Health Diagnostics]`
- **Job Match**: `[Screenshot Placeholder: Job Match & Skills Alignment]`
- **Analysis History**: `[Screenshot Placeholder: Scoped User History Table]`

---

## 13. Future Improvements

- **OCR Integration**: Tesseract OCR support for scanned PDF resumes.
- **Optional LLM Integration**: Generative bullet point rewriting suggestions.
- **Cloud Database Migration**: Migration path to PostgreSQL / Supabase.
- **Export Reports**: Downloadable PDF evaluation summary reports.

---

## 14. Portfolio Note

This project demonstrates a complete end-to-end software engineering effort combining **UI/UX Design**, **Web Development**, **Natural Language Processing**, **REST API Architecture**, and **Application Security Controls**.
