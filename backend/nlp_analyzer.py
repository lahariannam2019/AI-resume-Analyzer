"""
NLP Analyzer using spaCy, scikit-learn (TF-IDF, Cosine Similarity), and Skills Taxonomy matching.
"""

import os
import json
import re
from typing import List, Dict, Set, Tuple, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend.models import SkillItem

# Try importing spaCy, fallback gracefully if model is loading
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
except ImportError:
    nlp = None


def load_skills_data() -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Load canonical categories and alias mappings from skills.json."""
    skills_json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "skills.json")
    if not os.path.exists(skills_json_path):
        return {}, {}
        
    with open(skills_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    categories = data.get("categories", {})
    aliases = data.get("aliases", {})
    return categories, aliases


CATEGORIES_MAP, ALIASES_MAP = load_skills_data()

# Pre-build skill lookup dictionary: {normalized_name: (canonical_name, category)}
SKILL_LOOKUP: Dict[str, Tuple[str, str]] = {}
for category, skills in CATEGORIES_MAP.items():
    for skill in skills:
        norm = skill.lower().strip()
        SKILL_LOOKUP[norm] = (skill, category)

# Add aliases to lookup
for alias, canonical in ALIASES_MAP.items():
    norm_alias = alias.lower().strip()
    norm_canonical = canonical.lower().strip()
    category = SKILL_LOOKUP.get(norm_canonical, (canonical, "General"))[1]
    SKILL_LOOKUP[norm_alias] = (canonical, category)


def extract_skills(text: str) -> List[SkillItem]:
    """Extract skills from text using boundary-aware regex matching against skill taxonomy & aliases."""
    if not text:
        return []
        
    found_skills: Dict[str, SkillItem] = {}
    normalized_text = f" {text.lower()} "
    
    sorted_terms = sorted(SKILL_LOOKUP.keys(), key=lambda x: len(x), reverse=True)
    
    for term in sorted_terms:
        canonical, category = SKILL_LOOKUP[term]
        if canonical in found_skills:
            continue
            
        escaped_term = re.escape(term)
        pattern = r'(?:^|[\s,./()\[\]{}:;\-\"])' + escaped_term + r'(?:$|[\s,./()\[\]{}:;\-\"])'
        
        if re.search(pattern, normalized_text, re.IGNORECASE):
            found_skills[canonical] = SkillItem(name=canonical, category=category)
            
    return list(found_skills.values())


def extract_keywords_and_phrases(text: str, top_n: int = 25) -> List[str]:
    """Extract key terms and noun phrases using spaCy or TF-IDF fallback."""
    if not text or not text.strip():
        return []
        
    keywords: Set[str] = set()
    
    if nlp is not None:
        try:
            doc = nlp(text)
            for chunk in doc.noun_chunks:
                clean_chunk = chunk.text.strip().lower()
                if len(clean_chunk) > 3 and not re.match(r'^(the|a|an|this|that|these|those|we|our|you|your)$', clean_chunk):
                    keywords.add(clean_chunk.title())
        except Exception:
            pass
            
    # TF-IDF fallback for single words and bigrams
    try:
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=top_n)
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        for name in feature_names:
            keywords.add(name.title())
    except Exception:
        pass
        
    return sorted(list(keywords))[:top_n]


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate TF-IDF Cosine Similarity score (0.0 to 100.0) between two text documents."""
    if not text1 or not text2 or not text1.strip() or not text2.strip():
        return 0.0
        
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = vectorizer.fit_transform([text1, text2])
        sim_matrix = cosine_similarity(tfidf[0:1], tfidf[1:2])
        score = float(sim_matrix[0][0]) * 100.0
        return round(min(max(score, 0.0), 100.0), 1)
    except Exception:
        return 0.0


def extract_role_title_from_jd(jd_text: str) -> Optional[str]:
    """
    Attempt to auto-extract a likely job role title from a pasted/uploaded Job Description text.
    Returns extracted title if found, or None if uncertain.
    """
    if not jd_text or not jd_text.strip():
        return None
        
    patterns = [
        r'(?:position|role|job title|title)\s*:\s*([A-Za-z0-9\s/\-]{3,35})(?:\n|\.|$)',
        r'(?:hiring|seeking|looking for)\s+(?:a|an)?\s*([A-Za-z0-9\s/\-]{3,35})(?:\s+to|\s+who|\s+for|\n|\.|$)',
        r'^([A-Z][A-Za-z0-9\s/\-]{3,30})\s*(?:\n|\r)'
    ]
    
    for pat in patterns:
        match = re.search(pat, jd_text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            if 3 <= len(candidate) <= 40 and not any(kw in candidate.lower() for kw in ["responsibilities", "requirements", "company", "description"]):
                return candidate.title()
                
    return None
