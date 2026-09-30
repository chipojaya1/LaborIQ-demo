"""Skill extraction helpers for the chatbot RAG pipeline."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List

from rapidfuzz import fuzz

from ..utils.data_loader import load_glassdoor_skills

# Curated synonyms and skill families that we want to highlight in the demo.
SKILL_SYNONYMS = {
    "Python": {"python"},
    "SQL": {"sql", "structured query language"},
    "Tableau": {"tableau"},
    "Power BI": {"power bi", "power-bi"},
    "Machine Learning": {"machine learning", "ml"},
    "Deep Learning": {"deep learning", "neural network"},
    "Communication": {"communication", "stakeholder", "presentation"},
    "Project Management": {"project management", "agile", "scrum"},
    "Cloud Platforms": {"aws", "azure", "gcp", "cloud"},
}

FUZZY_THRESHOLD = 80


def _normalise(text: str) -> str:
    return text.lower()


def extract_skills(job_description: str, top_n: int = 5) -> List[Dict[str, object]]:
    """Return the most relevant skills mentioned in a job description."""
    corpus = load_glassdoor_skills()
    description = _normalise(job_description)
    counter: Counter[str] = Counter()

    # First pass: deterministic keyword extraction for curated skills.
    for canonical, variants in SKILL_SYNONYMS.items():
        score = sum(description.count(variant) for variant in variants)
        if score:
            counter[canonical] += score * 10  # Boost curated matches.

    # Second pass: fuzzy match against the Glassdoor corpus for serendipity.
    for skill in corpus:
        score = fuzz.partial_ratio(skill.lower(), description)
        if score >= FUZZY_THRESHOLD:
            counter[skill] += score

    if not counter:
        return []

    most_common = counter.most_common(top_n)
    return [
        {"skill": skill, "score": score / 100.0}
        for skill, score in most_common
    ]
