"""OpenAI-based skill extraction and insight generator.

This module replaces a local fine-tuned pipeline with calls to OpenAI's chat API
to extract skills and generate a short career insight for a job description.

Public helpers:
- extract_skills_openai(user_input: str, client: OpenAI, dataset_path: str, taxonomy_path: str) -> dict
- generate_insight_openai(user_input: str, combined_skills: list[str], client: OpenAI) -> dict

The functions return plain Python dicts that can be returned from a FastAPI endpoint.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import pandas as pd

from openai import OpenAI


# Skill taxonomy adapted from Lightcast (Emsi Burning Glass) skill framework
# This in-code fallback ensures classification works even if the JSON file is absent.
custom_tech_skills = {
    "programming_languages": [
        "python", "r", "c", "c++", "c#", "scala", "go", "perl", "oracle java"
    ],
    "data_management_query_languages": [
        "structured query language sql", "microsoft sql server", "postgresql",
        "mongodb", "nosql", "apache hive", "amazon redshift",
        "teradata database", "javascript", "json"
    ],
    "cloud_devops_platforms": [
        "amazon web services aws", "amazon simple storage service s3",
        "amazon elastic compute cloud ec2", "microsoft azure",
        "docker", "kubernetes", "jenkins ci", "linux", "unix", "bash"
    ],
    "machine_learning_ai_frameworks": [
        "tensorflow", "pytorch", "apache spark", "apache hadoop",
        "apache kafka", "sas", "ibm spss statistics", "the mathworks matlab",
        "alteryx software"
    ],
    "data_visualization_bi_tools": [
        "tableau", "microsoft power bi", "microsoft excel", "microsoft access",
        "microsoft powerpoint", "microsoft office software"
    ],
    "version_control_collaboration_tools": [
        "git", "github", "atlassian jira"
    ]
}


def _load_examples(dataset_path: str, n_examples: int = 3) -> List[str]:
    """Load a few representative examples from the CSV to use as few-shot context.

    The dataset is expected to contain columns:
    - description
    - skills_list (stringified list or comma-separated string)
    """
    df = pd.read_csv(dataset_path, on_bad_lines="skip", engine="python")
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    if "description" not in df.columns or "skills_list" not in df.columns:
        return []

    examples = []
    count = 0
    for _, row in df.iterrows():
        skills = row.get("skills_list")
        desc = row.get("description")
        if pd.isna(desc) or pd.isna(skills):
            continue
        # Normalize skills field to comma-separated string
        if isinstance(skills, list):
            skills_text = ", ".join(map(str, skills))
        else:
            skills_text = str(skills)
        examples.append(f"Job Description:\n{desc}\nSkills:\n{skills_text}")
        count += 1
        if count >= n_examples:
            break
    return examples


def _load_taxonomy(taxonomy_path: str) -> Dict[str, List[str]]:
    """Load taxonomy JSON file mapping category -> list of skill strings.

    If the file is missing, fall back to the in-module `custom_tech_skills`.
    """
    try:
        with open(taxonomy_path, "r") as fh:
            taxonomy = json.load(fh)
    except FileNotFoundError:
        taxonomy = custom_tech_skills
    # Normalize entries to lowercase for matching
    normalized = {k: [s.lower() for s in v] for k, v in taxonomy.items()}
    return normalized


def _classify_skills(skills: List[str], taxonomy: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Classify extracted skills into taxonomy categories using simple matching.

    Returns a dict with the same keys as taxonomy and lists of matching skills.
    Unmatched skills are omitted from categories but remain in raw_skills.
    """
    categories = {k: [] for k in taxonomy.keys()}
    for skill in skills:
        lower = skill.lower()
        matched = False
        for cat, patterns in taxonomy.items():
            for p in patterns:
                # match exact or substring
                if p in lower or lower in p:
                    categories[cat].append(skill)
                    matched = True
                    break
            if matched:
                break
    return categories


def extract_skills_openai(
    user_input: str,
    client: OpenAI,
    dataset_path: str = "data/glassdoor_extracted_skills.csv",
    taxonomy_path: str = "data/custom_tech_skills.json",
    n_examples: int = 3,
) -> Dict[str, object]:
    """Extract raw skills and classified categories for a job description using OpenAI.

    Returns a dict with keys: raw_skills (list), categories (dict).
    """
    examples = _load_examples(dataset_path, n_examples=n_examples)
    taxonomy = _load_taxonomy(taxonomy_path)

    system_prompt = (
        "You are a lightweight skill extractor. Given a job description, "
        "return JSON with two fields: 'raw_skills' (a list of short skill phrases) "
        "and 'categories' (a dict mapping category names to lists of skills).\n"
        "Use the examples to learn formatting. Respond with JSON only."
    )

    # Build user prompt with few-shot examples and the new job description
    few_shot_text = "\n\n".join(examples)
    user_prompt = f"EXAMPLES:\n{few_shot_text}\n\nNEW JOB DESCRIPTION:\n{user_input}\n\nRespond with JSON: {{\n  \"raw_skills\": [...],\n  \"categories\": {{ ... }}\n}}"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        raw_output = resp.choices[0].message.content
        try:
            parsed = json.loads(raw_output)
        except Exception:
            # Attempt to extract JSON substring
            start = raw_output.find("{")
            end = raw_output.rfind("}")
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(raw_output[start : end + 1])
                except Exception:
                    parsed = {"raw_skills": [], "categories": {k: [] for k in taxonomy.keys()}, "parse_error": raw_output}
            else:
                parsed = {"raw_skills": [], "categories": {k: [] for k in taxonomy.keys()}, "parse_error": raw_output}
    except Exception as e:
        # On API error, return an empty structure with error
        return {"raw_skills": [], "categories": {k: [] for k in taxonomy.keys()}, "error": str(e)}

    # Ensure keys are present
    raw_skills = parsed.get("raw_skills") or []
    # Normalize skills list to simple strings
    raw_skills = [str(s).strip() for s in raw_skills]

    # Classify using local taxonomy matching in case model didn't classify fully
    categories = parsed.get("categories")
    if not categories:
        categories = _classify_skills(raw_skills, taxonomy)
    else:
        # Normalize categories to ensure keys exist and values are lists
        normalized = {k: list(v) if isinstance(v, list) else [] for k, v in categories.items()}
        # Fill missing categories using local classifier
        for k in taxonomy.keys():
            if k not in normalized or not normalized[k]:
                normalized[k] = [s for s in _classify_skills(raw_skills, taxonomy).get(k, [])]
        categories = normalized

    return {"raw_skills": raw_skills, "categories": categories}


def generate_insight_openai(user_input: str, combined_skills: List[str], client: OpenAI) -> Dict[str, object]:
    """Generate a short structured insight given the job description and extracted skills.

    Returns a dict with keys: summary (str), top_skills (list), recommendations (list).
    """
    skills_text = ", ".join(combined_skills)
    system_prompt = "You are a professional career advisor AI. Produce a short JSON with summary, top_skills and recommendations."

    user_prompt = f"JOB DESCRIPTION:\n{user_input}\n\nEXTRACTED SKILLS:\n{skills_text}\n\nRespond with JSON: {json.dumps({'summary': '...', 'top_skills': ['...'], 'recommendations': ['...', '...']})}"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        raw_output = resp.choices[0].message.content
        try:
            parsed = json.loads(raw_output)
        except Exception:
            start = raw_output.find("{")
            end = raw_output.rfind("}")
            if start != -1 and end != -1:
                try:
                    parsed = json.loads(raw_output[start : end + 1])
                except Exception:
                    parsed = {"summary": "", "top_skills": combined_skills[:5], "recommendations": []}
            else:
                parsed = {"summary": "", "top_skills": combined_skills[:5], "recommendations": []}
    except Exception:
        parsed = {"summary": "", "top_skills": combined_skills[:5], "recommendations": []}

    # Ensure required fields exist
    summary = parsed.get("summary", "")
    top_skills = parsed.get("top_skills", combined_skills[:5])
    recommendations = parsed.get("recommendations", [])

    return {"summary": summary, "top_skills": top_skills, "recommendations": recommendations}


_SKILL_SUFFIXES = (" programming", " skills", " development", " management")
_SKILL_EXEMPTIONS = {
    "data analysis",
    "data visualization",
    "project management",
}


def _normalize_skill_name(raw: str) -> str:
    """Collapse redundant suffixes (e.g. 'Python programming' -> 'Python')."""
    if not raw:
        return raw
    text = str(raw).strip()
    if not text:
        return text

    lower = text.lower()
    if lower in _SKILL_EXEMPTIONS:
        return _restore_case(text)

    for suffix in _SKILL_SUFFIXES:
        if lower.endswith(suffix) and lower not in _SKILL_EXEMPTIONS:
            base = text[: -len(suffix)].rstrip(" -_/")
            if base:
                text = base
                lower = text.lower()
            break

    text = " ".join(text.split())
    if not text:
        return raw
    return _restore_case(text)


def _restore_case(text: str) -> str:
    tokens = text.split()
    normalised: List[str] = []
    for token in tokens:
        if token.isupper():
            normalised.append(token)
        elif len(token) <= 3:
            normalised.append(token.upper())
        else:
            normalised.append(token.capitalize())
    return " ".join(normalised)


def _dedupe_skills(skills: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for skill in skills:
        if not skill:
            continue
        if skill not in seen:
            seen.add(skill)
            ordered.append(skill)
    return ordered


def skill_match_from_resume(resume_text: str, skills: List[str]) -> Dict[str, object]:
    """Compute keyword match stats between resume text and a list of skills.

    Returns a dict with match_rate (percentage), matched_skills, missing_skills and a short summary.
    """
    resume_lower = (resume_text or "").lower()
    matched = [s for s in skills if s and s.lower() in resume_lower]
    missing = [s for s in skills if s and s.lower() not in resume_lower]
    coverage = round(len(matched) / len(skills) * 100, 1) if skills else 0.0
    return {
        "match_rate": coverage,
        "matched_skills": matched,
        "missing_skills": missing,
        "summary": f"Your resume has {len(matched)} out of {len(skills)} ({coverage}%) keywords from the job description."
    }


def analyze_job_description(
    user_input: str,
    api_key: Optional[str] = None,
    dataset_path: str = "data/glassdoor_extracted_skills.csv",
    taxonomy_path: str = "data/custom_tech_skills.json",
    resume_text: Optional[str] = None,
) -> Dict[str, object]:
    """High-level helper that runs extraction and insight generation using OpenAI.

    - Creates an OpenAI client (if api_key provided)
    - Extracts skills and categories
    - Generates a chatbot_insight structure
    - Returns the full combined JSON
    """
    if api_key is None:
        raise ValueError("api_key is required to call the OpenAI API")

    client = OpenAI(api_key=api_key)

    extraction = extract_skills_openai(user_input, client, dataset_path, taxonomy_path)
    raw_skills = [
        _normalize_skill_name(skill)
        for skill in extraction.get("raw_skills", [])
        if skill
    ]

    insight = generate_insight_openai(user_input, raw_skills, client)
    insight_top = [
        _normalize_skill_name(skill)
        for skill in insight.get("top_skills", []) or []
        if skill
    ]
    insight["top_skills"] = _dedupe_skills(insight_top)

    categories = extraction.get("categories", {})
    normalized_categories: Dict[str, List[str]] = {}
    for category, values in (categories or {}).items():
        normalized_categories[category] = _dedupe_skills(
            [_normalize_skill_name(value) for value in values if value]
        )

    # Build an "in_demand_skills" list - combine raw_skills and insight top_skills (deduplicated)
    in_demand = _dedupe_skills(raw_skills + insight["top_skills"])

    result: Dict[str, object] = {
        "raw_skills": raw_skills,
        "in_demand_skills": in_demand,
        "categories": normalized_categories,
        "chatbot_insight": insight,
    }

    # If resume_text supplied, compute resume matching stats
    if resume_text:
        resume_match = skill_match_from_resume(resume_text, in_demand)
        result["resume_match"] = resume_match

    return result


if __name__ == "__main__":
    # Simple local test harness
    import getpass

    key = getpass.getpass("Enter OpenAI API key: ")
    sample = "We are hiring a data scientist with Python, SQL, TensorFlow and AWS experience."
    out = analyze_job_description(sample, api_key=key)
    print(json.dumps(out, indent=2))
