"""FastAPI application exposing the demo chatbot backend."""
from __future__ import annotations

import os
from functools import lru_cache
import logging
import re
from pathlib import Path
from collections import Counter

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# pydantic models live in `models/schemas.py`
from typing import Optional
import pandas as pd

try:  # Optional dependency for PDF parsing; handled gracefully if missing.
    from PyPDF2 import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore[assignment]


from .models.schemas import (
    ChatMode,
    ChatResponse,
    ChatTurn,
    MarketInsightRequest,
    MarketInsightResponse,
    SkillAnalysisRequest,
    SkillAnalysisResponse,
)
from .services import market_insights, skills_analysis
from .utils.data_loader import load_oews_data
from .services.model_skill_analysis import analyze_job_description, skill_match_from_resume

app = FastAPI(title="LaborIQ Chatbot Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Minimal import-time log to help verify the module was imported by uvicorn
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)
_logger.info("Imported backend.app.main — routes registered")
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)
_logger.info("Imported backend.app.main — routes registered")


@app.get("/health", summary="Simple health check")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/skill_analysis")
async def analyze_skills(
    job_description: str = Form(...),
    resume_file: UploadFile | None = File(None),
) -> JSONResponse:
    """Analyze a job description (and optional resume) sent as multipart/form-data.

    The endpoint accepts:
    - job_description: str (required form field)
    - resume_file: UploadFile (optional PDF/DOCX/TXT)

    Returns a JSON payload with in-demand skills and optional resume match stats.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY environment variable not set")

    normalized_description = (job_description or "").strip()
    if not normalized_description:
        raise HTTPException(status_code=400, detail="job_description cannot be empty")

    resume_text = None
    if resume_file is not None:
        try:
            resume_text = extract_text_from_resume(resume_file)
        except Exception as exc:  # pragma: no cover - defensive logging
            _logger.error("Error extracting resume text: %s", exc)
            raise HTTPException(status_code=400, detail=f"Unable to extract resume text: {exc}")

    _logger.info("skill_analysis request received; resume: %s", resume_file.filename if resume_file else "none")

    try:
        analysis = analyze_job_description(
            user_input=normalized_description,
            resume_text=resume_text,
            api_key=api_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive logging
        _logger.exception("Error analyzing job description")
        raise HTTPException(status_code=500, detail="Internal error analyzing job description") from exc

    response_payload: dict[str, object] = {
        "in_demand_skills": analysis.get("in_demand_skills", []),
        "raw_skills": analysis.get("raw_skills", []),
        "categories": analysis.get("categories", {}),
        "chatbot_insight": analysis.get("chatbot_insight"),
    }

    resume_match = analysis.get("resume_match")
    if resume_match:
        response_payload["resume_match"] = {
            "match_rate": resume_match.get("match_rate", 0.0),
            "matched_skills": resume_match.get("matched_skills", []),
            "missing_skills": resume_match.get("missing_skills", []),
            "summary": resume_match.get("summary", ""),
        }

    return JSONResponse(content=response_payload)


@app.post("/api/insights", response_model=MarketInsightResponse)
def get_market_insights(request: MarketInsightRequest) -> MarketInsightResponse:
    states = market_insights.top_states_for_occupation(
        request.occupation,
        request.metric,
        request.top_n,
    )
    _logger.info("Returning %s states for %s", len(states), request.occupation)
    return MarketInsightResponse(states=states)


@app.get("/api/bls/unemployment_rate")
def get_unemployment_rate() -> dict:
    """Return U.S. national unemployment rate for the past 5 years (60 periods)."""
    # Call the helper which normally returns a list of records
    result = market_insights.macro_trend("unemployment_rate", periods=60)
    return {"data": result}


@app.get("/api/bls/{series_id}")
def get_macro_trend(series_id: str, periods: int = 12) -> dict:
    data = market_insights.macro_trend(series_id, periods)
    return {"data": data}


@app.post("/api/skills", response_model=SkillAnalysisResponse)
def analyse_skills(request: SkillAnalysisRequest) -> SkillAnalysisResponse:
    skills = skills_analysis.extract_skills(request.job_description, request.top_n)
    return SkillAnalysisResponse(skills=skills)


def _clean_text(text: str) -> str:
    """Clean text by removing filler words and normalizing for entity extraction.
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text with filler words removed and normalized spacing
    """
    text = text.lower()
    # Remove filler words and task-specific terms
    text = re.sub(
        r"\b(what|is|the|for|in|of|average|salary|pay|wage|how|much|make|does|do|earn|earning"
        r"|a|an|to|on|at|by|with|from|and|or|please|show|tell|me|about|which|has|have"
        r"|can|could|would|should|will|may|might|shall|who|whose|where|when|why|that|this"
        r"|these|those|it|as|than|then|so|such|just|like|but|if|else|also|too|very|really"
        r"|more|most|least|few|many|some|any|each|every|either|neither|both|all|no|not|nor)\b",
        "", 
        text
    )
    # Remove non-alphanumeric chars and normalize whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _detect_intent(text: str, keywords: list[str]) -> bool:
    """Check if any keyword appears in the text.
    
    Args:
        text: Text to check
        keywords: List of keywords to look for
    
    Returns:
        True if any keyword is found in the text
    """
    return any(k in text for k in keywords)


@app.post("/api/chat", response_model=ChatResponse)
def chat(turn: ChatTurn) -> ChatResponse:
    if not turn.question:
        return ChatResponse(
            answer="Hi there! How can I help you with your career insights today?",
            payload=None
        )
    
    # --- Check for small talk first ---
    text = turn.question.strip().lower()
    
    # Intent detection keywords
    greeting_keywords = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    thanks_keywords = ["thanks", "thank you", "thx"]
    farewell_keywords = ["bye", "goodbye", "see you", "talk later"]
    how_are_you_keywords = ["how are you", "how're you", "how you doing", "how do you do"]
    job_keywords = [
        "salary", "average salary", "highest salary", "job openings", "job postings", "employment", "hiring",
        "skills", "skill", "insight", "position", "opening", "vacancy", "pay", "wage"
    ]
    
    # Check if this is just small talk without any job/skill content
    is_small_talk = any(k in text for k in greeting_keywords + thanks_keywords + farewell_keywords + how_are_you_keywords)
    has_job_content = any(k in text for k in job_keywords)
    
    # Handle small talk responses
    if is_small_talk and not has_job_content:
        if any(k in text for k in greeting_keywords):
            return ChatResponse(answer="Hi there! How can I help you with your career insights today?", payload=None)
        if any(k in text for k in how_are_you_keywords):
            return ChatResponse(answer="I'm a career assistant — ready to help with skills and market insights. What would you like to explore?", payload=None)
        if any(k in text for k in thanks_keywords):
            return ChatResponse(answer="You're welcome — anything else I can help with?", payload=None)
        if any(k in text for k in farewell_keywords):
            return ChatResponse(answer="Goodbye! Feel free to ask about jobs, salaries, or skills anytime.", payload=None)

    # --- Check for skill analysis intent ---
    skill_keywords = ["skill", "skills", "job description", "job requirements", "qualifications", "requirements"]
    is_skill_query = any(k in text for k in skill_keywords) or len(text.split()) > 10  # Longer text likely needs skill analysis
    
    if is_skill_query:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        try:
            result = analyze_job_description(text, api_key=api_key)
            return ChatResponse(
                answer=result["chatbot_insight"]["summary"],
                payload=result,
                sources=["Skill extraction supported by GWU LAiSER Team"]
            )
        except Exception as e:
            _logger.error(f"OpenAI API error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error analyzing skills")

    # --- Market insights for salary/employment queries ---
    cleaned = _clean_text(text)
    
    # Extract occupation (normalize plural/singular)
    occupation = _extract_occupation(cleaned)
    if occupation and occupation.endswith("s") and not occupation.endswith("ss"):
        occupation = occupation[:-1]  # crude singularization

    # Extract state
    state = _extract_state(cleaned)

    # Metric intent
    if any(k in cleaned for k in ["salary", "average salary", "highest salary", "pay", "wage"]):
        intent_metric = "average_salary"
    elif any(k in cleaned for k in ["job openings", "job postings", "employment", "hiring"]):
        intent_metric = "total_employment"
    else:
        intent_metric = "average_salary"

    # Small-talk detection without any job content
    greeting_keywords = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    thanks_keywords = ["thanks", "thank you", "thx"]
    farewell_keywords = ["bye", "goodbye", "see you", "talk later"]
    how_are_you_keywords = ["how are you", "how're you", "how you doing", "how do you do"]
    job_keywords = [
        "salary", "average salary", "highest salary", "job openings", "job postings", "employment", "hiring",
        "skills", "skill", "insight", "position", "opening", "vacancy", "pay", "wage"
    ]
    
    is_small_talk = (
        any(k in text for k in greeting_keywords + thanks_keywords + farewell_keywords + how_are_you_keywords) and
        not any(k in text for k in job_keywords) and
        not occupation
    )
    
    if is_small_talk:
        if any(k in text for k in greeting_keywords):
            return ChatResponse(answer="Hi there! How can I help you with your career insights today?", payload=None)
        if any(k in text for k in how_are_you_keywords):
            return ChatResponse(answer="I'm a career assistant — ready to help with skills and market insights. What would you like to explore?", payload=None)
        if any(k in text for k in thanks_keywords):
            return ChatResponse(answer="You're welcome — anything else I can help with?", payload=None)
        if any(k in text for k in farewell_keywords):
            return ChatResponse(answer="Goodbye! Feel free to ask about jobs, salaries, or skills anytime.", payload=None)

    # --- Market insights for salary/employment queries ---
    if not occupation:
        cleaned = _clean_text(text)
        occupation = _extract_occupation(cleaned)
        if occupation and occupation.endswith("s") and not occupation.endswith("ss"):
            occupation = occupation[:-1]  # crude singularization

    if not state:
        state = _extract_state(text)

    # Default to salary mode unless specified
    if not intent_metric:
        intent_metric = "average_salary"
        if any(k in text for k in ["job openings", "job postings", "employment", "hiring"]):
            intent_metric = "total_employment"

    request = MarketInsightRequest(occupation=occupation, metric=intent_metric, top_n=60)
    insights = market_insights.top_states_for_occupation(
        request.occupation,
        request.metric,
        request.top_n,
    )

    if not insights:
        return ChatResponse(answer="No related data available — we'll update this soon.", payload=None)

    if state:
        state_data = next((s for s in insights if s["state_name"].lower() == state.lower()), None)
        if state_data:
            if intent_metric == "average_salary":
                answer = f"The average salary for {occupation} in {state} is ${state_data['average_salary']:,.0f}."
            else:
                answer = f"Estimated employment for {occupation} in {state}: {int(state_data.get('total_employment', 0)):,}."
            return ChatResponse(answer=answer, payload={"states": insights}, sources=["OEWS state-level dataset"])
        else:
            return ChatResponse(answer=f"Sorry, I don't have data for {occupation} in {state} yet. I'll update this soon.", payload=None)

    # Return top states if no specific state asked
    answer = _format_market_answer(insights[:5], occupation=occupation)
    return ChatResponse(answer=answer, payload={"states": insights}, sources=["OEWS state-level dataset"])


@lru_cache(maxsize=1)
def _state_corpus() -> tuple[str, ...]:
    """Get a list of all state names from the OEWS dataset."""
    df = load_oews_data()
    states = df.get("state_name")
    if states is None:
        return ()
    unique = {str(item).strip() for item in states.dropna() if str(item).strip()}
    return tuple(sorted(unique, key=str.lower))


@lru_cache(maxsize=1)
def _occupation_corpus() -> tuple[str, ...]:
    df = load_oews_data()
    occupations = df.get("occupation")
    if occupations is None:
        return ()
    unique = {str(item).strip() for item in occupations.dropna() if str(item).strip()}
    return tuple(sorted(unique, key=str.lower))


def _extract_state(question: str) -> str | None:
    """Extract state name from question using fuzzy matching."""
    if not question:
        return None
    cleaned = question.strip().lower()
    states = _state_corpus()
    
    # Direct match first
    for state in states:
        if state.lower() in cleaned:
            return state
    
    # Fuzzy match if no direct match
    if states:
        from rapidfuzz import fuzz, process
        candidate, score, _ = process.extractOne(
            cleaned,
            states,
            scorer=fuzz.WRatio,
            score_cutoff=80,  # Higher threshold for states to avoid false matches
        ) or (None, None, None)
        if candidate and score:
            return candidate
    return None


def _extract_occupation(question: str) -> str | None:
    if not question:
        return None
    cleaned = question.strip()
    if not cleaned:
        return None
    occupations = _occupation_corpus()
    lowered_question = cleaned.lower()
    direct_matches = [occ for occ in occupations if occ.lower() in lowered_question]
    if direct_matches:
        return max(direct_matches, key=len)
    if occupations:
        from rapidfuzz import fuzz, process
        candidate, score, _ = process.extractOne(
            cleaned,
            occupations,
            scorer=fuzz.WRatio,
            score_cutoff=70,
        ) or (None, None, None)
        if candidate and score:
            return candidate
    return None


def _format_market_answer(states: list[dict], occupation: str | None = None) -> str:
    top = states[0]
    others = ", ".join(f"{row['state_name']} (${row['average_salary']:.0f})" for row in states[1:])
    lead = f"{top['state_name']} currently leads with an average salary of ${top['average_salary']:.0f}."
    answer = lead
    if others:
        answer += f" Other notable states include {others}."
    if occupation:
        answer = f"For {occupation}, {answer}"
    return answer


def _format_skill_answer(skills: list[dict]) -> str:
    highlights = ", ".join(f"{item['skill']}" for item in skills[:3])
    return f"Key skills to highlight are {highlights}."
    # Remove filler words and normalize
    filler_pattern = r"\b(what|is|the|for|in|of|a|an|to|on|at|by|with|from|and|or|please|show|tell|me|about|which|has|have|does|do|can|could|would|should|will|may|might|shall|who|whose|where|when|how|why|that|this|these|those|it|as|than|then|so|such|just|like|but|if|else|also|too|very|really|more|most|least|few|many|some|any|each|every|either|neither|both|all|no|not|nor|yes|yeah|yep|yup|uh|um|oh|okay|ok|alright|alrighty|hi|hello|hey|thanks|thank you|bye|goodbye|see you|talk later|good morning|good afternoon|good evening)\b"
    cleaned = re.sub(filler_pattern, "", lowered)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Extract occupation (normalize plural/singular)
    occupation = _extract_occupation(cleaned)
    if occupation and occupation.endswith("s") and not occupation.endswith("ss"):
        occupation = occupation[:-1]  # crude singularization

    # Extract state
    state = _extract_state(cleaned)

    # Metric intent
    if any(k in cleaned for k in ["salary", "average salary", "highest salary", "pay", "wage"]):
        intent_metric = "average_salary"
    elif any(k in cleaned for k in ["job openings", "job postings", "employment", "hiring"]):
        intent_metric = "total_employment"
    else:
        intent_metric = "average_salary"

    # Small-talk detection (keep existing logic)
    def _contains_any(src: str, keywords: list[str]) -> bool:
        return any(k in src for k in keywords)
    greeting_keywords = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    thanks_keywords = ["thanks", "thank you", "thx"]
    farewell_keywords = ["bye", "goodbye", "see you", "talk later"]
    how_are_you_keywords = ["how are you", "how're you", "how you doing", "how do you do"]
    job_keywords = [
        "salary", "average salary", "highest salary", "job openings", "job postings", "employment", "hiring",
        "skills", "skill", "insight", "position", "opening", "vacancy", "pay", "wage"
    ]
    if (_contains_any(lowered, greeting_keywords + thanks_keywords + farewell_keywords + how_are_you_keywords)
        and not _contains_any(lowered, job_keywords)
        and not occupation):
        if _contains_any(lowered, greeting_keywords):
            return ChatResponse(answer="Hi there! How can I help you with your career insights today?", payload=None)
        if _contains_any(lowered, how_are_you_keywords):
            return ChatResponse(answer="I'm a career assistant — ready to help with skills and market insights. What would you like to explore?", payload=None)
        if _contains_any(lowered, thanks_keywords):
            return ChatResponse(answer="You're welcome — anything else I can help with?", payload=None)
        if _contains_any(lowered, farewell_keywords):
            return ChatResponse(answer="Goodbye! Feel free to ask about jobs, salaries, or skills anytime.", payload=None)

    # --- Main market insight logic ---
    if turn.mode is ChatMode.MARKET:
        if not occupation:
            occupation = _extract_occupation(cleaned)
        if not state:
            state = _extract_state(cleaned)
        if not intent_metric:
            intent_metric = "average_salary"
        request = MarketInsightRequest(occupation=occupation, metric=intent_metric, top_n=60)
        insights = market_insights.top_states_for_occupation(
            request.occupation,
            request.metric,
            request.top_n,
        )
        if not insights:
            return ChatResponse(answer="No related data available — we’ll update this soon.", payload=None)
        if state:
            state_data = next((s for s in insights if s["state_name"].lower() == state.lower()), None)
            if state_data:
                if intent_metric == "average_salary":
                    answer = f"The average salary for {occupation} in {state} is ${state_data['average_salary']:,.0f}."
                else:
                    answer = f"Estimated employment for {occupation} in {state}: {int(state_data.get('total_employment', 0)):,}."
                return ChatResponse(answer=answer, payload={"states": insights}, sources=["OEWS state-level dataset"])
            else:
                return ChatResponse(answer="No relevant data found for this state. We’ll update it soon.", payload=None)
        answer = _format_market_answer(insights[:5], occupation=occupation)
        return ChatResponse(answer=answer, payload={"states": insights}, sources=["OEWS state-level dataset"])
    if turn.mode is ChatMode.SKILLS:
        skills = skills_analysis.extract_skills(turn.question)
        if not skills:
            raise HTTPException(status_code=404, detail="No skills detected")
        answer = _format_skill_answer(skills)
        return ChatResponse(
            answer=answer,
            payload={"skills": skills},
            sources=["Glassdoor company-level postings"],
        )
    raise HTTPException(status_code=400, detail="Unsupported mode")
    top = counts.head(top_n)
    data = [{"skill": skill, "count": int(count)} for skill, count in top.items()]

    # Load taxonomy (optional)
    taxonomy = {}
    try:
        tax_path = Path(__file__).resolve().parents[2] / "data" / "custom_tech_skills.json"
        if tax_path.exists():
            taxonomy = json.loads(tax_path.read_text())
    except Exception:
        taxonomy = {}

    by_category: dict[str, list[dict]] = {}
    if taxonomy:
        # For each category, find skills that contain any taxonomy keyword
        for cat, keywords in taxonomy.items():
            found: dict[str, int] = {}
            for kw in keywords:
                kw_low = kw.lower()
                for skill, cnt in counts.items():
                    if kw_low in skill.lower():
                        found[skill] = max(found.get(skill, 0), int(cnt))
            if found:
                # Sort descending by count
                items = [{"skill": s, "count": found[s]} for s in sorted(found, key=lambda x: -found[x])]
                by_category[cat] = items

    return {
        "data": data,
        "source": "glassdoor_extracted_skills.csv",
        "note": f"Top {top_n} skills by frequency",
        "by_category": by_category,
    }

    # --- Main market insight logic ---
    if turn.mode is ChatMode.MARKET:
        # Fallback to keyword-based extraction if NLP fails
        if not occupation:
            occupation = _extract_occupation(text)
        if not state:
            state = _extract_state(text)
        if not intent_metric:
            intent_metric = "average_salary"
        request = MarketInsightRequest(occupation=occupation, metric=intent_metric, top_n=60)
        insights = market_insights.top_states_for_occupation(
            request.occupation,
            request.metric,
            request.top_n,
        )
        if not insights:
            return ChatResponse(answer="No related data available — we’ll update this soon.", payload=None)
        if state:
            state_data = next((s for s in insights if s["state_name"].lower() == state.lower()), None)
            if state_data:
                if intent_metric == "average_salary":
                    answer = f"The average salary for {occupation} in {state} is ${state_data['average_salary']:,.0f}."
                else:
                    answer = f"Estimated employment for {occupation} in {state}: {int(state_data.get('total_employment', 0)):,}."
                return ChatResponse(answer=answer, payload={"states": insights}, sources=["OEWS state-level dataset"])
            else:
                return ChatResponse(answer="No relevant data found for this state. We’ll update it soon.", payload=None)
        answer = _format_market_answer(insights[:5], occupation=occupation)
        return ChatResponse(answer=answer, payload={"states": insights}, sources=["OEWS state-level dataset"])
    if turn.mode is ChatMode.SKILLS:
        skills = skills_analysis.extract_skills(turn.question)
        if not skills:
            raise HTTPException(status_code=404, detail="No skills detected")
        answer = _format_skill_answer(skills)
        return ChatResponse(
            answer=answer,
            payload={"skills": skills},
            sources=["Glassdoor company-level postings"],
        )
    raise HTTPException(status_code=400, detail="Unsupported mode")


@lru_cache(maxsize=1)
def _state_corpus() -> tuple[str, ...]:
    """Get a list of all state names from the OEWS dataset."""
    df = load_oews_data()
    states = df.get("state_name")
    if states is None:
        return ()
    unique = {str(item).strip() for item in states.dropna() if str(item).strip()}
    return tuple(sorted(unique, key=str.lower))


@lru_cache(maxsize=1)
def _occupation_corpus() -> tuple[str, ...]:
    df = load_oews_data()
    occupations = df.get("occupation")
    if occupations is None:
        return ()
    unique = {str(item).strip() for item in occupations.dropna() if str(item).strip()}
    return tuple(sorted(unique, key=str.lower))


def _extract_state(question: str) -> str | None:
    """Extract state name from question using fuzzy matching."""
    if not question:
        return None
    cleaned = question.strip().lower()
    states = _state_corpus()
    
    # Direct match first
    for state in states:
        if state.lower() in cleaned:
            return state
    
    # Fuzzy match if no direct match
    if states:
        from rapidfuzz import fuzz, process
        candidate, score, _ = process.extractOne(
            cleaned,
            states,
            scorer=fuzz.WRatio,
            score_cutoff=80,  # Higher threshold for states to avoid false matches
        ) or (None, None, None)
        if candidate and score:
            return candidate
    return None


def _extract_occupation(question: str) -> str | None:
    if not question:
        return None
    cleaned = question.strip()
    if not cleaned:
        return None
    occupations = _occupation_corpus()
    lowered_question = cleaned.lower()
    direct_matches = [occ for occ in occupations if occ.lower() in lowered_question]
    if direct_matches:
        return max(direct_matches, key=len)
    if occupations:
        from rapidfuzz import fuzz, process

        candidate, score, _ = process.extractOne(
            cleaned,
            occupations,
            scorer=fuzz.WRatio,
            score_cutoff=70,
        ) or (None, None, None)
        if candidate and score:
            return candidate
    return None


def extract_text_from_resume(file: UploadFile) -> str:
    """Extract plain text from an uploaded resume (PDF, DOCX, TXT).

    Uses PyPDF2 for PDFs, python-docx for DOCX, and falls back to plain text.
    """
    if file is None:
        return ""
    filename = (file.filename or "").lower()
    # Ensure pointer at start
    try:
        file.file.seek(0)
    except Exception:
        pass

    if filename.endswith(".pdf"):
        try:
            if PdfReader is None:
                raise RuntimeError("PyPDF2 is not installed")
            reader = PdfReader(file.file)
            text = "".join(page.extract_text() or "" for page in reader.pages)
            return text
        except Exception as exc:
            raise RuntimeError(f"Unable to parse PDF: {exc}") from exc
    elif filename.endswith(".docx"):
        try:
            import docx

            doc = docx.Document(file.file)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            raise
    else:
        # Plain text or unknown - attempt to decode
        data = file.file.read()
        try:
            return data.decode("utf-8")
        except Exception:
            return data.decode("latin-1", errors="ignore")


@app.post("/api/resume_match")
async def resume_match(job_description: str = Form(...), resume_file: UploadFile | None = File(None)) -> dict:
    """Accept a job description and optional resume file and return skill match stats.

    - job_description: required text
    - resume_file: optional uploaded file (PDF, DOCX, TXT)
    """
    resume_text = ""
    if resume_file:
        try:
            resume_text = extract_text_from_resume(resume_file)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Unable to extract resume text: {e}")

    # Use the local lightweight extractor to get in-demand skills
    skills = skills_analysis.extract_skills(job_description)
    in_demand = [s["skill"] for s in skills]

    result: dict = {"in_demand_skills": in_demand}
    if resume_text:
        match = skill_match_from_resume(resume_text, in_demand)
        result["resume_match"] = match

    return result


def _format_market_answer(states: list[dict], occupation: str | None = None) -> str:
    top = states[0]
    others = ", ".join(f"{row['state_name']} (${row['average_salary']:.0f})" for row in states[1:])
    lead = f"{top['state_name']} currently leads with an average salary of ${top['average_salary']:.0f}."
    answer = lead
    if others:
        answer += f" Other notable states include {others}."
    if occupation:
        answer = f"For {occupation}, {answer}"
    return answer


def _format_skill_answer(skills: list[dict]) -> str:
    highlights = ", ".join(f"{item['skill']}" for item in skills[:3])
    return f"Key skills to highlight are {highlights}."
