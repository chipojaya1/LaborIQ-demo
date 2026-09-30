"""Pydantic schemas for the chatbot API."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ChatMode(str, Enum):
    MARKET = "market_insights"
    SKILLS = "skills_analysis"


class MarketInsightRequest(BaseModel):
    occupation: str = Field(..., description="Occupation query, e.g. 'Data Scientist'.")
    metric: str = Field(
        "average_salary",
        description=(
            "Metric to rank states by "
            "(average_salary, hourly_mean, average_salary_adjusted, hourly_mean_adjusted, "
            "median_salary, total_employment)."
        ),
    )
    top_n: int = Field(5, ge=1, le=60)


class MarketInsightResponse(BaseModel):
    states: List[dict]


class SkillAnalysisRequest(BaseModel):
    job_description: str = Field(..., description="Full job description text to analyse.")
    resume_text: Optional[str] = Field(None, description="Optional resume text to compare against in-demand skills.")
    top_n: int = Field(5, ge=1, le=20)


class SkillAnalysisResponse(BaseModel):
    skills: List[dict]


class ChatTurn(BaseModel):
    question: str
    mode: ChatMode


class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = None
    payload: Optional[dict] = None
