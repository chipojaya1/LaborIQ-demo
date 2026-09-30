# LaborIQ-demo

This repository contains a minimal end-to-end spike that links the LaborIQ chatbot backend with the Figma Make prototype.

LaborIQ-demo is a demo backend for a career-focused conversational assistant. It exposes a FastAPI service that answers user queries about salaries, employment, and job-skills. The service uses a mix of lightweight local heuristics, a Glassdoor-derived skill inventory, and optional OpenAI-powered skill extraction/insights.

## Project overview

- Provides conversational and programmatic access to state-level salary/employment data (OEWS / BLS).
- Offers a lightweight skill extraction flow (local fuzzy-matching) and an OpenAI-backed skill-extraction + insight generator.
- Exposes a unified `/api/chat` endpoint that routes small-talk, market-insight, and skill-analysis queries to the appropriate handlers.

## Repository structure
```
backend/
   app/
      main.py                 # FastAPI app, routes, and chat orchestration (intent detection + helpers)
      models/
         schemas.py            # Pydantic request/response models used by the API
      services/
         market_insights.py    # Helpers to compute top states for occupations and BLS macro trends
         skills_analysis.py    # Lightweight local skill extraction (fuzzy matching + curated synonyms)
         model_skill_analysis.py# OpenAI-powered skill extraction & insight generation (analyze_job_description)
      utils/
         data_loader.py        # Cached dataset loaders for OEWS, BLS, Glassdoor CSVs
   requirements.txt          # Python deps for the backend
data/                       # Dataset CSVs used by the demo (OEWS, BLS, Glassdoor files)
docs/                       # Integration notes and implementation docs (figma integration, etc.)
```
## What each Python module does

- `backend/app/main.py`
   - FastAPI application and route definitions. Implements endpoints: `/health`, `/api/chat`, `/api/skill_analysis`, `/api/insights`, `/api/skills`, `/api/bls/*`.
   - Contains helper functions for cleaning text, intent detection, fuzzy entity extraction (`_extract_state`, `_extract_occupation`), and response formatting.
   - Orchestrates which backend logic to call: small-talk reply, local skills extractor, OpenAI skill analysis, or market insights.

- `backend/app/models/schemas.py`
   - Pydantic models and enums used for request validation and response shapes (ChatTurn, ChatResponse, MarketInsightRequest/Response, SkillAnalysisRequest/Response, ChatMode).

- `backend/app/services/market_insights.py`
   - Functions to compute state-level rankings for occupations using OEWS data (`top_states_for_occupation`) and to return BLS macro trends (`macro_trend`).

- `backend/app/services/skills_analysis.py`
   - Lightweight local skill extraction using curated `SKILL_SYNONYMS`, fuzzy matching against a Glassdoor-derived skill inventory, and ranking logic (`extract_skills`).

- `backend/app/services/model_skill_analysis.py`
   - OpenAI-driven skill extraction and insight generator.
   - Functions: `extract_skills_openai`, `generate_insight_openai`, and `analyze_job_description` (top-level helper that creates an OpenAI client, extracts skills, and generates a short `chatbot_insight`).
   - Includes local fallbacks (taxonomy) and robust parsing logic for model outputs.

- `backend/app/utils/data_loader.py`
   - Cached data loaders that locate and normalize dataset CSVs: OEWS (`load_oews_data`), BLS (`load_bls_data`), Glassdoor jobs and skill inventory (`load_glassdoor_jobs`, `load_glassdoor_skill_inventory`, `load_glassdoor_skills`).
   - Raises `DataNotFoundError` when datasets are missing and normalizes column names/types.

Other package `__init__.py` files provide Python package semantics for importing.

## FastAPI endpoints (summary)

- `GET /health`
   - Basic health check.

- `POST /api/chat`  (body: `ChatTurn`)
   - Unified conversational entrypoint. Behavior:
      - Small-talk detection: returns short conversational replies (greetings, thanks) without calling analysis logic.
      - Skill analysis intent: calls `analyze_job_description(...)` (OpenAI client required) and returns a `chatbot_insight` summary and payload.
      - Market insight intent (salary/employment): extracts occupation/state and returns state-level salary or employment estimates using `market_insights` helpers.

- `POST /api/skill_analysis`  (body: `{job_description}`)
   - Runs `analyze_job_description` (OpenAI) and returns extracted skills, taxonomy categories, and a chatbot insight. Requires `OPENAI_API_KEY` environment variable.

- `POST /api/insights`  (body: `MarketInsightRequest`)
   - Returns `MarketInsightResponse` (top states for an occupation).

- `POST /api/skills`  (body: `SkillAnalysisRequest`)
   - Runs the lightweight local `extract_skills()` extractor and returns ranked skills.

- `GET /api/bls/unemployment_rate` and `GET /api/bls/{series_id}`
   - Return macro trends computed from `bls_final.csv`.

## Data processing summary

- Glassdoor: `data/glassdoor_final.csv` and `data/glassdoor_extracted_skills.csv` are used to build a skill inventory and few-shot examples for OpenAI prompts. The data loader merges job metadata and extracted-skill facts.
- OEWS: `data/oews_final.csv` supplies state-level occupational metrics (average_salary, median_salary, total_employment) used by `market_insights.top_states_for_occupation()`.
- BLS: `data/bls_final.csv` contains macro time-series; `load_bls_data()` normalizes and reshapes to long form for `macro_trend()`.

## Model & OpenAI integration

- Local extractor (`skills_analysis.py`) is fast and deterministic (no network calls). It uses `rapidfuzz` for fuzzy matching and curated skill synonyms.
- OpenAI-based pipeline (`model_skill_analysis.py`) builds few-shot prompts from Glassdoor examples and uses the OpenAI chat API to extract skills and generate a short career insight. The top-level helper `analyze_job_description()` requires `api_key` and returns a combined JSON with `raw_skills`, `categories`, and `chatbot_insight`.


## Usage / Run steps (quick)

1. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

2. Set environment variables (OpenAI key only required for OpenAI-backed endpoints):
```bash
export OPENAI_API_KEY="your_openai_key_here"
export PORT=8000
```

3. Start the server:
```bash
uvicorn backend.app.main:app --reload --port ${PORT:-8000}
```

4. Example requests:
```bash
curl -X GET http://localhost:8000/health

curl -X POST http://localhost:8000/api/chat -H 'Content-Type: application/json' -d \\
'{"question":"what is the average salary for data scientist in Texas","mode":"market_insights"}'

curl -X POST http://localhost:8000/api/skill_analysis -H 'Content-Type: application/json' -d '{"job_description":"We need a Data Scientist with Python, SQL, and TensorFlow experience."}'
```

## Notes & next steps

- Data files: The app expects CSVs under `data/` named like `*_final.csv`. The data loader will raise `DataNotFoundError` if these are absent.
- OpenAI usage: `model_skill_analysis.py` calls the OpenAI chat API; ensure `OPENAI_API_KEY` is set for those flows.
- Suggested improvements: add pytest tests for helpers, add more intent patterns, and optionally add a RAG index for richer retrieval context.
