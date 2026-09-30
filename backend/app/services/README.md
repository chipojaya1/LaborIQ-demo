model_skill_analysis module

Location

- `backend/app/services/model_skill_analysis.py`

Overview

This module provides a lightweight OpenAI-backed skill extraction and insight generator that replaces the previous local fine-tuned pipeline.
It is designed to be called from a FastAPI endpoint (or used directly from other code). The module reads a small reference dataset for few-shot examples and a taxonomy JSON to classify skills.

Key functions

- `extract_skills_openai(user_input: str, client: OpenAI, dataset_path: str = "data/glassdoor_extracted_skills.csv", taxonomy_path: str = "data/custom_tech_skills.json", n_examples: int = 3) -> Dict`
  - Uses the OpenAI Chat API (gpt-4o-mini) with few-shot examples from `glassdoor_extracted_skills.csv` to extract `raw_skills` and (optionally) `categories`.
  - Falls back to a local taxonomy-based classifier if the model doesn't return categories.
  - Returns: `{"raw_skills": [...], "categories": {...}}`.

- `generate_insight_openai(user_input: str, combined_skills: List[str], client: OpenAI) -> Dict`
  - Calls the OpenAI Chat API to produce a short structured insight containing `summary`, `top_skills` and `recommendations`.
  - Returns: `{"summary": "...", "top_skills": [...], "recommendations": [...]}`.

- `analyze_job_description(user_input: str, api_key: Optional[str], dataset_path: str = "data/glassdoor_extracted_skills.csv", taxonomy_path: str = "data/custom_tech_skills.json") -> Dict`
  - Convenience wrapper that creates an OpenAI `client` and performs extraction + insight generation.
  - Returns the full structure:
    {
      "raw_skills": [...],
      "categories": {...},
      "chatbot_insight": {
        "summary": "...",
        "top_skills": [...],
        "recommendations": [...]
      }
    }

Default data paths

- `data/glassdoor_extracted_skills.csv` — used for few-shot examples (must contain `description` and `skills_list` columns).
- `data/custom_tech_skills.json` — taxonomy JSON mapping category keys to lists of skill strings.

Dependencies

- `openai` (the official `OpenAI` client package used in the repo)
- `pandas`

Install quick dependencies (in your virtualenv):

```bash
pip install openai pandas
```

Example usage (direct call from Python)

```python
from openai import OpenAI
from backend.app.services.model_skill_analysis import analyze_job_description

api_key = "sk-..."  # or load from env
client_payload = "We are hiring a data scientist with Python, SQL, TensorFlow and AWS experience."

result = analyze_job_description(client_payload, api_key=api_key)
print(result)
```

Example FastAPI endpoint (suggestion)

If you want to expose this as an HTTP endpoint, add a route in `backend/app/main.py` similar to:

```python
from openai import OpenAI
from fastapi import APIRouter, Depends
from backend.app.services.model_skill_analysis import analyze_job_description

router = APIRouter()

@router.post("/api/model-skill-analysis")
def model_skill_analysis_endpoint(payload: dict):
    # payload expected to contain `description` and optionally `api_key`
    description = payload.get("description")
    api_key = payload.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key required")
    out = analyze_job_description(description, api_key=api_key)
    return out
```

Example curl (if you expose the endpoint in FastAPI)

```bash
curl -X POST "https://<your-public-url>/api/model-skill-analysis" \
  -H "Content-Type: application/json" \
  -d '{"description": "We are hiring a data scientist with Python, SQL, TensorFlow and AWS experience.", "api_key":"sk-..."}' | jq .
```

Notes & tips

- Keep your OpenAI API key secret. Prefer passing it via environment variable in production (e.g. `OPENAI_API_KEY`).
- The module is resilient to model output formatting but best results come when the few-shot examples in the CSV are representative and well-formed.
- The local taxonomy classifier is intentionally simple (substring matching). If you need higher precision use fuzzy matching or a small vector search over normalized skill lexicon.
- This module is designed to be easy to replace later with a RAG/fine-tuning flow if desired.

Contact

If you need me to wire the suggested FastAPI route into `backend/app/main.py` or add unit tests / CI checks, tell me and I will add them next.
