# Repository Guidelines

## Project Structure & Module Organization
The demo centers on `backend/app`, a FastAPI project split into `models/` for Pydantic schemas, `services/` for analytics logic, and `utils/` for shared loaders. Add new API routes in `backend/app/main.py` and keep helper logic inside `services/`. Sample datasets live in `data/` (OEWS, BLS, Glassdoor CSVs) and integration notes sit under `docs/`. Avoid scattering notebooks—store exploratory work inside `backend/` and keep production code importable.

## Build, Test, and Development Commands
Create a local virtual environment and install backend dependencies before running anything:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```
Start the API with `uvicorn backend.app.main:app --reload`. Use `curl http://localhost:8000/health` to confirm the service is up. When datasets change, restart the server to clear the `lru_cache` used by the data loaders.

## Coding Style & Naming Conventions
Follow standard Python conventions: 4-space indentation, type hints, and docstrings for public functions (see `backend/app/main.py`). Keep modules small and cohesive; group related helpers into packages rather than single large files. Prefer descriptive snake_case names for variables and functions, and Title Case for enums and Pydantic models. Reuse existing loader patterns instead of hard-coding file paths.

## Testing Guidelines
There is no automated test harness yet, so validate changes with targeted exercises: hit `/health` and `/api/chat` using `curl` or a REST client, and confirm expected payloads. When adding analytics logic, craft minimal fixtures or sample payloads in `data/` and document the manual steps in your PR. If you introduce pytest, place tests under `tests/` and name files `test_<feature>.py`.

## Commit & Pull Request Guidelines
Commits follow short, imperative messages (e.g. `Use final dataset filenames in data loader`). Bundle related work and avoid mixing code and dataset refreshes unless necessary. Pull requests should summarize scope, list manual verification, and flag any dataset or API contract changes. Link relevant tickets and include screenshots or sample responses when behavior changes.

## Data Handling & Security Notes
Datasets are versioned CSVs; keep filenames aligned with the `*_final.csv` pattern or update `_DATASET_ALIASES` if you add variants. Do not commit proprietary data. CORS is permissive by design—review `backend/app/main.py` before tightening it for production environments.
